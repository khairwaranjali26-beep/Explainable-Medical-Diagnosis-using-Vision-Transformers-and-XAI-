import os
import sqlite3
import pymysql
from pymysql.cursors import DictCursor
from config import Config

def get_db_connection():
    """
    Returns a database connection.
    Attempts MySQL first if configured or available, otherwise falls back seamlessly to SQLite.
    """
    if Config.USE_MYSQL:
        try:
            conn = pymysql.connect(
                host=Config.MYSQL_HOST,
                port=Config.MYSQL_PORT,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                cursorclass=DictCursor,
                autocommit=True
            )
            return conn, 'mysql'
        except Exception as e:
            print(f"[DB Warning] MySQL connection failed ({e}). Falling back to SQLite.")
    
    # SQLite connection
    os.makedirs(os.path.dirname(Config.SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn, 'sqlite'

def query_db(query, args=(), one=False):
    """
    Executes a SELECT query and returns list of dicts or single dict.
    """
    conn, engine = get_db_connection()
    try:
        cur = conn.cursor()
        
        clean_args = []
        for a in args:
            if hasattr(a, 'isoformat'):
                clean_args.append(a.isoformat())
            else:
                clean_args.append(a)

        # Replace %s with ? for SQLite if needed
        formatted_query = query
        if engine == 'sqlite':
            formatted_query = query.replace('%s', '?')
            
        cur.execute(formatted_query, clean_args)
        
        if engine == 'sqlite':
            rows = cur.fetchall()
            rv = [dict(row) for row in rows]
        else:
            rv = cur.fetchall()
            
        cur.close()
        conn.close()
        return (rv[0] if rv else None) if one else rv
    except Exception as e:
        conn.close()
        print(f"[DB Query Error]: {e} in query: {query}")
        raise e

def execute_db(query, args=()):
    """
    Executes INSERT, UPDATE, DELETE queries.
    Returns the lastrowid or affected rows.
    """
    conn, engine = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Format args for sqlite if needed
        clean_args = []
        for a in args:
            if hasattr(a, 'isoformat'):
                clean_args.append(a.isoformat())
            else:
                clean_args.append(a)
                
        formatted_query = query
        if engine == 'sqlite':
            formatted_query = query.replace('%s', '?')
            
        cur.execute(formatted_query, clean_args)
        
        if engine == 'sqlite':
            conn.commit()
            last_id = cur.lastrowid
        else:
            last_id = cur.lastrowid
            
        cur.close()
        conn.close()
        return last_id
    except Exception as e:
        conn.close()
        print(f"[DB Execute Error]: {e} in query: {query}")
        raise e

def init_db():
    """
    Initializes the database schema for users, analyses, doctor reviews, shares, and audit logs.
    """
    conn, engine = get_db_connection()
    cur = conn.cursor()
    
    # Common schema statements with standard types
    statements = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(120) NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) NOT NULL DEFAULT 'patient',
            specialty VARCHAR(100) DEFAULT NULL,
            license_no VARCHAR(100) DEFAULT NULL,
            phone VARCHAR(30) DEFAULT NULL,
            gender VARCHAR(20) DEFAULT 'Not specified',
            age INTEGER DEFAULT 30,
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id VARCHAR(50) UNIQUE NOT NULL,
            patient_id INTEGER NOT NULL,
            patient_name VARCHAR(120) NOT NULL,
            original_image_path VARCHAR(255) NOT NULL,
            heatmap_path VARCHAR(255) NOT NULL,
            overlay_path VARCHAR(255) NOT NULL,
            patch_grid_path VARCHAR(255) DEFAULT NULL,
            predicted_class VARCHAR(50) NOT NULL,
            confidence REAL NOT NULL,
            normal_prob REAL NOT NULL,
            pneumonia_prob REAL NOT NULL,
            findings_summary TEXT,
            affected_regions TEXT,
            model_type VARCHAR(100) DEFAULT 'Vision Transformer (ViT-B/16 Custom)',
            model_status VARCHAR(50) DEFAULT 'Active Inference',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS doctor_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            doctor_name VARCHAR(120) NOT NULL,
            doctor_specialty VARCHAR(100),
            review_status VARCHAR(50) DEFAULT 'Reviewed',
            clinical_verdict VARCHAR(100) NOT NULL,
            clinical_notes TEXT,
            recommendations TEXT,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
            FOREIGN KEY(doctor_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS report_shares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL,
            patient_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            patient_notes TEXT,
            share_status VARCHAR(50) DEFAULT 'Pending Review',
            shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
            FOREIGN KEY(patient_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(doctor_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT NULL,
            user_email VARCHAR(120) DEFAULT NULL,
            action VARCHAR(100) NOT NULL,
            details TEXT,
            ip_address VARCHAR(50) DEFAULT '127.0.0.1',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]
    
    if engine == 'mysql':
        # Adjust AUTOINCREMENT for MySQL
        for stmt in statements:
            mysql_stmt = stmt.replace('INTEGER PRIMARY KEY AUTOINCREMENT', 'INT AUTO_INCREMENT PRIMARY KEY')
            cur.execute(mysql_stmt)
    else:
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
        
    cur.close()
    conn.close()
    print(f"[DB] Initialized database schema successfully ({engine} mode).")
