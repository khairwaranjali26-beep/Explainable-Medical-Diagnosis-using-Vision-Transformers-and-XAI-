-- =====================================================================
-- MediLens AI – Database Schema (MySQL & SQLite Compatible)
-- B.Tech CSE (AI & ML) Capstone Project
-- =====================================================================

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'patient', -- 'patient', 'doctor', 'admin'
    specialty VARCHAR(100) DEFAULT NULL,
    license_no VARCHAR(100) DEFAULT NULL,
    phone VARCHAR(30) DEFAULT NULL,
    gender VARCHAR(20) DEFAULT 'Not specified',
    age INT DEFAULT 30,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    report_id VARCHAR(50) UNIQUE NOT NULL,
    patient_id INT NOT NULL,
    patient_name VARCHAR(120) NOT NULL,
    original_image_path VARCHAR(255) NOT NULL,
    heatmap_path VARCHAR(255) NOT NULL,
    overlay_path VARCHAR(255) NOT NULL,
    patch_grid_path VARCHAR(255) DEFAULT NULL,
    predicted_class VARCHAR(50) NOT NULL, -- 'Normal', 'Pneumonia'
    confidence FLOAT NOT NULL,
    normal_prob FLOAT NOT NULL,
    pneumonia_prob FLOAT NOT NULL,
    findings_summary TEXT,
    affected_regions TEXT,
    model_type VARCHAR(100) DEFAULT 'Vision Transformer (ViT-B/16 Custom)',
    model_status VARCHAR(50) DEFAULT 'Active Inference',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(patient_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS doctor_reviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id INT NOT NULL,
    doctor_id INT NOT NULL,
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

CREATE TABLE IF NOT EXISTS report_shares (
    id INT AUTO_INCREMENT PRIMARY KEY,
    analysis_id INT NOT NULL,
    patient_id INT NOT NULL,
    doctor_id INT NOT NULL,
    patient_notes TEXT,
    share_status VARCHAR(50) DEFAULT 'Pending Review', -- 'Pending Review', 'Reviewed'
    shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE,
    FOREIGN KEY(patient_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY(doctor_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS system_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT DEFAULT NULL,
    user_email VARCHAR(120) DEFAULT NULL,
    action VARCHAR(100) NOT NULL,
    details TEXT,
    ip_address VARCHAR(50) DEFAULT '127.0.0.1',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
