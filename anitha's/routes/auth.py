from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from database.db import query_db, execute_db

auth_bp = Blueprint('auth', __name__)

def login_required(roles=None):
    """
    Decorator for enforcing authentication and role-based access control (RBAC).
    roles can be a single role string (e.g. 'doctor') or a list/tuple of allowed roles.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash("Please log in to access this page.", "warning")
                return redirect(url_for('auth.login', next=request.url))
            
            user_role = session.get('user_role')
            if roles is not None:
                allowed_roles = [roles] if isinstance(roles, str) else roles
                if user_role not in allowed_roles:
                    flash(f"Access denied. You do not have permissions to view this section ({user_role} role).", "danger")
                    # Redirect to appropriate dashboard
                    if user_role == 'patient':
                        return redirect(url_for('patient.dashboard'))
                    elif user_role == 'doctor':
                        return redirect(url_for('doctor.dashboard'))
                    elif user_role == 'admin':
                        return redirect(url_for('admin.dashboard'))
                    return redirect(url_for('auth.login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        selected_role = request.form.get('role', '')

        if not email or not password:
            flash("Please provide both email address and password.", "danger")
            return render_template('auth/login.html')

        user = query_db("SELECT * FROM users WHERE email = %s", (email,), one=True)
        if not user or not check_password_hash(user['password_hash'], password):
            flash("Invalid email or password. Please try again.", "danger")
            return render_template('auth/login.html')

        if user['status'] != 'active':
            flash("This account is currently inactive. Please contact the administrator.", "danger")
            return render_template('auth/login.html')

        # Role verification if role was chosen
        if selected_role and user['role'] != selected_role:
            flash(f"Account exists, but its registered role is '{user['role']}', not '{selected_role}'.", "warning")
            return render_template('auth/login.html')

        # Set session
        session['user_id'] = user['id']
        session['user_name'] = user['name']
        session['user_email'] = user['email']
        session['user_role'] = user['role']
        session['user_specialty'] = user.get('specialty')

        # Audit log
        execute_db(
            "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
            (user['id'], user['email'], 'User Login', f"Logged in with role: {user['role']}")
        )

        flash(f"Welcome back, {user['name']}!", "success")

        # Redirect based on role
        next_url = request.args.get('next')
        if next_url:
            return redirect(next_url)

        if user['role'] == 'patient':
            return redirect(url_for('patient.dashboard'))
        elif user['role'] == 'doctor':
            return redirect(url_for('doctor.dashboard'))
        elif user['role'] == 'admin':
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('index'))

    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role = request.form.get('role', 'patient')
        phone = request.form.get('phone', '').strip()
        gender = request.form.get('gender', 'Not specified')
        age = request.form.get('age', 30)
        
        # Doctor specific fields
        specialty = request.form.get('specialty', '').strip()
        license_no = request.form.get('license_no', '').strip()

        if not name or not email or not password:
            flash("All required fields must be filled.", "danger")
            return render_template('auth/register.html')

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template('auth/register.html')

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "warning")
            return render_template('auth/register.html')

        existing = query_db("SELECT id FROM users WHERE email = %s", (email,), one=True)
        if existing:
            flash("An account with this email address already exists. Please log in.", "warning")
            return render_template('auth/register.html')

        password_hash = generate_password_hash(password)
        
        user_id = execute_db(
            """
            INSERT INTO users (name, email, password_hash, role, specialty, license_no, phone, gender, age, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
            """,
            (name, email, password_hash, role, specialty if role == 'doctor' else None, license_no if role == 'doctor' else None, phone, gender, age)
        )

        execute_db(
            "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
            (user_id, email, 'User Registration', f"Registered as {role}")
        )

        flash("Registration successful! You can now log in with your credentials.", "success")
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')

@auth_bp.route('/logout')
def logout():
    user_id = session.get('user_id')
    user_email = session.get('user_email')
    if user_id:
        execute_db(
            "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
            (user_id, user_email, 'User Logout', 'Session terminated by user.')
        )
    session.clear()
    flash("You have been securely logged out.", "info")
    return redirect(url_for('auth.login'))
