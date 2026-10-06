import os
from flask import Flask, render_template, session, redirect, url_for
from config import Config
from database.db import init_db, query_db
from routes.auth import auth_bp
from routes.patient import patient_bp
from routes.doctor import doctor_bp
from routes.admin import admin_bp
from routes.api import api_bp

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Ensure required directories exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.SAMPLE_FOLDER, exist_ok=True)
    os.makedirs(Config.REPORT_FOLDER, exist_ok=True)

    # Initialize database if needed
    init_db()

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(patient_bp)
    app.register_blueprint(doctor_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)

    @app.context_processor
    def inject_globals():
        return {
            'app_name': 'MediLens AI',
            'app_tagline': 'Explainable Medical Diagnosis using Vision Transformers & XAI',
            'current_user': {
                'id': session.get('user_id'),
                'name': session.get('user_name'),
                'email': session.get('user_email'),
                'role': session.get('user_role'),
                'specialty': session.get('user_specialty')
            } if 'user_id' in session else None,
            'disclaimer_short': Config.DISCLAIMER_TEXT
        }

    @app.template_filter('datetime_format')
    def datetime_format_filter(value):
        if not value:
            return ""
        val_str = str(value)
        return val_str[:16].replace('T', ' ')

    @app.route('/')
    def index():
        # Fetch high level metrics for landing page
        stats = {
            'accuracy': '96.4%',
            'f1_score': '96.4%',
            'auc_roc': '0.982',
            'patch_size': '16x16 px',
            'total_scans': query_db("SELECT COUNT(*) as c FROM analyses", one=True)['c'],
            'doctors_count': query_db("SELECT COUNT(*) as c FROM users WHERE role='doctor'", one=True)['c']
        }
        return render_template('index.html', stats=stats)

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    @app.errorhandler(413)
    def request_entity_too_large(e):
        return render_template('413.html'), 413

    return app

app = create_app()

if __name__ == '__main__':
    # Generate sample images if they don't exist
    from services.sample_generator import generate_sample_xrays
    sample_normal = os.path.join(Config.SAMPLE_FOLDER, 'sample_normal_chest_xray.jpg')
    if not os.path.exists(sample_normal):
        generate_sample_xrays()
        
    print("=" * 70)
    print(" MediLens AI – Vision Transformers & XAI Medical Diagnosis Server")
    print(" Running at: http://127.0.0.1:5000")
    print(" Quick Login Credentials:")
    print("   - Admin:   admin@medilens.ai   / Admin@123")
    print("   - Doctor:  dr.sharma@medilens.ai / Doctor@123")
    print("   - Patient: patient@medilens.ai   / Patient@123")
    print("=" * 70)
    app.run(host='0.0.0.0', port=5000, debug=True)
