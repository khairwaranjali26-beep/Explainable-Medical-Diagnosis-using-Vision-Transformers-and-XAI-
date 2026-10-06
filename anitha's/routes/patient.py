import os
import uuid
import shutil
from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file, jsonify
from werkzeug.utils import secure_filename
from config import Config
from database.db import query_db, execute_db
from models.vit_model import ViTDiagnosticEngine
from routes.auth import login_required
from services.pdf_service import generate_medical_pdf

patient_bp = Blueprint('patient', __name__, url_prefix='/patient')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@patient_bp.route('/dashboard')
@login_required('patient')
def dashboard():
    user_id = session['user_id']
    
    # 1. Total analyses
    total_analyses = query_db(
        "SELECT COUNT(*) as count FROM analyses WHERE patient_id = %s", (user_id,), one=True
    )['count']
    
    # 2. Shared reports count
    total_shared = query_db(
        "SELECT COUNT(*) as count FROM report_shares WHERE patient_id = %s", (user_id,), one=True
    )['count']
    
    # 3. Doctor reviews completed
    total_reviewed = query_db(
        """
        SELECT COUNT(DISTINCT a.id) as count 
        FROM analyses a
        JOIN doctor_reviews dr ON a.id = dr.analysis_id
        WHERE a.patient_id = %s
        """, (user_id,), one=True
    )['count']
    
    # 4. Recent analyses
    recent_analyses = query_db(
        """
        SELECT a.*, 
               dr.review_status as doc_status,
               dr.doctor_name,
               rs.share_status
        FROM analyses a
        LEFT JOIN doctor_reviews dr ON a.id = dr.analysis_id
        LEFT JOIN report_shares rs ON a.id = rs.analysis_id
        WHERE a.patient_id = %s
        ORDER BY a.created_at DESC LIMIT 5
        """, (user_id,)
    )
    
    # 5. Notifications
    notifications = []
    # Notifications for doctor reviews
    reviewed_items = query_db(
        """
        SELECT dr.doctor_name, dr.clinical_verdict, dr.reviewed_at, a.report_id
        FROM doctor_reviews dr
        JOIN analyses a ON dr.analysis_id = a.id
        WHERE a.patient_id = %s
        ORDER BY dr.reviewed_at DESC LIMIT 3
        """, (user_id,)
    )
    for r in reviewed_items:
        notifications.append({
            'title': f"Clinical Review Completed for {r['report_id']}",
            'message': f"{r['doctor_name']} reviewed your chest scan: '{r['clinical_verdict']}'.",
            'time': str(r['reviewed_at'])[:16],
            'type': 'success',
            'report_id': r['report_id']
        })
        
    if total_analyses == 0:
        notifications.append({
            'title': "Welcome to MediLens AI",
            'message': "Upload your first chest radiograph or try the sample normal/pneumonia scans to begin.",
            'time': "Just now",
            'type': 'info',
            'report_id': None
        })

    return render_template(
        'patient/dashboard.html',
        total_analyses=total_analyses,
        total_shared=total_shared,
        total_reviewed=total_reviewed,
        recent_analyses=recent_analyses,
        notifications=notifications,
        disclaimer=Config.DISCLAIMER_TEXT
    )

@patient_bp.route('/upload')
@login_required('patient')
def upload_page():
    return render_template('patient/upload.html', disclaimer=Config.DISCLAIMER_TEXT)

@patient_bp.route('/analyze', methods=['POST'])
@login_required('patient')
def analyze_image():
    user_id = session['user_id']
    patient = query_db("SELECT * FROM users WHERE id = %s", (user_id,), one=True)
    
    # Check if a sample image was chosen
    sample_type = request.form.get('sample_type')
    file = request.files.get('image_file')
    
    report_id = f"ML-{datetime.now().year}-{str(uuid.uuid4())[:8].upper()}"
    target_filename = f"{report_id}_orig.jpg"
    target_path = os.path.join(Config.UPLOAD_FOLDER, target_filename)
    
    if sample_type in ('sample_normal', 'sample_pneumonia'):
        src_filename = 'sample_normal_chest_xray.jpg' if sample_type == 'sample_normal' else 'sample_pneumonia_chest_xray.jpg'
        src_path = os.path.join(Config.SAMPLE_FOLDER, src_filename)
        
        if not os.path.exists(src_path):
            from services.sample_generator import generate_sample_xrays
            generate_sample_xrays()
            
        shutil.copyfile(src_path, target_path)
    elif file and file.filename != '':
        if not allowed_file(file.filename):
            flash("Invalid file format. Please upload a JPG, JPEG, or PNG chest radiograph.", "danger")
            return redirect(url_for('patient.upload_page'))
            
        file.save(target_path)
    else:
        flash("Please select a chest radiograph image file or choose a demonstration sample.", "warning")
        return redirect(url_for('patient.upload_page'))
        
    try:
        # Run Vision Transformer & XAI Attention Rollout
        engine = ViTDiagnosticEngine.get_instance()
        res = engine.analyze(target_path, Config.UPLOAD_FOLDER, report_id)
        
        # Save to database
        rel_orig = f"uploads/{target_filename}"
        rel_heat = f"uploads/{res['heatmap_file']}"
        rel_over = f"uploads/{res['overlay_file']}"
        rel_patch = f"uploads/{res['patch_file']}"
        
        execute_db(
            """
            INSERT INTO analyses (
                report_id, patient_id, patient_name, original_image_path, heatmap_path, overlay_path,
                patch_grid_path, predicted_class, confidence, normal_prob, pneumonia_prob,
                findings_summary, affected_regions, model_type, model_status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                report_id, user_id, patient['name'],
                rel_orig, rel_heat, rel_over, rel_patch,
                res['predicted_class'], res['confidence'], res['normal_prob'], res['pneumonia_prob'],
                res['findings_summary'], res['affected_regions'],
                res['model_type'], res['model_status'], datetime.now()
            )
        )
        
        # Audit log
        execute_db(
            "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
            (user_id, patient['email'], 'ViT Analysis', f"Generated analysis for {report_id} ({res['predicted_class']} - {res['confidence']}%)")
        )
        
        flash(f"Analysis completed successfully! Report ID: {report_id}", "success")
        return redirect(url_for('patient.results_page', report_id=report_id))
        
    except Exception as e:
        print(f"[Analysis Error]: {e}")
        flash(f"Error during AI analysis: {str(e)}", "danger")
        return redirect(url_for('patient.upload_page'))

@patient_bp.route('/report/<report_id>')
@login_required('patient')
def results_page(report_id):
    user_id = session['user_id']
    analysis = query_db(
        "SELECT * FROM analyses WHERE report_id = %s AND patient_id = %s",
        (report_id, user_id), one=True
    )
    if not analysis:
        flash("Report not found or unauthorized.", "danger")
        return redirect(url_for('patient.history_page'))
        
    doctor_review = query_db(
        "SELECT * FROM doctor_reviews WHERE analysis_id = %s", (analysis['id'],), one=True
    )
    
    # Get active doctors for sharing
    doctors = query_db("SELECT id, name, specialty, license_no FROM users WHERE role = 'doctor' AND status = 'active'")
    
    # Check if currently shared
    share_record = query_db(
        "SELECT rs.*, u.name as doctor_name FROM report_shares rs JOIN users u ON rs.doctor_id = u.id WHERE rs.analysis_id = %s",
        (analysis['id'],), one=True
    )

    return render_template(
        'patient/results.html',
        analysis=analysis,
        doctor_review=doctor_review,
        doctors=doctors,
        share_record=share_record,
        disclaimer=Config.DISCLAIMER_TEXT
    )

@patient_bp.route('/explanation/<report_id>')
@login_required('patient')
def explanation_page(report_id):
    user_id = session['user_id']
    analysis = query_db(
        "SELECT * FROM analyses WHERE report_id = %s AND patient_id = %s",
        (report_id, user_id), one=True
    )
    if not analysis:
        flash("Analysis not found.", "danger")
        return redirect(url_for('patient.history_page'))

    return render_template(
        'patient/explanation.html',
        analysis=analysis,
        disclaimer=Config.DISCLAIMER_TEXT
    )

@patient_bp.route('/history')
@login_required('patient')
def history_page():
    user_id = session['user_id']
    filter_class = request.args.get('filter_class', '')
    
    query = """
        SELECT a.*, 
               dr.review_status as doc_status,
               dr.doctor_name,
               dr.clinical_verdict,
               rs.share_status
        FROM analyses a
        LEFT JOIN doctor_reviews dr ON a.id = dr.analysis_id
        LEFT JOIN report_shares rs ON a.id = rs.analysis_id
        WHERE a.patient_id = %s
    """
    params = [user_id]
    
    if filter_class:
        query += " AND a.predicted_class = %s"
        params.append(filter_class)
        
    query += " ORDER BY a.created_at DESC"
    
    analyses = query_db(query, tuple(params))
    
    return render_template(
        'patient/history.html',
        analyses=analyses,
        filter_class=filter_class,
        disclaimer=Config.DISCLAIMER_TEXT
    )

@patient_bp.route('/share/<report_id>', methods=['POST'])
@login_required('patient')
def share_report(report_id):
    user_id = session['user_id']
    analysis = query_db("SELECT * FROM analyses WHERE report_id = %s AND patient_id = %s", (report_id, user_id), one=True)
    if not analysis:
        flash("Invalid report.", "danger")
        return redirect(url_for('patient.history_page'))
        
    doctor_id = request.form.get('doctor_id')
    patient_notes = request.form.get('patient_notes', '').strip()
    
    if not doctor_id:
        flash("Please select a licensed doctor to share your report with.", "warning")
        return redirect(url_for('patient.results_page', report_id=report_id))
        
    # Check if already shared
    existing = query_db("SELECT id FROM report_shares WHERE analysis_id = %s AND doctor_id = %s", (analysis['id'], doctor_id), one=True)
    if existing:
        execute_db(
            "UPDATE report_shares SET patient_notes = %s, share_status = 'Pending Review', shared_at = %s WHERE id = %s",
            (patient_notes, datetime.now(), existing['id'])
        )
    else:
        execute_db(
            """
            INSERT INTO report_shares (analysis_id, patient_id, doctor_id, patient_notes, share_status, shared_at)
            VALUES (%s, %s, %s, %s, 'Pending Review', %s)
            """,
            (analysis['id'], user_id, doctor_id, patient_notes, datetime.now())
        )
        
    doc = query_db("SELECT name FROM users WHERE id = %s", (doctor_id,), one=True)
    doc_name = doc['name'] if doc else "Selected Doctor"
    flash(f"Report {report_id} has been securely shared with {doc_name} for clinical consultation.", "success")
    return redirect(url_for('patient.results_page', report_id=report_id))

@patient_bp.route('/download-pdf/<report_id>')
@login_required('patient')
def download_pdf(report_id):
    user_id = session['user_id']
    analysis = query_db(
        "SELECT a.*, u.gender as patient_gender, u.age as patient_age "
        "FROM analyses a JOIN users u ON a.patient_id = u.id "
        "WHERE a.report_id = %s AND a.patient_id = %s",
        (report_id, user_id), one=True
    )
    if not analysis:
        flash("Report not found.", "danger")
        return redirect(url_for('patient.history_page'))
        
    doctor_review = query_db(
        "SELECT * FROM doctor_reviews WHERE analysis_id = %s", (analysis['id'],), one=True
    )
    
    try:
        pdf_path = generate_medical_pdf(analysis, doctor_review)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"MediLens_Report_{report_id}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"[PDF Download Error]: {e}")
        flash(f"Could not generate PDF report: {str(e)}", "danger")
        return redirect(url_for('patient.results_page', report_id=report_id))
