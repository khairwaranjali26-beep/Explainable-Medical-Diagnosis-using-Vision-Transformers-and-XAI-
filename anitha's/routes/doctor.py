from datetime import datetime
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, send_file
from config import Config
from database.db import query_db, execute_db
from routes.auth import login_required
from services.pdf_service import generate_medical_pdf

doctor_bp = Blueprint('doctor', __name__, url_prefix='/doctor')

@doctor_bp.route('/dashboard')
@login_required('doctor')
def dashboard():
    doctor_id = session['user_id']
    
    # Strictly query only reports explicitly shared with this doctor
    shared_reports = query_db(
        """
        SELECT rs.id as share_id, rs.shared_at, rs.patient_notes, rs.share_status,
               a.id as analysis_id, a.report_id, a.patient_name, a.predicted_class, 
               a.confidence, a.original_image_path, a.heatmap_path, a.overlay_path,
               dr.review_status, dr.clinical_verdict, dr.reviewed_at
        FROM report_shares rs
        JOIN analyses a ON rs.analysis_id = a.id
        LEFT JOIN doctor_reviews dr ON a.id = dr.analysis_id AND dr.doctor_id = %s
        WHERE rs.doctor_id = %s
        ORDER BY rs.shared_at DESC
        """, (doctor_id, doctor_id)
    )
    
    # Calculate stats
    total_shared = len(shared_reports)
    pending_count = sum(1 for r in shared_reports if r['share_status'] == 'Pending Review')
    reviewed_count = sum(1 for r in shared_reports if r['share_status'] == 'Reviewed')
    
    return render_template(
        'doctor/dashboard.html',
        shared_reports=shared_reports,
        total_shared=total_shared,
        pending_count=pending_count,
        reviewed_count=reviewed_count,
        disclaimer=Config.DISCLAIMER_TEXT
    )

@doctor_bp.route('/inspect/<report_id>')
@login_required('doctor')
def inspect_report(report_id):
    doctor_id = session['user_id']
    
    # Verify that this report is explicitly shared with this doctor
    share = query_db(
        """
        SELECT rs.*, u.gender as patient_gender, u.age as patient_age, u.phone as patient_phone, u.email as patient_email
        FROM report_shares rs
        JOIN analyses a ON rs.analysis_id = a.id
        JOIN users u ON rs.patient_id = u.id
        WHERE a.report_id = %s AND rs.doctor_id = %s
        """, (report_id, doctor_id), one=True
    )
    
    if not share:
        flash("Access Denied: You do not have permission to inspect this medical report as it was not explicitly shared with you.", "danger")
        return redirect(url_for('doctor.dashboard'))
        
    analysis = query_db("SELECT * FROM analyses WHERE report_id = %s", (report_id,), one=True)
    existing_review = query_db(
        "SELECT * FROM doctor_reviews WHERE analysis_id = %s AND doctor_id = %s",
        (analysis['id'], doctor_id), one=True
    )
    
    return render_template(
        'doctor/inspect.html',
        analysis=analysis,
        share=share,
        existing_review=existing_review,
        disclaimer=Config.DISCLAIMER_TEXT
    )

@doctor_bp.route('/submit-review/<report_id>', methods=['POST'])
@login_required('doctor')
def submit_review(report_id):
    doctor_id = session['user_id']
    doctor_name = session['user_name']
    doctor_specialty = session.get('user_specialty', 'Consultant Physician')
    
    analysis = query_db("SELECT * FROM analyses WHERE report_id = %s", (report_id,), one=True)
    if not analysis:
        flash("Analysis not found.", "danger")
        return redirect(url_for('doctor.dashboard'))
        
    # Check share authorization
    share = query_db(
        "SELECT * FROM report_shares WHERE analysis_id = %s AND doctor_id = %s",
        (analysis['id'], doctor_id), one=True
    )
    if not share:
        flash("Unauthorized review attempt.", "danger")
        return redirect(url_for('doctor.dashboard'))
        
    verdict = request.form.get('verdict', '').strip()
    notes = request.form.get('notes', '').strip()
    recommendations = request.form.get('recommendations', '').strip()
    
    if not verdict or not notes:
        flash("Please provide a clinical verdict and medical notes.", "warning")
        return redirect(url_for('doctor.inspect_report', report_id=report_id))
        
    existing_review = query_db(
        "SELECT id FROM doctor_reviews WHERE analysis_id = %s AND doctor_id = %s",
        (analysis['id'], doctor_id), one=True
    )
    
    if existing_review:
        execute_db(
            """
            UPDATE doctor_reviews 
            SET clinical_verdict = %s, clinical_notes = %s, recommendations = %s, review_status = 'Reviewed', reviewed_at = %s
            WHERE id = %s
            """,
            (verdict, notes, recommendations, datetime.now(), existing_review['id'])
        )
    else:
        execute_db(
            """
            INSERT INTO doctor_reviews (
                analysis_id, doctor_id, doctor_name, doctor_specialty, review_status,
                clinical_verdict, clinical_notes, recommendations, reviewed_at
            ) VALUES (%s, %s, %s, %s, 'Reviewed', %s, %s, %s, %s)
            """,
            (analysis['id'], doctor_id, doctor_name, doctor_specialty, verdict, notes, recommendations, datetime.now())
        )
        
    # Update share record to 'Reviewed'
    execute_db(
        "UPDATE report_shares SET share_status = 'Reviewed' WHERE id = %s",
        (share['id'],)
    )
    
    # Audit log
    execute_db(
        "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
        (doctor_id, session['user_email'], 'Doctor Review', f"Reviewed report {report_id} with verdict: '{verdict}'")
    )
    
    flash(f"Clinical review for report {report_id} successfully submitted and patient record updated.", "success")
    return redirect(url_for('doctor.dashboard'))

@doctor_bp.route('/download-pdf/<report_id>')
@login_required('doctor')
def download_pdf(report_id):
    doctor_id = session['user_id']
    share = query_db(
        "SELECT a.*, u.gender as patient_gender, u.age as patient_age "
        "FROM report_shares rs JOIN analyses a ON rs.analysis_id = a.id "
        "JOIN users u ON a.patient_id = u.id "
        "WHERE a.report_id = %s AND rs.doctor_id = %s",
        (report_id, doctor_id), one=True
    )
    if not share:
        flash("Report not found or not shared with you.", "danger")
        return redirect(url_for('doctor.dashboard'))
        
    doctor_review = query_db(
        "SELECT * FROM doctor_reviews WHERE analysis_id = %s AND doctor_id = %s", (share['id'], doctor_id), one=True
    )
    
    try:
        pdf_path = generate_medical_pdf(share, doctor_review)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f"MediLens_Clinical_Report_{report_id}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"[Doctor PDF Download Error]: {e}")
        flash(f"Could not generate PDF: {str(e)}", "danger")
        return redirect(url_for('doctor.inspect_report', report_id=report_id))
