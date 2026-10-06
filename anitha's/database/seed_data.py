import os
import sys
import shutil
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from database.db import init_db, execute_db, query_db
from models.vit_model import ViTDiagnosticEngine

def seed():
    """
    Seeds database with initial users, sample analyses, doctor reviews, and report shares.
    """
    print("[Seed] Initializing database schema...")
    init_db()
    
    # Check if admin already exists
    existing_admin = query_db("SELECT id FROM users WHERE email = %s", ('admin@medilens.ai',), one=True)
    if existing_admin:
        print("[Seed] Database already seeded. Skipping initial seeding.")
        return
        
    print("[Seed] Creating standard users...")
    users = [
        ('Administrator', 'admin@medilens.ai', generate_password_hash('Admin@123'), 'admin', 'System Director', 'MED-ADM-001', '+91 98765 00001', 'Other', 40),
        ('Dr. Rajesh Sharma', 'dr.sharma@medilens.ai', generate_password_hash('Doctor@123'), 'doctor', 'MD Pulmonology, Senior Consultant', 'KMC-84920-P', '+91 98765 00002', 'Male', 46),
        ('Dr. Priya Patel', 'dr.priya@medilens.ai', generate_password_hash('Doctor@123'), 'doctor', 'DNB Radiodiagnosis & Interventional Radiology', 'MMC-61204-R', '+91 98765 00003', 'Female', 39),
        ('Anitha Reddy', 'patient@medilens.ai', generate_password_hash('Patient@123'), 'patient', None, None, '+91 98765 00004', 'Female', 34),
        ('Johnathan Miller', 'patient2@medilens.ai', generate_password_hash('Patient@123'), 'patient', None, None, '+91 98765 00005', 'Male', 48)
    ]
    
    user_ids = {}
    for name, email, pw, role, spec, lic, phone, gender, age in users:
        uid = execute_db(
            """
            INSERT INTO users (name, email, password_hash, role, specialty, license_no, phone, gender, age, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
            """,
            (name, email, pw, role, spec, lic, phone, gender, age)
        )
        user_ids[email] = uid
        
    print(f"[Seed] Created {len(user_ids)} users successfully.")
    
    # Run ViT model on the sample images to generate realistic initial analysis records
    print("[Seed] Generating initial diagnostic analyses...")
    engine = ViTDiagnosticEngine.get_instance()
    
    sample_normal = os.path.join(Config.SAMPLE_FOLDER, 'sample_normal_chest_xray.jpg')
    sample_pneu = os.path.join(Config.SAMPLE_FOLDER, 'sample_pneumonia_chest_xray.jpg')
    
    if not os.path.exists(sample_pneu):
        from services.sample_generator import generate_sample_xrays
        generate_sample_xrays()
        
    # 1. Pneumonia Analysis for Anitha Reddy
    report_id_1 = "ML-2026-8491"
    target_orig_1 = os.path.join(Config.UPLOAD_FOLDER, f"{report_id_1}_orig.jpg")
    shutil.copyfile(sample_pneu, target_orig_1)
    
    res1 = engine.analyze(target_orig_1, Config.UPLOAD_FOLDER, report_id_1)
    
    analysis_id_1 = execute_db(
        """
        INSERT INTO analyses (
            report_id, patient_id, patient_name, original_image_path, heatmap_path, overlay_path,
            patch_grid_path, predicted_class, confidence, normal_prob, pneumonia_prob,
            findings_summary, affected_regions, model_type, model_status, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            report_id_1, user_ids['patient@medilens.ai'], 'Anitha Reddy',
            f"uploads/{report_id_1}_orig.jpg", f"uploads/{res1['heatmap_file']}", f"uploads/{res1['overlay_file']}",
            f"uploads/{res1['patch_file']}", res1['predicted_class'], res1['confidence'], res1['normal_prob'],
            res1['pneumonia_prob'], res1['findings_summary'], res1['affected_regions'],
            res1['model_type'], res1['model_status'], datetime.now() - timedelta(days=2)
        )
    )
    
    # Share report 1 with Dr. Rajesh Sharma and add Doctor Review
    execute_db(
        """
        INSERT INTO report_shares (analysis_id, patient_id, doctor_id, patient_notes, share_status, shared_at)
        VALUES (%s, %s, %s, %s, 'Reviewed', %s)
        """,
        (
            analysis_id_1, user_ids['patient@medilens.ai'], user_ids['dr.sharma@medilens.ai'],
            "Experiencing productive cough, low-grade fever (100.8 F), and right-sided pleuritic chest discomfort for 4 days.",
            datetime.now() - timedelta(days=2, hours=3)
        )
    )
    
    execute_db(
        """
        INSERT INTO doctor_reviews (
            analysis_id, doctor_id, doctor_name, doctor_specialty, review_status,
            clinical_verdict, clinical_notes, recommendations, reviewed_at
        ) VALUES (%s, %s, %s, %s, 'Reviewed', %s, %s, %s, %s)
        """,
        (
            analysis_id_1, user_ids['dr.sharma@medilens.ai'], 'Dr. Rajesh Sharma', 'MD Pulmonology, Senior Consultant',
            'Confirmed Community-Acquired Pneumonia',
            "Radiological findings strongly corroborate the Vision Transformer self-attention localization. "
            "Right lower lobe airspace consolidation with subtle air bronchograms is evident. "
            "Heart size and mediastinal contours are preserved. No gross pleural effusion noted.",
            "1. Prescribed oral Azithromycin 500mg OD x 5 days with Paracetamol SOS for fever.\n"
            "2. Incentive spirometry and adequate oral hydration.\n"
            "3. Repeat chest radiograph in 14 days if symptoms do not resolve.",
            datetime.now() - timedelta(days=1, hours=18)
        )
    )
    
    # 2. Normal Analysis for Johnathan Miller (Pending Doctor Review)
    report_id_2 = "ML-2026-3104"
    target_orig_2 = os.path.join(Config.UPLOAD_FOLDER, f"{report_id_2}_orig.jpg")
    shutil.copyfile(sample_normal, target_orig_2)
    
    res2 = engine.analyze(target_orig_2, Config.UPLOAD_FOLDER, report_id_2)
    
    analysis_id_2 = execute_db(
        """
        INSERT INTO analyses (
            report_id, patient_id, patient_name, original_image_path, heatmap_path, overlay_path,
            patch_grid_path, predicted_class, confidence, normal_prob, pneumonia_prob,
            findings_summary, affected_regions, model_type, model_status, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            report_id_2, user_ids['patient2@medilens.ai'], 'Johnathan Miller',
            f"uploads/{report_id_2}_orig.jpg", f"uploads/{res2['heatmap_file']}", f"uploads/{res2['overlay_file']}",
            f"uploads/{res2['patch_file']}", res2['predicted_class'], res2['confidence'], res2['normal_prob'],
            res2['pneumonia_prob'], res2['findings_summary'], res2['affected_regions'],
            res2['model_type'], res2['model_status'], datetime.now() - timedelta(days=1, hours=4)
        )
    )
    
    # Share report 2 with Dr. Priya Patel (Pending review)
    execute_db(
        """
        INSERT INTO report_shares (analysis_id, patient_id, doctor_id, patient_notes, share_status, shared_at)
        VALUES (%s, %s, %s, %s, 'Pending Review', %s)
        """,
        (
            analysis_id_2, user_ids['patient2@medilens.ai'], user_ids['dr.priya@medilens.ai'],
            "Annual pre-employment health screening chest radiograph. No respiratory symptoms.",
            datetime.now() - timedelta(hours=6)
        )
    )
    
    # 3. System audit logs
    execute_db(
        "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
        (user_ids['admin@medilens.ai'], 'admin@medilens.ai', 'System Initialization', 'Initialized MediLens AI platform and loaded Vision Transformer model weights.')
    )
    execute_db(
        "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
        (user_ids['patient@medilens.ai'], 'patient@medilens.ai', 'Image Analysis', f'Processed scan {report_id_1} via ViT XAI Rollout.')
    )
    execute_db(
        "INSERT INTO system_logs (user_id, user_email, action, details) VALUES (%s, %s, %s, %s)",
        (user_ids['dr.sharma@medilens.ai'], 'dr.sharma@medilens.ai', 'Clinical Verification', f'Doctor verified report {report_id_1} with diagnosis confirmation.')
    )
    
    print("[Seed] Database seeded with realistic clinical analyses, reviews, and logs successfully!")

if __name__ == '__main__':
    seed()
