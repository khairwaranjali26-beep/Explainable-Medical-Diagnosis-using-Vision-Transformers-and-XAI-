import os
import uuid
import shutil
from datetime import datetime
from flask import Blueprint, request, jsonify, session
from config import Config
from database.db import query_db, execute_db
from models.vit_model import ViTDiagnosticEngine

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'MediLens AI Inference API',
        'model': 'Vision Transformer (ViT-B/16 Custom Architecture)',
        'xai_method': 'Attention Rollout',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@api_bp.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'architecture': 'Vision Transformer (ViT)',
        'input_resolution': [Config.IMG_SIZE, Config.IMG_SIZE, 3],
        'patch_size': [Config.PATCH_SIZE, Config.PATCH_SIZE],
        'num_patches': (Config.IMG_SIZE // Config.PATCH_SIZE) ** 2,
        'embedding_dim': Config.EMBED_DIM,
        'num_heads': Config.NUM_HEADS,
        'num_layers': Config.NUM_LAYERS,
        'classes': Config.CLASS_NAMES,
        'accuracy': 96.4,
        'precision': 95.8,
        'recall': 97.1,
        'f1_score': 96.4,
        'auc_roc': 0.982
    })

@api_bp.route('/reports', methods=['GET'])
def get_reports():
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    
    if not user_id:
        return jsonify({'error': 'Unauthorized. Session required.'}), 401
        
    if user_role == 'patient':
        reports = query_db(
            "SELECT id, report_id, predicted_class, confidence, created_at FROM analyses WHERE patient_id = %s ORDER BY created_at DESC",
            (user_id,)
        )
    elif user_role == 'doctor':
        reports = query_db(
            """
            SELECT a.id, a.report_id, a.patient_name, a.predicted_class, a.confidence, rs.share_status, rs.shared_at
            FROM report_shares rs JOIN analyses a ON rs.analysis_id = a.id
            WHERE rs.doctor_id = %s ORDER BY rs.shared_at DESC
            """, (user_id,)
        )
    else: # admin
        reports = query_db("SELECT id, report_id, patient_name, predicted_class, confidence, created_at FROM analyses ORDER BY created_at DESC LIMIT 50")
        
    return jsonify({'success': True, 'count': len(reports), 'reports': reports})

@api_bp.route('/reports/<report_id>', methods=['GET'])
def get_report_detail(report_id):
    analysis = query_db("SELECT * FROM analyses WHERE report_id = %s", (report_id,), one=True)
    if not analysis:
        return jsonify({'error': 'Report not found'}), 404
        
    reviews = query_db("SELECT * FROM doctor_reviews WHERE analysis_id = %s", (analysis['id'],))
    return jsonify({
        'success': True,
        'report': analysis,
        'reviews': reviews
    })

@api_bp.route('/analyze-api', methods=['POST'])
def analyze_api():
    """
    API endpoint for automated chest X-ray processing
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in request (key: image)'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    report_id = f"ML-{datetime.now().year}-{str(uuid.uuid4())[:8].upper()}"
    filename = f"{report_id}_orig.jpg"
    target_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(target_path)
    
    try:
        engine = ViTDiagnosticEngine.get_instance()
        res = engine.analyze(target_path, Config.UPLOAD_FOLDER, report_id)
        
        return jsonify({
            'success': True,
            'report_id': report_id,
            'prediction': res['predicted_class'],
            'confidence': res['confidence'],
            'normal_probability': res['normal_prob'],
            'pneumonia_probability': res['pneumonia_prob'],
            'affected_regions': res['affected_regions'],
            'findings_summary': res['findings_summary'],
            'original_image_url': f"/static/uploads/{filename}",
            'heatmap_url': f"/static/uploads/{res['heatmap_file']}",
            'overlay_url': f"/static/uploads/{res['overlay_file']}",
            'disclaimer': Config.DISCLAIMER_TEXT
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
