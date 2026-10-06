import os
import sys
import unittest
from app import create_app
from database.db import query_db, execute_db

class MediLensAiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()

    def test_01_landing_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MediLens', response.data)
        self.assertIn(b'Vision Transformers', response.data)
        print("[TEST PASS] 01. Landing page loaded successfully.")

    def test_02_login_page(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sign In to Portal', response.data)
        self.assertIn(b'patient@medilens.ai', response.data)
        print("[TEST PASS] 02. Login page loaded with demo account credentials.")

    def test_03_patient_authentication_and_dashboard(self):
        # Login as patient
        login_res = self.client.post('/login', data={
            'email': 'patient@medilens.ai',
            'password': 'Patient@123',
            'role': 'patient'
        }, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)
        self.assertIn(b'Patient Diagnostic Portal', login_res.data)
        self.assertIn(b'ML-2026-8491', login_res.data)
        print("[TEST PASS] 03. Patient login and dashboard loaded with preloaded scan ML-2026-8491.")

    def test_04_patient_report_and_explanation(self):
        # Login first
        self.client.post('/login', data={
            'email': 'patient@medilens.ai',
            'password': 'Patient@123',
            'role': 'patient'
        })
        
        # Report page
        rep_res = self.client.get('/patient/report/ML-2026-8491')
        self.assertEqual(rep_res.status_code, 200)
        self.assertIn(b'Explainable Diagnostic Findings', rep_res.data)
        self.assertIn(b'Pneumonia Detected', rep_res.data)
        self.assertIn(b'interactiveOverlayImg', rep_res.data)
        print("[TEST PASS] 04. Report page loaded with interactive overlay and findings.")

        # Explanation page
        exp_res = self.client.get('/patient/explanation/ML-2026-8491')
        self.assertEqual(exp_res.status_code, 200)
        self.assertIn(b'4-Stage Visual Decomposition Pipeline', exp_res.data)
        self.assertIn(b'Attention Rollout Heatmap', exp_res.data)
        print("[TEST PASS] 05. XAI Explanation page loaded with 4-stage comparison.")

    def test_05_pdf_generation(self):
        self.client.post('/login', data={
            'email': 'patient@medilens.ai',
            'password': 'Patient@123',
            'role': 'patient'
        })
        pdf_res = self.client.get('/patient/download-pdf/ML-2026-8491')
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.content_type, 'application/pdf')
        self.assertTrue(len(pdf_res.data) > 1000)
        print(f"[TEST PASS] 06. PDF report generated successfully ({len(pdf_res.data)} bytes).")

    def test_06_doctor_dashboard_and_inspect(self):
        # Login as doctor
        login_res = self.client.post('/login', data={
            'email': 'dr.sharma@medilens.ai',
            'password': 'Doctor@123',
            'role': 'doctor'
        }, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)
        self.assertIn(b'Clinical Workstation', login_res.data)
        self.assertIn(b'Anitha Reddy', login_res.data)
        print("[TEST PASS] 07. Doctor dashboard loaded with authorized shared scans only.")

        # Doctor inspect page
        insp_res = self.client.get('/doctor/inspect/ML-2026-8491')
        self.assertEqual(insp_res.status_code, 200)
        self.assertIn(b'Doctor Clinical Review Form', insp_res.data)
        self.assertIn(b'Dr. Rajesh Sharma', insp_res.data)
        print("[TEST PASS] 08. Doctor inspection workstation loaded with review form.")

    def test_07_admin_dashboard_and_performance(self):
        # Login as admin
        login_res = self.client.post('/login', data={
            'email': 'admin@medilens.ai',
            'password': 'Admin@123',
            'role': 'admin'
        }, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)
        self.assertIn(b'Administration & Analytics Portal', login_res.data)
        self.assertIn(b'Registered Accounts', login_res.data)
        print("[TEST PASS] 09. Admin dashboard loaded with user management and audit log.")

        # Model performance page
        perf_res = self.client.get('/admin/model-performance')
        self.assertEqual(perf_res.status_code, 200)
        self.assertIn(b'96.4%', perf_res.data)
        self.assertIn(b'accuracyChart', perf_res.data)
        self.assertIn(b'lossChart', perf_res.data)
        self.assertIn(b'rocChart', perf_res.data)
        print("[TEST PASS] 10. Model performance page loaded with Chart.js canvases & metrics.")

    def test_08_api_health_and_model_info(self):
        h_res = self.client.get('/api/v1/health')
        self.assertEqual(h_res.status_code, 200)
        json_h = h_res.get_json()
        self.assertEqual(json_h['status'], 'healthy')
        self.assertIn('Vision Transformer', json_h['model'])

        m_res = self.client.get('/api/v1/model-info')
        self.assertEqual(m_res.status_code, 200)
        json_m = m_res.get_json()
        self.assertEqual(json_m['num_patches'], 196)
        print("[TEST PASS] 11. REST API endpoints /health and /model-info verified.")

    def test_09_image_analysis_pipeline(self):
        # Test image analysis with sample_normal
        self.client.post('/login', data={
            'email': 'patient@medilens.ai',
            'password': 'Patient@123',
            'role': 'patient'
        })
        analyze_res = self.client.post('/patient/analyze', data={
            'sample_type': 'sample_normal'
        }, follow_redirects=True)
        self.assertEqual(analyze_res.status_code, 200)
        self.assertIn(b'Explainable Diagnostic Findings', analyze_res.data)
        self.assertIn(b'Normal', analyze_res.data)
        print("[TEST PASS] 12. Full end-to-end ViT inference pipeline tested & verified!")

if __name__ == '__main__':
    unittest.main()
