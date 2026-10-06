import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from config import Config

def generate_medical_pdf(analysis_dict, doctor_review=None, output_filename=None):
    """
    Generates a clinical-grade medical diagnosis report PDF.
    analysis_dict contains:
      - report_id
      - patient_name
      - patient_age
      - patient_gender
      - predicted_class
      - confidence
      - normal_prob
      - pneumonia_prob
      - original_image_path
      - heatmap_path
      - overlay_path
      - findings_summary
      - affected_regions
      - created_at
      - model_type
    doctor_review contains:
      - doctor_name
      - doctor_specialty
      - clinical_verdict
      - clinical_notes
      - recommendations
      - reviewed_at
    """
    os.makedirs(Config.REPORT_FOLDER, exist_ok=True)
    if not output_filename:
        output_filename = f"report_{analysis_dict['report_id']}.pdf"
        
    pdf_path = os.path.join(Config.REPORT_FOLDER, output_filename)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    header_style = ParagraphStyle(
        'ClinicHeader',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0A2540'),
        alignment=0
    )
    
    sub_header_style = ParagraphStyle(
        'ClinicSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#0D9488')
    )
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0A2540'),
        alignment=1,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1E293B')
    )
    
    bold_body = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    alert_style = ParagraphStyle(
        'AlertPneumonia',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#DC2626')
    )
    
    success_style = ParagraphStyle(
        'AlertNormal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#16A34A')
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#64748B'),
        alignment=1
    )
    
    story = []
    
    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>MediLens AI</b> | Diagnostic Imaging Support System", header_style),
            Paragraph("<b>Report ID:</b> " + str(analysis_dict.get('report_id', 'ML-2026-0001')) + "<br/><b>Date:</b> " + str(analysis_dict.get('created_at', '2026-09-24'))[:19], body_style)
        ],
        [
            Paragraph("Center for Explainable Medical AI & Vision Transformers • B.Tech CSE (AI & ML) Capstone", sub_header_style),
            Paragraph("<b>Status:</b> Official Clinical Report", body_style)
        ]
    ]
    
    header_table = Table(header_data, colWidths=[4.2 * inch, 2.8 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0D9488'), spaceBefore=2, spaceAfter=10))
    
    # Title
    story.append(Paragraph("EXPLAINABLE PULMONARY DIAGNOSIS REPORT", title_style))
    
    # 2. Patient Demographics & Exam Info Table
    patient_info = [
        [
            Paragraph("<b>Patient Name:</b> " + str(analysis_dict.get('patient_name', 'Anonymous')), body_style),
            Paragraph("<b>Age / Gender:</b> " + str(analysis_dict.get('patient_age', '34')) + " Yrs / " + str(analysis_dict.get('patient_gender', 'Female')), body_style),
            Paragraph("<b>Modality:</b> Digital Chest X-Ray (PA)", body_style)
        ],
        [
            Paragraph("<b>Patient ID:</b> " + f"PID-{analysis_dict.get('patient_id', 1):05d}", body_style),
            Paragraph("<b>Model:</b> Vision Transformer (ViT-B/16)", body_style),
            Paragraph("<b>XAI Method:</b> Attention Rollout", body_style)
        ]
    ]
    patient_table = Table(patient_info, colWidths=[2.5 * inch, 2.5 * inch, 2.0 * inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 10))
    
    # 3. AI Prediction & Confidence Box
    is_pneumonia = analysis_dict.get('predicted_class') == 'Pneumonia'
    pred_style = alert_style if is_pneumonia else success_style
    pred_label = "PNEUMONIA DETECTED (Focal Consolidation)" if is_pneumonia else "NORMAL (No Infiltrates Detected)"
    conf_val = analysis_dict.get('confidence', 95.0)
    
    pred_data = [
        [
            Paragraph("<b>AI Diagnosis:</b>", bold_body),
            Paragraph(pred_label, pred_style),
            Paragraph(f"<b>Confidence:</b> {conf_val}%", bold_body),
            Paragraph(f"<b>Pneumonia Prob:</b> {analysis_dict.get('pneumonia_prob', 0)}%", body_style)
        ]
    ]
    pred_table = Table(pred_data, colWidths=[1.3 * inch, 2.9 * inch, 1.4 * inch, 1.4 * inch])
    bg_color = colors.HexColor('#FEF2F2') if is_pneumonia else colors.HexColor('#F0FDF4')
    border_color = colors.HexColor('#F87171') if is_pneumonia else colors.HexColor('#4ADE80')
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_color),
        ('BOX', (0,0), (-1,-1), 1.2, border_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 12))
    
    # 4. Images: Original X-Ray and Heatmap Side-by-Side
    orig_img_path = analysis_dict.get('original_image_path', '')
    heat_img_path = analysis_dict.get('heatmap_path', '')
    
    # Normalize paths
    if orig_img_path and not os.path.isabs(orig_img_path):
        orig_img_path = os.path.join(Config.BASE_DIR, orig_img_path.lstrip('/\\'))
    if heat_img_path and not os.path.isabs(heat_img_path):
        heat_img_path = os.path.join(Config.BASE_DIR, heat_img_path.lstrip('/\\'))
        
    img_cells = []
    if os.path.exists(orig_img_path) and os.path.exists(heat_img_path):
        try:
            rl_orig = RLImage(orig_img_path, width=2.4 * inch, height=2.4 * inch)
            rl_heat = RLImage(heat_img_path, width=2.4 * inch, height=2.4 * inch)
            
            img_table = Table([
                [Paragraph("<b>Figure A:</b> Original Chest Radiograph", bold_body), Paragraph("<b>Figure B:</b> ViT Attention Heatmap (JET XAI)", bold_body)],
                [rl_orig, rl_heat]
            ], colWidths=[3.5 * inch, 3.5 * inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 10))
        except Exception as img_err:
            print(f"[PDF Image Error]: {img_err}")
            
    # 5. Radiological Findings & Attention Rollout Interpretation
    findings_data = [
        [Paragraph("<b>Explainable AI Findings & Localization:</b>", bold_body)],
        [Paragraph(f"<b>Affected Anatomical Zone:</b> {analysis_dict.get('affected_regions', 'N/A')}", body_style)],
        [Paragraph(str(analysis_dict.get('findings_summary', '')), body_style)]
    ]
    findings_table = Table(findings_data, colWidths=[7.0 * inch])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 10))
    
    # 6. Doctor Clinical Review Section
    if doctor_review:
        doc_data = [
            [Paragraph("<b>PHYSICIAN CLINICAL VERIFICATION & NOTES:</b>", bold_body)],
            [Paragraph(f"<b>Attending Physician:</b> {doctor_review.get('doctor_name')} ({doctor_review.get('doctor_specialty', 'Pulmonologist')}) | <b>Reviewed:</b> {str(doctor_review.get('reviewed_at', ''))[:19]}", body_style)],
            [Paragraph(f"<b>Clinical Verdict:</b> <b>{doctor_review.get('clinical_verdict', 'Confirmed')}</b>", body_style)],
            [Paragraph(f"<b>Physician Notes:</b> {doctor_review.get('clinical_notes', 'Diagnosis reviewed and correlated with clinical symptoms.')}", body_style)],
            [Paragraph(f"<b>Recommendations:</b> {doctor_review.get('recommendations', 'Continue supportive care and follow-up as advised.')}", body_style)]
        ]
    else:
        doc_data = [
            [Paragraph("<b>PHYSICIAN CLINICAL REVIEW STATUS:</b>", bold_body)],
            [Paragraph("<b>Status:</b> <i>Pending Clinical Review</i>. This report has not yet been signed off by an attending radiologist/doctor.", body_style)],
            [Paragraph("<b>Physician Notes:</b> ____________________________________________________________________________", body_style)],
            [Paragraph("<b>Signature / Stamp:</b> _________________________________________   <b>Date:</b> ____________", body_style)]
        ]
        
    doc_table = Table(doc_data, colWidths=[7.0 * inch])
    doc_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#93C5FD')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(doc_table)
    story.append(Spacer(1, 10))
    
    # 7. Mandatory Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#CBD5E1'), spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(f"<b>RESEARCH NOTICE & DISCLAIMER:</b> {Config.DISCLAIMER_TEXT}", disclaimer_style))
    
    doc.build(story)
    return pdf_path
