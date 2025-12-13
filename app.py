from flask import Flask, render_template, request, session, make_response
from datetime import datetime
import joblib
import os
from rules import validate_leave_request
from nlp_utils import extract_nlp_features
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Required for sessions

# Load trained models
MODEL_PATH = 'models/ml_model.joblib'
VECTORIZER_PATH = 'models/tfidf_vectorizer.joblib'

ml_model = None
tfidf_vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    ml_model = joblib.load(MODEL_PATH)
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Extract form data
        employee_name = request.form.get('employee_name')
        employee_id = request.form.get('employee_id')
        department = request.form.get('department')
        employment_status = request.form.get('employment_status')
        joining_date_str = request.form.get('joining_date')
        leave_type = request.form.get('leave_type')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')
        reason = request.form.get('reason')

        # Parse dates
        joining_date = datetime.strptime(joining_date_str, '%Y-%m-%d')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

        # Calculate derived fields
        leave_duration = (end_date - start_date).days + 1
        service_period = (datetime.now() - joining_date).days

        # Step 1: Rule-based validation
        rule_result = validate_leave_request(
            employment_status=employment_status,
            service_period=service_period,
            leave_duration=leave_duration,
            leave_type=leave_type,
            start_date=start_date,
            end_date=end_date,
            reason=reason
        )

        if not rule_result['passed']:
            result_data = {
                'decision': 'Rejected',
                'reason': rule_result['message'],
                'method': 'Rule-Based',
                'employee_name': employee_name,
                'employee_id': employee_id,
                'department': department,
                'employment_status': employment_status,
                'joining_date': joining_date_str,
                'leave_type': leave_type,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'leave_duration': leave_duration,
                'reason_text': reason,
                'confidence_score': None,
                'probabilities': None,
                'reason_category': None
            }
            session['result_data'] = result_data
            return render_template('result.html', **result_data)

        # Check for auto-approval (Emergency ≤ 2 days)
        if rule_result.get('auto_approved'):
            result_data = {
                'decision': 'Approved',
                'reason': rule_result['message'],
                'method': 'Auto-Approved (Rule-Based)',
                'employee_name': employee_name,
                'employee_id': employee_id,
                'department': department,
                'employment_status': employment_status,
                'joining_date': joining_date_str,
                'leave_type': leave_type,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'leave_duration': leave_duration,
                'reason_text': reason,
                'confidence_score': None,
                'probabilities': None,
                'reason_category': None
            }
            session['result_data'] = result_data
            return render_template('result.html', **result_data)

        # Step 2: NLP Processing
        nlp_features = extract_nlp_features(reason, tfidf_vectorizer)
        reason_category = nlp_features['category']
        reason_vector = nlp_features['vector']

        # Step 3: ML Prediction
        if ml_model is None or tfidf_vectorizer is None:
            result_data = {
                'decision': 'Error',
                'reason': 'ML model not trained. Please run train.py first.',
                'method': 'System Error',
                'employee_name': employee_name,
                'employee_id': employee_id,
                'department': department,
                'employment_status': employment_status,
                'joining_date': joining_date_str,
                'leave_type': leave_type,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'leave_duration': leave_duration,
                'reason_text': reason,
                'confidence_score': None,
                'probabilities': None,
                'reason_category': None
            }
            session['result_data'] = result_data
            return render_template('result.html', **result_data)

        # Prepare feature vector for ML
        employment_encoding = {'Permanent': 2, 'Probation': 1, 'Contract': 0}
        leave_type_encoding = {'Sick': 0, 'Casual': 1, 'Earned': 2, 'Emergency': 3}
        category_encoding = {'Medical': 0, 'Personal': 1, 'Emergency': 2, 'Other': 3}

        employment_encoded = employment_encoding.get(employment_status, 0)
        leave_type_encoded = leave_type_encoding.get(leave_type, 0)
        category_encoded = category_encoding.get(reason_category, 3)

        # Combine features
        feature_vector = [
            service_period,
            leave_duration,
            employment_encoded,
            leave_type_encoded,
            category_encoded
        ]

        # Add TF-IDF features
        feature_vector.extend(reason_vector.tolist())

        # Predict with probability
        prediction = ml_model.predict([feature_vector])[0]
        
        # Get prediction probabilities
        try:
            probabilities = ml_model.predict_proba([feature_vector])[0]
            confidence_score = round(probabilities[prediction] * 100, 2)
            
            prob_breakdown = {
                'Rejected': round(probabilities[0] * 100, 2),
                'Approved': round(probabilities[1] * 100, 2)
            }
        except:
            confidence_score = None
            prob_breakdown = None

        decision = 'Approved' if prediction == 1 else 'Rejected'
        ml_reason = f"ML Model Decision based on historical patterns. Reason categorized as: {reason_category}"

        result_data = {
            'decision': decision,
            'reason': ml_reason,
            'method': 'Machine Learning + NLP',
            'employee_name': employee_name,
            'employee_id': employee_id,
            'department': department,
            'employment_status': employment_status,
            'joining_date': joining_date_str,
            'leave_type': leave_type,
            'start_date': start_date_str,
            'end_date': end_date_str,
            'leave_duration': leave_duration,
            'reason_text': reason,
            'reason_category': reason_category,
            'confidence_score': confidence_score,
            'probabilities': prob_breakdown
        }
        
        # Store result data in session for PDF generation
        session['result_data'] = result_data
        
        return render_template('result.html', **result_data)

    except Exception as e:
        result_data = {
            'decision': 'Error',
            'reason': f'An error occurred: {str(e)}',
            'method': 'System Error',
            'employee_name': 'N/A',
            'employee_id': 'N/A',
            'department': 'N/A',
            'employment_status': 'N/A',
            'joining_date': 'N/A',
            'leave_type': 'N/A',
            'start_date': 'N/A',
            'end_date': 'N/A',
            'leave_duration': 0,
            'reason_text': 'N/A',
            'confidence_score': None,
            'probabilities': None,
            'reason_category': None
        }
        session['result_data'] = result_data
        return render_template('result.html', **result_data)

@app.route('/download-report')
def download_report():
    """Generate and download PDF report of leave request analysis"""
    result_data = session.get('result_data')
    
    if not result_data:
        return "No result data available. Please submit a leave request first.", 400
    
    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#7c5fc5'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=6
    )
    
    # Title
    elements.append(Paragraph("🤖 AI HR Leave Request Analysis Report", title_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Decision Status Box
    decision = result_data.get('decision', 'Unknown')
    decision_color = colors.HexColor('#48bb78') if decision == 'Approved' else colors.HexColor('#f56565') if decision == 'Rejected' else colors.HexColor('#ed8936')
    
    decision_data = [[Paragraph(f"<b>Decision: {decision}</b>", normal_style)]]
    decision_table = Table(decision_data, colWidths=[6.5*inch])
    decision_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), decision_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('ROUNDEDCORNERS', [10, 10, 10, 10]),
    ]))
    elements.append(decision_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Employee Information
    elements.append(Paragraph("Employee Information", heading_style))
    emp_data = [
        ['Employee Name:', result_data.get('employee_name', 'N/A')],
        ['Employee ID:', result_data.get('employee_id', 'N/A')],
        ['Department:', result_data.get('department', 'N/A')],
        ['Employment Status:', result_data.get('employment_status', 'N/A')],
        ['Joining Date:', result_data.get('joining_date', 'N/A')],
    ]
    emp_table = Table(emp_data, colWidths=[2*inch, 4.5*inch])
    emp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fc')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2d3748')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecf2')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(emp_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Leave Request Details
    elements.append(Paragraph("Leave Request Details", heading_style))
    leave_data = [
        ['Leave Type:', result_data.get('leave_type', 'N/A')],
        ['Start Date:', result_data.get('start_date', 'N/A')],
        ['End Date:', result_data.get('end_date', 'N/A')],
        ['Duration:', f"{result_data.get('leave_duration', 0)} day(s)"],
        ['Decision Engine:', result_data.get('method', 'N/A')],
    ]
    leave_table = Table(leave_data, colWidths=[2*inch, 4.5*inch])
    leave_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fc')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2d3748')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecf2')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
    ]))
    elements.append(leave_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Reason for Leave
    elements.append(Paragraph("Reason for Leave", heading_style))
    reason_text = result_data.get('reason_text', 'N/A')
    reason_para = Paragraph(reason_text, normal_style)
    reason_data = [[reason_para]]
    reason_table = Table(reason_data, colWidths=[6.5*inch])
    reason_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2d3748')),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecf2')),
    ]))
    elements.append(reason_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Decision Explanation
    elements.append(Paragraph("Decision Explanation", heading_style))
    explanation = result_data.get('reason', 'No explanation available.')
    if result_data.get('reason_category'):
        explanation += f" | Category: {result_data.get('reason_category')}"
    
    exp_para = Paragraph(explanation, normal_style)
    exp_data = [[exp_para]]
    exp_table = Table(exp_data, colWidths=[6.5*inch])
    exp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fc')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2d3748')),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecf2')),
    ]))
    elements.append(exp_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Confidence Score & Probabilities
    if result_data.get('confidence_score'):
        elements.append(Paragraph("Model Confidence Analysis", heading_style))
        
        conf_data = [
            ['Confidence Score:', f"{result_data.get('confidence_score')}%"],
        ]
        
        if result_data.get('probabilities'):
            for label, prob in result_data['probabilities'].items():
                conf_data.append([f'{label} Probability:', f'{prob}%'])
        
        conf_table = Table(conf_data, colWidths=[2.5*inch, 4*inch])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fc')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#7c5fc5')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e9ecf2')),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        elements.append(conf_table)
        elements.append(Spacer(1, 0.2*inch))
    
    # Footer
    elements.append(Spacer(1, 0.3*inch))
    footer_text = f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"
    footer_para = Paragraph(footer_text, ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER
    ))
    elements.append(footer_para)
    
    powered_by = Paragraph("Powered by Rule-Based AI + Machine Learning + NLP", ParagraphStyle(
        'PoweredBy',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7c5fc5'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    elements.append(powered_by)
    # Add copyright line
    copyright_line = Paragraph(
        "© 2025 HR-AI Decision System • Built by Charan",
        ParagraphStyle(
            'Copyright',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#718096'),
            alignment=TA_CENTER
            )
            )
    elements.append(copyright_line)

    
    # Build PDF
    doc.build(elements)
    
    # Prepare response
    buffer.seek(0)
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Leave_Request_Report_{result_data.get("employee_id", "Report")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    return response

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)

    app.run(debug=True, port=5000)