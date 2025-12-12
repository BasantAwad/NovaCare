from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
import os
import json
from datetime import datetime, timedelta
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, Role, Alert, VitalSign, Medication, MedicationLog, HealthReport, EmotionLog, SystemLog, ChatHistory
from novabrain import NovaBrain, get_nova
from system_logger import get_logger

# Get the directory where this file is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Configure Flask
app = Flask(__name__, 
            template_folder=os.path.join(basedir, 'app', 'templates'),
            static_folder=os.path.join(basedir, 'app', 'static'))
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'novacare.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Initialize Logger
logger = get_logger()

# Initialize NovaBrain
print("Initializing NovaBrain...")
nova = get_nova()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
        logger.set_db(db)
        
        # Seed Roles
        roles = ['primary', 'caregiver', 'doctor', 'emergency']
        for r_name in roles:
            if not Role.query.filter_by(name=r_name).first():
                db.session.add(Role(name=r_name))
        db.session.commit()

        # Seed Users
        users_to_seed = [
            ('user', 'password', 'primary'),
            ('guardian', 'password', 'caregiver'),
            ('doctor', 'password', 'doctor'),
            ('emergency', 'password', 'emergency')
        ]
        for username, pwd, role_name in users_to_seed:
            if not User.query.filter_by(username=username).first():
                r = Role.query.filter_by(name=role_name).first()
                u = User(username=username, password_hash=pwd, role=r)
                db.session.add(u)
        db.session.commit()
        
        logger.info('SYSTEM', 'Database initialized and seeded')
        print("Database initialized and seeded.")

# Initialize DB on start
init_db()

# ==================== PAGE ROUTES ====================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.password_hash == password:
            login_user(user)
            logger.info('AUTH', f'User {username} logged in', user_id=user.id)
            return redirect(url_for('dashboard'))
        logger.warning('AUTH', f'Failed login attempt for {username}')
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logger.info('AUTH', f'User {current_user.username} logged out', user_id=current_user.id)
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    role = current_user.role.name
    if role == 'primary':
        meds = Medication.query.filter_by(user_id=current_user.id, is_active=True).all()
        return render_template('dashboard_primary.html', user=current_user, medications=meds)
    elif role == 'caregiver':
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(10).all()
        vitals = VitalSign.query.order_by(VitalSign.timestamp.desc()).limit(10).all()
        return render_template('dashboard_caregiver.html', user=current_user, alerts=alerts, vitals=vitals)
    elif role == 'doctor':
        vitals = VitalSign.query.order_by(VitalSign.timestamp.desc()).limit(20).all()
        reports = HealthReport.query.order_by(HealthReport.generated_at.desc()).limit(5).all()
        return render_template('dashboard_doctor.html', user=current_user, vitals=vitals, reports=reports)
    elif role == 'emergency':
        alerts = Alert.query.filter_by(status='New').order_by(Alert.timestamp.desc()).all()
        return render_template('dashboard_emergency.html', user=current_user, alerts=alerts)
    else:
        return "Role not recognized", 403

# ==================== API ROUTES ====================

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Main chat endpoint with full AI processing"""
    try:
        data = request.json
        user_input = data.get('message', '')
        user_id = current_user.id if current_user.is_authenticated else None
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Process with NovaBrain
        result = nova.process_input(user_input, user_id=user_id)
        
        # ========== CONSOLE LOGGING FOR EMOTION ==========
        if result.get('text_emotion'):
            emotion = result['text_emotion'].get('emotion', 'unknown')
            confidence = result['text_emotion'].get('confidence', 0)
            print(f"\n{'='*50}")
            print(f"[EMOTION DETECTED] User: '{user_input[:50]}...'")
            print(f"[EMOTION DETECTED] Emotion: {emotion.upper()} (Confidence: {confidence:.0%})")
            print(f"{'='*50}\n")
        
        # Log chat to database
        if user_id:
            try:
                chat_log = ChatHistory(
                    user_id=user_id,
                    user_message=user_input,
                    bot_response=result['response'],
                    detected_intent=result['intent'],
                    detected_emotion=result['text_emotion'].get('emotion') if result['text_emotion'] else None
                )
                db.session.add(chat_log)
                
                # Log emotion
                if result['text_emotion']:
                    emotion_log = EmotionLog(
                        user_id=user_id,
                        emotion=result['text_emotion']['emotion'],
                        confidence=result['text_emotion']['confidence'],
                        source='text'
                    )
                    db.session.add(emotion_log)
                
                db.session.commit()
                logger.log_chat(user_id, user_input, result['response'])
            except Exception as db_error:
                print(f"[DB ERROR] Failed to log chat: {db_error}")
                db.session.rollback()
        
        # If emergency detected, create alert
        if result.get('is_emergency') and user_id:
            try:
                alert = Alert(
                    user_id=user_id,
                    type='Emergency Detected',
                    message=user_input,
                    status='New'
                )
                db.session.add(alert)
                db.session.commit()
                logger.log_alert(user_id, 'Emergency Detected', 'New')
            except Exception as alert_error:
                print(f"[ALERT ERROR] Failed to create alert: {alert_error}")
                db.session.rollback()
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[CHAT API ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'response': "I'm sorry, I encountered an error. Please try again.",
            'error': str(e),
            'intent': 'error',
            'text_emotion': {'emotion': 'neutral', 'confidence': 0.5},
            'is_emergency': False
        }), 200

@app.route('/api/emergency', methods=['POST'])
def emergency_api():
    """Trigger emergency alert"""
    data = request.json
    user_id = data.get('user_id') or (current_user.id if current_user.is_authenticated else 1)
    alert_type = data.get('type', 'Manual Emergency')
    message = data.get('message', 'Emergency button pressed')
    lat = data.get('latitude')
    lng = data.get('longitude')
    
    alert = Alert(
        user_id=user_id,
        type=alert_type,
        message=message,
        status='New',
        latitude=lat,
        longitude=lng
    )
    db.session.add(alert)
    db.session.commit()
    
    logger.log_alert(user_id, alert_type, 'New')
    
    return jsonify({
        'success': True,
        'alert_id': alert.id,
        'message': 'Emergency alert created and dispatched'
    })

@app.route('/api/vitals', methods=['POST'])
def vitals_api():
    """Record vital signs"""
    data = request.json
    user_id = data.get('user_id') or (current_user.id if current_user.is_authenticated else 1)
    heart_rate = data.get('heart_rate')
    spo2 = data.get('spo2')
    stress = data.get('stress_level', 'normal')
    
    vital = VitalSign(
        user_id=user_id,
        heart_rate=heart_rate,
        spo2=spo2,
        stress_level=stress
    )
    db.session.add(vital)
    db.session.commit()
    
    logger.log_vital(user_id, heart_rate, spo2)
    
    # Check for abnormal readings
    if heart_rate and (heart_rate < 50 or heart_rate > 120):
        alert = Alert(user_id=user_id, type='Abnormal Heart Rate', message=f'HR: {heart_rate}', status='New')
        db.session.add(alert)
        db.session.commit()
        logger.log_alert(user_id, 'Abnormal Heart Rate', 'New')
    
    return jsonify({'success': True, 'vital_id': vital.id})

@app.route('/api/vitals/<int:user_id>', methods=['GET'])
def get_vitals_api(user_id):
    """Get recent vitals for a user"""
    limit = request.args.get('limit', 20, type=int)
    vitals = VitalSign.query.filter_by(user_id=user_id).order_by(VitalSign.timestamp.desc()).limit(limit).all()
    return jsonify([{
        'id': v.id,
        'heart_rate': v.heart_rate,
        'spo2': v.spo2,
        'stress_level': v.stress_level,
        'timestamp': v.timestamp.isoformat()
    } for v in vitals])

@app.route('/api/alerts', methods=['GET'])
def get_alerts_api():
    """Get alerts, optionally filtered by status"""
    status = request.args.get('status')
    query = Alert.query.order_by(Alert.timestamp.desc())
    if status:
        query = query.filter_by(status=status)
    alerts = query.limit(50).all()
    return jsonify([{
        'id': a.id,
        'user_id': a.user_id,
        'type': a.type,
        'message': a.message,
        'status': a.status,
        'latitude': a.latitude,
        'longitude': a.longitude,
        'timestamp': a.timestamp.isoformat()
    } for a in alerts])

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert_api(alert_id):
    """Acknowledge an alert"""
    alert = Alert.query.get_or_404(alert_id)
    alert.status = 'Acknowledged'
    db.session.commit()
    logger.info('ALERT', f'Alert {alert_id} acknowledged', user_id=current_user.id if current_user.is_authenticated else None)
    return jsonify({'success': True})

@app.route('/api/medication', methods=['GET', 'POST'])
def medication_api():
    """CRUD for medications"""
    if request.method == 'POST':
        data = request.json
        user_id = data.get('user_id') or (current_user.id if current_user.is_authenticated else 1)
        from datetime import time
        schedule_time = datetime.strptime(data.get('schedule_time', '08:00'), '%H:%M').time()
        
        med = Medication(
            user_id=user_id,
            name=data.get('name'),
            dosage=data.get('dosage'),
            schedule_time=schedule_time,
            frequency=data.get('frequency', 'daily'),
            notes=data.get('notes')
        )
        db.session.add(med)
        db.session.commit()
        logger.log_medication(user_id, med.name, 'created')
        return jsonify({'success': True, 'medication_id': med.id})
    
    else:
        user_id = request.args.get('user_id', type=int)
        query = Medication.query.filter_by(is_active=True)
        if user_id:
            query = query.filter_by(user_id=user_id)
        meds = query.all()
        return jsonify([{
            'id': m.id,
            'name': m.name,
            'dosage': m.dosage,
            'schedule_time': m.schedule_time.strftime('%H:%M') if m.schedule_time else None,
            'frequency': m.frequency
        } for m in meds])

@app.route('/api/report/<int:user_id>', methods=['GET'])
def generate_report_api(user_id):
    """Generate health report for a user"""
    days = request.args.get('days', 7, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Gather data
    vitals = VitalSign.query.filter(
        VitalSign.user_id == user_id,
        VitalSign.timestamp >= start_date
    ).all()
    
    alerts = Alert.query.filter(
        Alert.user_id == user_id,
        Alert.timestamp >= start_date
    ).all()
    
    meds = MedicationLog.query.filter(
        MedicationLog.user_id == user_id,
        MedicationLog.scheduled_time >= start_date
    ).all()
    
    # Calculate metrics
    avg_hr = sum(v.heart_rate for v in vitals if v.heart_rate) / len(vitals) if vitals else 0
    avg_spo2 = sum(v.spo2 for v in vitals if v.spo2) / len(vitals) if vitals else 0
    taken_meds = len([m for m in meds if m.status == 'taken'])
    total_meds = len(meds) if meds else 1
    adherence = (taken_meds / total_meds) * 100
    
    # Generate summary
    summary = f"""
Health Report for User ID: {user_id}
Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

Vital Signs:
- Average Heart Rate: {avg_hr:.1f} BPM
- Average SpO2: {avg_spo2:.1f}%
- Total Readings: {len(vitals)}

Alerts:
- Total Alerts: {len(alerts)}
- Emergency Alerts: {len([a for a in alerts if 'emergency' in a.type.lower()])}

Medication Adherence: {adherence:.1f}%
"""
    
    # Save report
    report = HealthReport(
        user_id=user_id,
        period_start=start_date,
        period_end=end_date,
        avg_heart_rate=avg_hr,
        avg_spo2=avg_spo2,
        alert_count=len(alerts),
        medication_adherence=adherence,
        summary=summary
    )
    db.session.add(report)
    db.session.commit()
    
    logger.info('HEALTH', f'Generated health report for user {user_id}', user_id=user_id)
    
    return jsonify({
        'report_id': report.id,
        'summary': summary,
        'metrics': {
            'avg_heart_rate': avg_hr,
            'avg_spo2': avg_spo2,
            'alert_count': len(alerts),
            'medication_adherence': adherence
        }
    })

@app.route('/api/logs', methods=['GET'])
def get_logs_api():
    """Get system logs (admin only)"""
    category = request.args.get('category')
    level = request.args.get('level')
    limit = request.args.get('limit', 100, type=int)
    
    query = SystemLog.query.order_by(SystemLog.timestamp.desc())
    if category:
        query = query.filter_by(category=category.upper())
    if level:
        query = query.filter_by(level=level.upper())
    
    logs = query.limit(limit).all()
    return jsonify([{
        'id': l.id,
        'level': l.level,
        'category': l.category,
        'message': l.message,
        'user_id': l.user_id,
        'details': l.details,
        'timestamp': l.timestamp.isoformat()
    } for l in logs])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
