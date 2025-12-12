from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User, Role, Alert, VitalSign
from novabrain import NovaBrain

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

# Initialize NovaBrain
print("Initializing NovaBrain...")
nova = NovaBrain(use_local=True) 

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
        # Seed Roles
        roles = ['primary', 'caregiver', 'doctor', 'emergency']
        for r_name in roles:
            if not Role.query.filter_by(name=r_name).first():
                db.session.add(Role(name=r_name))
        db.session.commit()

        # Seed Users
        if not User.query.filter_by(username='user').first():
            r = Role.query.filter_by(name='primary').first()
            u = User(username='user', password_hash='password', role=r)
            db.session.add(u)
        
        if not User.query.filter_by(username='guardian').first():
            r = Role.query.filter_by(name='caregiver').first()
            u = User(username='guardian', password_hash='password', role=r)
            db.session.add(u)

        if not User.query.filter_by(username='doctor').first():
            r = Role.query.filter_by(name='doctor').first()
            u = User(username='doctor', password_hash='password', role=r)
            db.session.add(u)

        if not User.query.filter_by(username='emergency').first():
            r = Role.query.filter_by(name='emergency').first()
            u = User(username='emergency', password_hash='password', role=r)
            db.session.add(u)
            
        db.session.commit()
        print("Database initialized and seeded.")

# Initialize DB on start
init_db()

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
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    role = current_user.role.name
    if role == 'primary':
        return render_template('dashboard_primary.html', user=current_user)
    elif role == 'caregiver':
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(5).all()
        return render_template('dashboard_caregiver.html', user=current_user, alerts=alerts)
    elif role == 'doctor':
        return render_template('dashboard_doctor.html', user=current_user)
    elif role == 'emergency':
        alerts = Alert.query.filter_by(status='New').all()
        return render_template('dashboard_emergency.html', user=current_user, alerts=alerts)
    else:
        return "Role not recognized", 403

# --- API Endpoints ---
@app.route('/api/chat', methods=['POST'])
def chat_api():
    # Helper to allow unauthenticated chat for demo if needed, but preferred authenticated
    data = request.json
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    response_text = nova.process_input(user_input)
    return jsonify({
        'response': response_text,
        'history': nova.get_history()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
