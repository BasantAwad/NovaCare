# NovaCare - AI Healthcare Companion

An AI-powered healthcare companion with emotional support, medical Q&A, and fall detection capabilities. Built with **Flask** and following **SOLID principles**.

## ✨ Features

- 🧠 **Conversational AI** - Emotional support with fine-tuned DialoGPT
- 😊 **Emotion Analysis** - Text and facial emotion detection (unified)
- 🩺 **Medical Q&A** - Fine-tuned Flan-T5 on medical datasets
- 🚨 **Emergency Detection** - Automatic alert triggering
- 📊 **Dashboard** - Role-based views for patients, caregivers, doctors
- 🔐 **Authentication** - User login with role management

---

## 📁 Project Structure (SOLID)

```
NovaCare/
├── run.py                    # Entry point
├── novabrain.py              # AI orchestrator
├── models.py                 # Database models
├── system_logger.py          # Logging system
├── train_models.py           # Model training CLI
│
├── ai/                       # AI Package (SOLID)
│   ├── __init__.py           # Exports + DI getters
│   ├── interfaces/           # ISP - Focused interfaces
│   │   ├── emotion_analyzer.py     → IEmotionAnalyzer
│   │   ├── conversational_agent.py → IConversationalAgent
│   │   └── medical_qa.py           → IMedicalQA
│   ├── impl/                 # SRP - Single responsibility
│   │   ├── emotion_analyzer.py     → EmotionAnalyzer
│   │   ├── conversational_ai.py    → ConversationalAI
│   │   └── medical_qa.py           → MedicalQA
│   └── trained_models/       # Saved model weights
│
├── app/                      # Flask app (SRP via Blueprints)
│   ├── __init__.py           # App factory + DI
│   ├── routes/
│   │   ├── auth.py           # Login/logout
│   │   ├── dashboard.py      # Role-based dashboards
│   │   └── api/              # REST endpoints
│   │       ├── chat.py
│   │       ├── alerts.py
│   │       ├── vitals.py
│   │       ├── medication.py
│   │       └── reports.py
│   ├── templates/
│   └── static/
│
├── services/                 # DI Container
│   └── container.py
│
└── testing/                  # Test utilities
```

---

## 🚀 Quick Start

### Option 1: Quick Start (Windows)

```bash
start.bat
```

### Option 2: Manual

```bash
pip install -r requirements.txt
python run.py
```

Open: `http://localhost:5000`

---

## 🔧 AI Module Usage

```python
# Import interfaces and implementations
from ai import IEmotionAnalyzer, EmotionAnalyzer
from ai import IConversationalAgent, ConversationalAI
from ai import IMedicalQA, MedicalQA

# Use singleton getters (Dependency Injection)
from ai import get_emotion_analyzer, get_conversational_ai, get_medical_qa

# Unified emotion analysis (text + face)
analyzer = get_emotion_analyzer()
result = analyzer.analyze("I'm feeling happy!")     # Text input
result = analyzer.analyze(face_image_array)         # Face image (48x48)
result = analyzer.analyze_text("I'm sad")           # Explicit text
result = analyzer.analyze_face(image)               # Explicit face

# Conversational AI
ai = get_conversational_ai()
response = ai.generate_response("Hello!", emotion="happy")

# Medical Q&A
qa = get_medical_qa()
answer = qa.query("What should I do for a headache?")
```

---

## 🎯 Training Models

```bash
# Train medical QA
python train_models.py --medical

# Train conversational AI
python train_models.py --conversation

# Train emotion (face)
python train_models.py --emotion-face --emotion-dataset /path/to/fer

# Train emotion (text)
python train_models.py --emotion-text --text-emotion-dataset /path/to/csv

# Download datasets
python train_models.py --download
```

---

## 🏗️ SOLID Principles

| Principle                 | Implementation                                                   |
| ------------------------- | ---------------------------------------------------------------- |
| **S**ingle Responsibility | Each impl class has one job; Flask Blueprints separate routes    |
| **O**pen/Closed           | Extend via new interface implementations                         |
| **L**iskov Substitution   | All impls are interchangeable via interfaces                     |
| **I**nterface Segregation | 3 focused interfaces in `ai/interfaces/`                         |
| **D**ependency Inversion  | `ai/__init__.py` provides singleton getters; app factory uses DI |

---

## 📱 API Endpoints

| Endpoint              | Method   | Description                 |
| --------------------- | -------- | --------------------------- |
| `/api/chat`           | POST     | Chat with NovaBrain         |
| `/api/emergency`      | POST     | Trigger emergency alert     |
| `/api/vitals`         | POST/GET | Record/retrieve vital signs |
| `/api/alerts`         | GET/PUT  | Manage alerts               |
| `/api/medication`     | CRUD     | Medication management       |
| `/api/reports/health` | GET      | Generate health report      |

---

## 👥 Team

- **Basant Awad** (22101405)
- **Nadira El-Sirafy** (22101377)
- **Noureen Yasser** (22101109)
- **Muhammad Mustafa** (22101336)
- **Ramez Asaad** (22100506)

---

## 📊 Tech Stack

- **Backend**: Flask, Flask-SQLAlchemy, Flask-Login
- **AI**: HuggingFace Transformers, TensorFlow/Keras, scikit-learn
- **Frontend**: Bootstrap 5, Chart.js
- **Database**: SQLite

---

**Built with ❤️ following SOLID principles**
