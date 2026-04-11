# NovaCare - AI Healthcare Companion

AI-powered healthcare companion with emotional support, medical Q&A, and vital monitoring.

> **Monorepo note:** The production-style **NovaBot chat API** lives in **`../services/llm-backend`** (Flask, port 5000). It supports **local Ollama** (fast) and **Hugging Face Inference** (quality), with `llm_profile` / `prefer_quality` on `POST /api/chat`. See **`../services/llm-backend/README.md`** and the root **`../README.md`**. The `frontend/ai/` package below is used by the integrated Flask dashboard (`frontend/backend`) and may use different provider config than the LLM Backend.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python run.py
```

Open: **http://localhost:5000**

---

## 📁 Project Structure

```
NovaCare/
├── run.py              ← Entry point
├── novabrain.py        ← AI orchestrator
├── models.py           ← Database models
├── system_logger.py    ← Logging
│
├── ai/
│   ├── config.py       ← HuggingFace API config
│   ├── impl/
│   │   ├── conversational_ai.py
│   │   ├── emotion_analyzer.py
│   │   └── medical_qa.py
│   └── data/           ← Knowledge base
│
└── app/
    ├── routes/         ← Flask blueprints
    │   ├── auth.py
    │   ├── dashboard.py
    │   └── api/
    └── templates/
```

---

## ✨ Features

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| 💬 **Chat**        | Emotional support chatbot    |
| 😊 **Emotion**     | Text emotion detection       |
| 🩺 **Medical Q&A** | Health question answering    |
| 📊 **Vitals**      | Track heart rate, SpO2, etc. |
| 🚨 **Alerts**      | Emergency detection          |

---

## 🔧 API Configuration

Edit `ai/config.py` to set your HuggingFace token:

```python
HF_TOKEN = "hf_your_token_here"
```

Or set environment variable:

```bash
set HF_API_TOKEN=hf_your_token_here
```

---

## 📱 API Endpoints

| Endpoint          | Method   | Description       |
| ----------------- | -------- | ----------------- |
| `/api/chat`       | POST     | Chat with NovaBot |
| `/api/vitals`     | POST/GET | Vital signs       |
| `/api/alerts`     | GET/PUT  | Manage alerts     |
| `/api/medication` | CRUD     | Medications       |

---

## 👥 Team

- Basant Awad (22101405)
- Nadira El-Sirafy (22101377)
- Noureen Yasser (22101109)
- Muhammad Mustafa (22101336)
- Ramez Asaad (22100506)

---

## 📊 Tech Stack

**Backend:** Flask, SQLAlchemy  
**AI:** Provider-specific (e.g. Gemini / Hugging Face in `ai/`); separate **LLM Backend** uses Ollama + Hugging Face — see `services/llm-backend/README.md`  
**Frontend:** Bootstrap 5, Chart.js  
**Database:** SQLite
