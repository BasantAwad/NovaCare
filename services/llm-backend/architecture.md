# 🤖 LLM Backend — Architecture
**Dual-Routed Conversational AI with Facial Emotion Detection**

The LLM Backend is a multi-capability Flask service that powers NovaBot's intelligence. It doesn't just forward messages to an LLM; it **routes** requests through a dual-backend system with automatic failover, **detects** facial emotions using a Vision Transformer, and **exposes** a unified API consumed by both the Next.js frontend and a bundled test page.

---

## 🏗️ Architecture Overview

The service operates two independent AI pipelines behind a single Flask API:

```mermaid
graph TD
    A[🖥️ Client Request] --> B[Flask API Server]
    
    B --> C{Endpoint?}
    C -->|POST /api/chat| D[Profile Router]
    C -->|POST /api/emotion/detect| E[Emotion Pipeline]
    C -->|GET /health| F[Health Check]
    
    D --> G{Profile?}
    G -->|fast| H[Ollama Local]
    G -->|quality| I[HuggingFace API]
    H -->|fail| I
    I -->|fail| H
    
    H --> J[✅ AI Response]
    I --> J
    
    E --> K[Base64 Decode]
    K --> L[OpenCV Face Detect]
    L --> M[ViT Inference]
    M --> N[✅ Emotion + Confidence]
```

---

## 🧩 Core Components

1. **🔀 Profile Router**: Inspects each request for `llm_profile` or `prefer_quality` and routes to the appropriate LLM backend. Falls back automatically on failure.
2. **⚡ Ollama Client** (`fast`): Calls a locally-running Ollama instance via `urllib` (zero extra dependencies). Optimized for low-latency, free inference.
3. **☁️ HuggingFace Client** (`quality`): Calls the HuggingFace Inference API for higher-quality responses. Uses `InferenceClient` with chat-completion and text-generation fallback.
4. **😊 Face Emotion Analyzer**: A lazy-loaded ViT model (`trpakov/vit-face-expression`) that detects 7 emotion categories from facial images. Uses OpenCV Haar Cascade for face cropping.
5. **🧠 System Prompt**: A carefully crafted persona prompt that defines NovaBot as an empathetic rover assistant — consistent across all conversations.
6. **📊 Health Reporter**: Exposes LLM configuration state (which backends are enabled, default profile, model names) via `GET /health`.

---

## 🔀 Dual LLM Routing

The routing system ensures maximum availability — if the primary backend fails, the secondary is tried automatically.

```mermaid
graph LR
    subgraph "fast profile"
        F1[Ollama Local] -->|fail| F2[HuggingFace Fallback]
    end
    
    subgraph "quality profile"
        Q1[HuggingFace API] -->|fail| Q2[Ollama Fallback]
    end
    
    R[Request] --> P{llm_profile?}
    P -->|fast / default| F1
    P -->|quality| Q1
```

### Default Profile Resolution

```mermaid
graph TD
    A{CHAT_LLM_DEFAULT_PROFILE set?} -->|Yes| B[Use its value]
    A -->|No| C{OLLAMA_MODEL set?}
    C -->|Yes| D["Default: fast"]
    C -->|No| E{HUGGINGFACE_API_KEY set?}
    E -->|Yes| F["Default: quality"]
    E -->|No| G["Default: fast"]
```

### Per-Request Override

| Field | Type | Effect |
|-------|------|--------|
| `llm_profile` | `"fast"` \| `"quality"` | Sets the routing profile for this request |
| `prefer_quality` | `boolean` | If `true`, equivalent to `llm_profile: "quality"` |

### Response Tracking

Every response includes `llm_profile` (profile used) and `llm_route` (actual backend hit):

| `llm_route` value | Meaning |
|--------------------|---------|
| `ollama` | Primary Ollama succeeded |
| `huggingface` | Primary HuggingFace succeeded |
| `huggingface_fallback` | Ollama failed → HuggingFace succeeded |
| `ollama_fallback` | HuggingFace failed → Ollama succeeded |

---

## 😊 Emotion Detection Pipeline

```mermaid
graph TD
    A["📷 Base64 Image"] --> B["Decode → NumPy Array"]
    B --> C["OpenCV Haar Cascade"]
    C --> D{Face Detected?}
    D -->|Yes| E["Crop + 15% Padding"]
    D -->|No| F["Use Full Image"]
    E --> G["Preprocess: BGR→RGB,<br/>Resize 224×224, → PIL"]
    F --> G
    G --> H["ViT Model Inference<br/>(trpakov/vit-face-expression)"]
    H --> I["Softmax → 7 Classes"]
    I --> J["✅ Emotion + Confidence + All Scores"]
```

| Label | Categories |
|-------|-----------|
| **7 Emotions** | angry, disgust, fear, happy, sad, surprise, neutral |
| **Model** | Vision Transformer fine-tuned on FER2013 |
| **Loading** | Lazy — loaded on first `/api/emotion/detect` call to avoid slow startup |

---

## 📁 Module Breakdown

| Module | File | Purpose |
|--------|------|---------|
| **API Server** | `api_server.py` | Flask app: routes, CORS, lazy emotion loading |
| **Entry Point** | `start_server.py` | `python start_server.py` → Flask on port 5000 |
| **Chat Engine** | `LLMs/conversational_ai.py` | `ConversationalAI`: dual routing, system prompt, chat history, Ollama + HF clients, `describe_llm_config()` |
| **Emotion Engine** | `emotion_detection.py` | `FaceEmotionAnalyzer`: Haar face detection, ViT inference, base64 handling. Singleton via `get_analyzer()` |
| **Utilities** | `utils/` | Shared utility functions |
| **Test UI** | `templates/test_novabot.html` | Bundled HTML test page (voice/text/TTS) |
| **Client JS** | `static/js/` | NovaBotClient, STT, TTS for the test page |

---

## 🚀 Request Lifecycle

### Phase 1: Chat Request
Client sends `POST /api/chat` with `{"message": "Hello", "llm_profile": "fast"}`.

### Phase 2: Profile Resolution
Router resolves profile → `fast`. Checks if Ollama is configured.

### Phase 3: LLM Inference
Ollama is called via HTTP (`POST /api/chat` on `127.0.0.1:11434`). System prompt + user message are sent. If Ollama fails, HuggingFace is tried as fallback.

### Phase 4: Response
```json
{
  "response": "Hello! I'm doing great. How can I help you today?",
  "status": "success",
  "llm_profile": "fast",
  "llm_route": "ollama"
}
```

---

## 🎯 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual LLM routing** | Ollama is fast/free/local; HF API is higher quality. Fallback ensures availability |
| **Lazy emotion loading** | ViT model is large (~350MB) — loading on first request avoids slow startup if unused |
| **System prompt baked in** | NovaBot persona (empathetic rover) is consistent across all conversations |
| **No external state** | Chat history is in-memory per process — stateless for simplicity |
| **`urllib` over `requests`** | Ollama client uses stdlib to avoid an extra dependency |
| **Singleton analyzer** | `get_analyzer()` ensures the ViT model is loaded only once across all requests |
