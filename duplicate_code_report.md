# Duplicate Code Report (report-only)

Status: **In progress** — no code moves or deletions yet.

## Initial high-confidence candidates
- **Conversational AI wrappers**
  - `LLMs/conversational_ai.py`
  - `services/llm-backend/LLMs/conversational_ai.py`
- **Emotion detection pipeline modules**
  - `Emotion_Detection/*` (root)
  - Potential overlapping implementations inside service folders

## Assets duplication candidates
- NovaBot client JS / STT / TTS assets
  - root `static/js/` equivalents
  - `services/llm-backend/static/js/`

## Next step
Run a targeted search for duplicated module/function names and build a canonicalization plan (copy-first + shims).

