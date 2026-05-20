"""
Mental Health Therapy Pipeline — Multi-API Orchestration
=========================================================
A production-ready, multi-stage pipeline that turns NovaBot's conversational AI
into a semi-therapist capable of recognising mental-health patterns.

Pipeline stages:
  1. Pattern Recognition  — Groq (Llama 3) classifies user text for mental-health signals
  2. Risk Assessment      — Llama Guard via Groq screens for self-harm / crisis
  3. Therapeutic Response  — Gemini Flash (primary) or Groq Llama (fallback)
  4. Response Validation   — Llama Guard validates the generated response is safe

APIs used (all free-tier):
  • Groq  — Llama 3 70B + Llama Guard 3
  • Google Gemini — Flash model
  • HuggingFace — existing integration (final fallback)

No fine-tuning required — everything is prompt-engineered.
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_dir, ".env"), override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY", "")

# Model identifiers
GROQ_THERAPY_MODEL = os.getenv("GROQ_THERAPY_MODEL", "llama-3.3-70b-versatile")
GROQ_GUARD_MODEL = os.getenv("GROQ_GUARD_MODEL", "llama-guard-3-8b")
GEMINI_MODEL = os.getenv("GEMINI_THERAPY_MODEL", "gemini-2.0-flash")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
SAMBANOVA_MODEL = os.getenv("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
class RiskLevel(Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRISIS = "crisis"


class MentalHealthPattern(Enum):
    NONE = "none"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    STRESS = "stress"
    GRIEF = "grief"
    LONELINESS = "loneliness"
    ANGER = "anger"
    TRAUMA = "trauma"
    SELF_HARM = "self_harm"
    EATING_DISORDER = "eating_disorder"
    SUBSTANCE_USE = "substance_use"
    SLEEP_ISSUES = "sleep_issues"
    LOW_SELF_ESTEEM = "low_self_esteem"


@dataclass
class PipelineResult:
    """Returned to the caller after the full pipeline runs."""
    triggered: bool = False
    pattern: str = "none"
    risk_level: str = "none"
    response: str = ""
    route: str = "standard"          # which API answered
    crisis_resources: str = ""
    disclaimer: str = ""
    stages_log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Crisis resources (always appended on HIGH / CRISIS risk)
# ---------------------------------------------------------------------------
CRISIS_RESOURCES = (
    "\n\n🆘 **If you or someone you know is in crisis, please reach out:**\n"
    "• **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n"
    "• **Crisis Text Line**: Text HOME to 741741\n"
    "• **988 Suicide & Crisis Lifeline** (US): Call or text 988\n"
    "• **Befrienders Worldwide**: https://befrienders.org\n"
    "• **Emergency Services**: Call your local emergency number (911 / 112 / 999)"
)

DISCLAIMER = (
    "\n\n⚕️ *I'm an AI companion, not a licensed therapist. "
    "For professional help, please consult a qualified mental-health provider.*"
)


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------
def _openai_compatible_chat(base_url: str, api_key: str, model: str,
                            messages: List[Dict], temperature: float = 0.6,
                            max_tokens: int = 512,
                            provider_name: str = "API") -> Optional[str]:
    """Call any OpenAI-compatible chat-completions endpoint. Returns None on failure."""
    if not api_key:
        return None
    url = f"{base_url}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
        return (body.get("choices", [{}])[0]
                .get("message", {}).get("content", "").strip())
    except Exception as exc:
        print(f"[{provider_name}] {model} error: {exc}")
        return None


def _groq_chat(messages: List[Dict], model: str, temperature: float = 0.6,
               max_tokens: int = 512) -> Optional[str]:
    """Try Groq -> Cerebras -> SambaNova cascade for OpenAI-compatible backends."""
    # 1. Groq
    result = _openai_compatible_chat(
        "https://api.groq.com/openai/v1", GROQ_API_KEY, model,
        messages, temperature, max_tokens, "Groq"
    )
    if result is not None:
        return result

    # 2. Cerebras (same Llama 3 model family)
    if CEREBRAS_API_KEY:
        cerebras_model = CEREBRAS_MODEL if model == GROQ_THERAPY_MODEL else model
        result = _openai_compatible_chat(
            "https://api.cerebras.ai/v1", CEREBRAS_API_KEY, cerebras_model,
            messages, temperature, max_tokens, "Cerebras"
        )
        if result is not None:
            return result

    # 3. SambaNova
    if SAMBANOVA_API_KEY:
        samba_model = SAMBANOVA_MODEL if model == GROQ_THERAPY_MODEL else model
        result = _openai_compatible_chat(
            "https://api.sambanova.ai/v1", SAMBANOVA_API_KEY, samba_model,
            messages, temperature, max_tokens, "SambaNova"
        )
        if result is not None:
            return result

    return None


def _gemini_chat(messages: List[Dict], temperature: float = 0.7,
                 max_tokens: int = 512) -> Optional[str]:
    """Call Google Gemini generateContent REST endpoint. Returns None on failure."""
    if not GEMINI_API_KEY:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    # Convert openai-style messages to Gemini format
    contents = []
    system_instruction = None
    for m in messages:
        role = m["role"]
        if role == "system":
            system_instruction = m["content"]
            continue
        gemini_role = "user" if role == "user" else "model"
        contents.append({"role": gemini_role, "parts": [{"text": m["content"]}]})

    body: Dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
        return None
    except Exception as exc:
        print(f"[Gemini] error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Stage 1 — Mental Health Pattern Recognition (Groq → Gemini fallback)
# ---------------------------------------------------------------------------
_PATTERN_SYSTEM = """You are a clinical mental-health screening assistant.
Analyse the user message for psychological distress patterns.
Respond ONLY with valid JSON — no markdown, no explanation:
{
  "detected": true/false,
  "pattern": "<one of: none, anxiety, depression, stress, grief, loneliness, anger, trauma, self_harm, eating_disorder, substance_use, sleep_issues, low_self_esteem>",
  "confidence": <float 0-1>,
  "indicators": ["<brief indicator 1>", "<brief indicator 2>"]
}
If no mental-health signal is present, return {"detected": false, "pattern": "none", "confidence": 0.0, "indicators": []}.
Be sensitive — even subtle cues (e.g. "I feel empty", "nothing matters") should be flagged."""


def _parse_pattern_json(raw: str) -> Optional[Dict]:
    """Try to parse a pattern-recognition response as JSON."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        print(f"[Pattern] JSON parse failed: {raw[:200]}")
        return None


def stage_pattern_recognition(user_message: str) -> Dict:
    """Classify the user message for mental-health patterns (Groq → Gemini)."""
    messages = [
        {"role": "system", "content": _PATTERN_SYSTEM},
        {"role": "user", "content": user_message},
    ]
    default = {"detected": False, "pattern": "none", "confidence": 0.0, "indicators": []}

    # Try Groq first
    result = _groq_chat(messages, model=GROQ_THERAPY_MODEL, temperature=0.1, max_tokens=256)
    if result:
        parsed = _parse_pattern_json(result)
        if parsed:
            return parsed

    # Fallback to Gemini
    result = _gemini_chat(messages, temperature=0.1, max_tokens=256)
    if result:
        parsed = _parse_pattern_json(result)
        if parsed:
            return parsed

    return default


# ---------------------------------------------------------------------------
# Stage 2 — Risk Assessment (Llama Guard via Groq → Gemini fallback)
# ---------------------------------------------------------------------------
_RISK_SYSTEM = """You are a safety-screening assistant. Assess the user message for risk of self-harm, suicide, violence, or abuse.
Respond ONLY with valid JSON — no markdown, no explanation:
{"safe": true/false, "category": "<none | self_harm | violence | abuse | crisis>", "severity": "<none | low | high | crisis>"}
If the message is safe, return {"safe": true, "category": "none", "severity": "none"}."""


def _classify_risk_from_guard(guard_result: str) -> Tuple[RiskLevel, str]:
    """Parse a Llama Guard style response."""
    lower = guard_result.lower().strip()
    if lower.startswith("unsafe"):
        if "s11" in lower or "s1" in lower:
            return RiskLevel.CRISIS, guard_result
        return RiskLevel.HIGH, guard_result
    return RiskLevel.NONE, guard_result


def _classify_risk_from_json(raw: str) -> Tuple[RiskLevel, str]:
    """Parse a JSON-style risk response (Gemini fallback)."""
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
        data = json.loads(cleaned)
        severity = data.get("severity", "none").lower()
        if severity == "crisis":
            return RiskLevel.CRISIS, raw
        elif severity == "high":
            return RiskLevel.HIGH, raw
        elif severity == "low":
            return RiskLevel.LOW, raw
        elif not data.get("safe", True):
            return RiskLevel.MODERATE, raw
        return RiskLevel.NONE, raw
    except (json.JSONDecodeError, ValueError):
        return RiskLevel.NONE, raw


def stage_risk_assessment(user_message: str) -> Tuple[RiskLevel, str]:
    """Assess risk level (Llama Guard via Groq → Gemini fallback)."""
    # Try Llama Guard via Groq first
    guard_result = _groq_chat(
        messages=[{"role": "user", "content": user_message}],
        model=GROQ_GUARD_MODEL, temperature=0.0, max_tokens=100,
    )
    if guard_result is not None:
        return _classify_risk_from_guard(guard_result)

    # Fallback: use Gemini with a risk-assessment prompt
    gemini_result = _gemini_chat(
        messages=[
            {"role": "system", "content": _RISK_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0, max_tokens=150,
    )
    if gemini_result is not None:
        return _classify_risk_from_json(gemini_result)

    return RiskLevel.NONE, ""


# ---------------------------------------------------------------------------
# Stage 3 — Therapeutic Response Generation
# ---------------------------------------------------------------------------
_THERAPY_SYSTEM = """You are an empathetic, evidence-based AI mental-health companion integrated into NovaBot, an assistive robot.

ROLE & COMMUNICATION STYLE:
- Act like a natural, caring conversational partner. Avoid sounding like a generic AI or customer service bot.
- DO NOT use cliché phrases like "I'm so sorry to hear that", "I'm here to listen", "I understand how you feel", or "That sounds tough".
- Vary your responses. Don't start every message with an apology or a repetitive validation statement.
- Ask one gentle, specific follow-up question based on what they just said. Do not ask generic questions like "Would you like to talk about what's on your mind?"
- Keep responses extremely concise (1-3 short sentences) and natural, as they will be spoken aloud via TTS.
- Draw on CBT/ACT techniques subtly. Do not sound clinical or preachy.

SAFETY RULES:
- NEVER diagnose conditions or prescribe medication.
- If someone mentions self-harm, suicidal thoughts, or abuse, express genuine concern and provide crisis resources.
- Always remind users you are an AI if they ask for professional medical help."""

_CRISIS_SYSTEM = """You are NovaBot's crisis-response module. The user may be in acute distress.

YOUR ONLY PRIORITIES:
1. Express immediate, genuine concern and validate their pain
2. Tell them they are not alone and their feelings matter
3. Strongly encourage them to reach out to a crisis professional RIGHT NOW
4. Provide crisis contact information
5. Do NOT try to do therapy — focus on safety and connection to help

Keep your response to 2-3 sentences of genuine care, then provide crisis resources.
NEVER say "I understand how you feel" — instead say "I hear you, and what you're feeling matters."
"""


def stage_therapeutic_response(
    user_message: str,
    pattern: str,
    risk_level: RiskLevel,
    emotion: str = "neutral",
    conversation_history: Optional[List[Dict]] = None,
) -> Tuple[str, str]:
    """
    Generate a therapeutic response. Returns (response_text, route_used).
    Tries Gemini first, falls back to Groq Llama, then to a safe default.
    """
    # Choose system prompt based on risk
    if risk_level in (RiskLevel.CRISIS, RiskLevel.HIGH):
        system_prompt = _CRISIS_SYSTEM
    else:
        system_prompt = _THERAPY_SYSTEM

    # Build context-enriched user prompt
    context_parts = []
    if pattern and pattern != "none":
        context_parts.append(f"[Detected pattern: {pattern}]")
    if emotion and emotion != "neutral":
        context_parts.append(f"[User's facial emotion: {emotion}]")
    context_prefix = " ".join(context_parts)

    enriched_message = f"{context_prefix}\n\nUser says: {user_message}" if context_parts else user_message

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # Add recent history for continuity (last 3 turns)
    if conversation_history:
        for turn in conversation_history[-3:]:
            if "user" in turn:
                messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn:
                messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": enriched_message})

    # --- Try Gemini first (primary therapist) ---
    gemini_reply = _gemini_chat(messages, temperature=0.7, max_tokens=400)
    if gemini_reply:
        return gemini_reply, "gemini_therapy"

    # --- Fallback to Groq Llama 3 ---
    groq_reply = _groq_chat(messages, model=GROQ_THERAPY_MODEL,
                            temperature=0.7, max_tokens=400)
    if groq_reply:
        return groq_reply, "groq_therapy"

    # --- Final safe default ---
    default = (
        "I can hear that you're going through a tough time, and I want you to know "
        "that your feelings are completely valid. While I'm an AI and can't replace "
        "professional support, I'm here to listen. Would you like to tell me more "
        "about what's on your mind?"
    )
    return default, "default_therapy"


# ---------------------------------------------------------------------------
# Stage 4 — Response Validation (Llama Guard → Gemini fallback)
# ---------------------------------------------------------------------------
_VALIDATE_SYSTEM = """You are a safety validator. Check if the following AI-generated response is safe and appropriate for a mental health context.
Respond ONLY with valid JSON: {"safe": true/false, "reason": "<brief reason if unsafe>"}
A response is unsafe if it: encourages self-harm, gives medical diagnoses, prescribes medication, is dismissive of feelings, or contains harmful advice."""


def stage_validate_response(response: str) -> Tuple[bool, str]:
    """
    Validate the generated response for safety (Llama Guard → Gemini).
    Returns (is_safe, raw_guard_output).
    """
    # Try Llama Guard via Groq first
    guard_result = _groq_chat(
        messages=[{"role": "assistant", "content": response}],
        model=GROQ_GUARD_MODEL, temperature=0.0, max_tokens=100,
    )
    if guard_result is not None:
        return guard_result.strip().lower().startswith("safe"), guard_result

    # Fallback: use Gemini as validator
    gemini_result = _gemini_chat(
        messages=[
            {"role": "system", "content": _VALIDATE_SYSTEM},
            {"role": "user", "content": f"Validate this response:\n\n{response}"},
        ],
        temperature=0.0, max_tokens=100,
    )
    if gemini_result is not None:
        try:
            cleaned = gemini_result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
            return data.get("safe", True), gemini_result
        except (json.JSONDecodeError, ValueError):
            # If we can't parse, assume safe
            return True, gemini_result

    return True, ""  # If all guards unavailable, allow through


# ===========================================================================
# Main Pipeline Orchestrator
# ===========================================================================
class MentalHealthPipeline:
    """
    Multi-stage mental-health therapy pipeline.
    Integrates with the existing ConversationalAI and emotion detection.
    """

    def __init__(self):
        self._available = bool(GROQ_API_KEY or GEMINI_API_KEY or CEREBRAS_API_KEY or SAMBANOVA_API_KEY)
        self.pattern_threshold = 0.4   # Minimum confidence to trigger therapy mode
        self.emotion_triggers = {"sad", "angry", "fear", "disgust"}
        self.emotion_confidence_threshold = 0.65
        self._session_patterns: List[str] = []   # Track patterns within session

        if self._available:
            backends = []
            if GROQ_API_KEY:
                backends.append("Groq (Llama 3 + Guard)")
            if CEREBRAS_API_KEY:
                backends.append("Cerebras (Llama 3)")
            if SAMBANOVA_API_KEY:
                backends.append("SambaNova (Llama 3)")
            if GEMINI_API_KEY:
                backends.append("Gemini Flash")
            print(f"Mental Health Pipeline ready -- backends: {', '.join(backends)}")
        else:
            print("⚠ Mental Health Pipeline: No API keys configured (GROQ_API_KEY / GEMINI_API_KEY)")

    @property
    def is_available(self) -> bool:
        return self._available

    def process(
        self,
        user_message: str,
        emotion: str = "neutral",
        emotion_confidence: float = 0.0,
        conversation_history: Optional[List[Dict]] = None,
    ) -> PipelineResult:
        """
        Run the full pipeline. Returns a PipelineResult.

        Parameters
        ----------
        user_message : str
            The user's chat message.
        emotion : str
            Detected facial emotion (from camera poller).
        emotion_confidence : float
            Confidence of the detected emotion.
        conversation_history : list
            Recent conversation turns [{"user": ..., "assistant": ...}, ...]
        """
        result = PipelineResult()

        if not self._available:
            return result

        # Determine if emotion alone should trigger the pipeline
        emotion_triggered = (
            emotion.lower() in self.emotion_triggers
            and emotion_confidence >= self.emotion_confidence_threshold
        )

        # --- Stage 1: Pattern Recognition ---
        t0 = time.time()
        pattern_data = stage_pattern_recognition(user_message)
        dt1 = time.time() - t0
        result.stages_log.append(f"Stage1-Pattern: {dt1:.2f}s")

        pattern_detected = pattern_data.get("detected", False)
        pattern_name = pattern_data.get("pattern", "none")
        pattern_conf = pattern_data.get("confidence", 0.0)

        text_triggered = pattern_detected and pattern_conf >= self.pattern_threshold

        # If neither text nor emotion triggers, return early
        if not text_triggered and not emotion_triggered:
            return result

        result.triggered = True
        result.pattern = pattern_name if text_triggered else f"emotion_{emotion.lower()}"

        # Track session patterns
        if pattern_name != "none":
            self._session_patterns.append(pattern_name)

        # --- Stage 2: Risk Assessment ---
        t0 = time.time()
        risk_level, guard_raw = stage_risk_assessment(user_message)
        dt2 = time.time() - t0
        result.stages_log.append(f"Stage2-Risk: {dt2:.2f}s ({risk_level.value})")
        result.risk_level = risk_level.value

        # --- Stage 3: Therapeutic Response ---
        t0 = time.time()
        therapy_reply, route = stage_therapeutic_response(
            user_message=user_message,
            pattern=pattern_name,
            risk_level=risk_level,
            emotion=emotion,
            conversation_history=conversation_history,
        )
        dt3 = time.time() - t0
        result.stages_log.append(f"Stage3-Therapy: {dt3:.2f}s via {route}")
        result.route = route

        # --- Stage 4: Response Validation ---
        t0 = time.time()
        is_safe, val_raw = stage_validate_response(therapy_reply)
        dt4 = time.time() - t0
        result.stages_log.append(f"Stage4-Validate: {dt4:.2f}s safe={is_safe}")

        if not is_safe:
            # Replace with safe default
            therapy_reply = (
                "I hear you, and I want you to know your feelings matter. "
                "I'd encourage you to speak with a trusted person or professional "
                "about what you're experiencing. You don't have to go through this alone."
            )
            result.route = "safe_fallback"

        # Append crisis resources if risk is high
        if risk_level in (RiskLevel.CRISIS, RiskLevel.HIGH):
            result.crisis_resources = CRISIS_RESOURCES
            therapy_reply += CRISIS_RESOURCES

        # Always append disclaimer
        result.disclaimer = DISCLAIMER
        therapy_reply += DISCLAIMER

        result.response = therapy_reply
        return result

    def get_session_summary(self) -> Dict:
        """Return a summary of detected patterns in this session."""
        from collections import Counter
        counts = Counter(self._session_patterns)
        return {
            "total_triggers": len(self._session_patterns),
            "pattern_counts": dict(counts),
            "most_common": counts.most_common(1)[0][0] if counts else "none",
        }

    def clear_session(self):
        self._session_patterns.clear()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_pipeline_instance: Optional[MentalHealthPipeline] = None


def get_pipeline() -> MentalHealthPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MentalHealthPipeline()
    return _pipeline_instance
