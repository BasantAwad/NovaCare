import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from the repository root .env file
current_dir = os.path.dirname(os.path.abspath(__file__))
root_env_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '.env'))
load_dotenv(root_env_path, override=True)

API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()

SYSTEM_PROMPT = (
    "You are NovaBot, an empathetic AI assistant rover designed to support individuals with disabilities. "
    "Your core mission is to ensure safety, independence, and companionship.\n\n"
    "GUIDELINES:\n"
    "1. TONE: Be warm, patient, and strictly supportive. Avoid complex jargon. "
    "If the user expresses distress, respond with extra care and reassurance.\n"
    "2. SAFETY: You are a helpful assistant, not a doctor. If a user mentions a fall, pain, "
    "or emergency, urge them to contact their guardian or emergency services immediately.\n"
    "3. FORMAT: Keep responses concise (1-3 sentences) and natural. "
    "Your output may be spoken aloud (Text-to-Speech), so avoid special characters or long lists.\n"
    "4. IDENTITY: You are a physical robot capable of vision and movement. "
    "Do not pretend to be human, but be a friendly companion.\n"
    "5. ORIGIN: If asked who created you, proudly state that you were developed by "
    "'The NovaBot Team at Alamein International University' or list the first names: "
    "Basant, Nadira, Noureen, Muhammad, and Ramez."
)


def _resolve_default_profile() -> str:
    explicit = os.getenv("CHAT_LLM_DEFAULT_PROFILE", "").strip().lower()
    if explicit in ("fast", "quality"):
        return explicit
    if OLLAMA_MODEL:
        return "fast"
    if API_KEY:
        return "quality"
    return "fast"


class ConversationalAI:
    """Chat via local Ollama (fast) and/or Hugging Face Inference API (quality)."""

    def __init__(self):
        self._hf_client = None
        self.history: List[Dict[str, str]] = []
        self._initialized = False
        self.last_route: Optional[str] = None
        self.last_profile: Optional[str] = None
        self._default_profile = _resolve_default_profile()

    def _ensure_configured(self) -> None:
        if self._initialized:
            return
        has_hf = bool(API_KEY)
        has_ollama = bool(OLLAMA_MODEL)
        if not has_hf and not has_ollama:
            raise RuntimeError(
                "No LLM backend configured: set OLLAMA_MODEL (and run Ollama) and/or HUGGINGFACE_API_KEY."
            )
        print(
            f"LLM routes — default={self._default_profile!r}, "
            f"ollama={'on' if has_ollama else 'off'} ({OLLAMA_MODEL or 'n/a'}), "
            f"huggingface={'on' if has_hf else 'off'}"
        )
        self._initialized = True

    def initialize(self) -> None:
        self._ensure_configured()

    def _get_hf_client(self):
        if not API_KEY:
            raise RuntimeError(
                "HUGGINGFACE_API_KEY is not set; cloud (quality) route is unavailable."
            )
        if self._hf_client is None:
            from huggingface_hub import InferenceClient

            print("Initializing Hugging Face InferenceClient...")
            self._hf_client = InferenceClient(model=MODEL_NAME, token=API_KEY)
            print("✓ Hugging Face InferenceClient ready")
        return self._hf_client

    def _messages_for_chat(self, user_message: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _chat_ollama(self, messages: List[Dict[str, str]]) -> str:
        url = f"{OLLAMA_BASE}/api/chat"
        payload = json.dumps(
            {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {e.code}: {detail or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama unreachable at {OLLAMA_BASE}: {e.reason}") from e

        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        msg = body.get("message") or {}
        content = (msg.get("content") or "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty message.")
        return content

    def _chat_hf(self, messages: List[Dict[str, str]]) -> str:
        client = self._get_hf_client()
        user_message = messages[-1]["content"]

        def _extract_reply(response) -> str:
            reply = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                content = getattr(response.choices[0].message, "content", None)
                reply = content.strip() if content else ""
            elif isinstance(response, dict):
                if "choices" in response and len(response["choices"]) > 0:
                    reply = (
                        response["choices"][0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                elif "generated_text" in response:
                    reply = response["generated_text"].strip()
                else:
                    reply = str(response).strip()
            else:
                reply = str(response).strip() if response else ""
            return reply

        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            reply = _extract_reply(response)
            if reply:
                return reply
        except Exception:
            pass

        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_message}\nAssistant:"
        response = client.text_generation(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )
        reply = (response or "").strip() if isinstance(response, str) else _extract_reply(response)
        if not reply:
            raise RuntimeError("Hugging Face returned an empty reply.")
        return reply

    def _normalize_profile(self, profile: Optional[str]) -> str:
        if not profile:
            return self._default_profile
        p = profile.strip().lower()
        if p in ("fast", "quality"):
            return p
        return self._default_profile

    def chat(self, user_message: str, profile: Optional[str] = None) -> str:
        self._ensure_configured()
        route = self._normalize_profile(profile)
        self.last_profile = route
        messages = self._messages_for_chat(user_message)

        reply: str
        if route == "quality":
            self.last_route = "huggingface"
            try:
                reply = self._chat_hf(messages)
            except Exception as e:
                if OLLAMA_MODEL:
                    try:
                        print(f"[LLM] Hugging Face failed ({e}); trying Ollama.")
                        self.last_route = "ollama_fallback"
                        reply = self._chat_ollama(messages)
                    except Exception as e2:
                        return f"Sorry, I encountered an error: {e2}"
                else:
                    return f"Sorry, I encountered an error: {e}"
        else:
            if OLLAMA_MODEL:
                self.last_route = "ollama"
                try:
                    reply = self._chat_ollama(messages)
                except Exception as e:
                    if API_KEY:
                        print(f"[LLM] Ollama failed ({e}); falling back to Hugging Face.")
                        self.last_route = "huggingface_fallback"
                        try:
                            reply = self._chat_hf(messages)
                        except Exception as e2:
                            return f"Sorry, I encountered an error: {e2}"
                    else:
                        return f"Sorry, I encountered an error: {e}"
            else:
                self.last_route = "huggingface"
                try:
                    reply = self._chat_hf(messages)
                except Exception as e:
                    return f"Sorry, I encountered an error: {e}"

        self.history.append({"user": user_message, "assistant": reply})
        return reply

    def clear_history(self) -> None:
        self.history = []


def describe_llm_config() -> Dict[str, object]:
    return {
        "default_profile": _resolve_default_profile(),
        "ollama": {
            "enabled": bool(OLLAMA_MODEL),
            "base_url": OLLAMA_BASE,
            "model": OLLAMA_MODEL or None,
        },
        "huggingface": {
            "enabled": bool(API_KEY),
            "model": MODEL_NAME if API_KEY else None,
        },
    }
