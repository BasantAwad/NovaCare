"""
HTTP client for NovaBot chat via the Flask LLM Backend (services/llm-backend).
Used by NovaBrain instead of calling Gemini in-process.
"""
import json
import os
import urllib.error
import urllib.request
from typing import Optional

from ai.config import Config


class FlaskLlmConversationalClient:
    """Same surface as ConversationalAI: generate_response, clear_history."""

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 120.0):
        self.base_url = (base_url or Config.NOVABOT_LLM_API_URL).rstrip("/")
        self.timeout_s = timeout_s
        prof = os.getenv("NOVABOT_LLM_PROFILE", "").strip().lower()
        self._llm_profile: Optional[str] = prof if prof in ("fast", "quality") else None
        print(f"[FlaskLlmConversationalClient] Chat API: {self.base_url}")

    def _post_json(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {e.code}: {body or e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM API unreachable ({url}): {e.reason}") from e

    def generate_response(self, user_input: str, emotion: Optional[str] = None) -> str:
        message = user_input
        if emotion and emotion not in ("neutral", "unknown"):
            message = f"(User's detected mood: {emotion}.)\n{user_input}"

        body: dict = {"message": message}
        if self._llm_profile:
            body["llm_profile"] = self._llm_profile

        try:
            data = self._post_json("/api/chat", body)
        except Exception as e:
            print(f"[FlaskLlmConversationalClient] Error: {e}")
            return (
                "I'm having trouble reaching the NovaBot chat service. "
                "Please ensure the LLM Backend is running (port 5000) and try again."
            )

        if data.get("status") == "error":
            return data.get("response") or data.get("error") or "Sorry, something went wrong."

        reply = (data.get("response") or "").strip()
        return reply or "I'm sorry, I didn't get a response. Please try again."

    def clear_history(self) -> None:
        try:
            self._post_json("/api/clear", {})
            print("[FlaskLlmConversationalClient] Server chat history cleared")
        except Exception as e:
            print(f"[FlaskLlmConversationalClient] clear_history: {e}")

    def train(self, dataset_path: str = None, **kwargs) -> bool:
        print("[FlaskLlmConversationalClient] Training not applicable (remote API)")
        return True
