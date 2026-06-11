import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file in parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, ".env")
load_dotenv(env_path, override=True)

API_KEY = os.getenv("HUGGINGFACE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()

SYSTEM_PROMPT = (
    "You are NovaBot, an empathetic AI assistant rover designed to support individuals with disabilities. "
    "Your core mission is to ensure safety, independence, and companionship.\n\n"
    "GUIDELINES:\n"
    "1. TONE: Be warm, patient, and strictly supportive. Avoid complex jargon. "
    "If the user expresses distress, respond with extra care.\n"
    "2. SAFETY: If a user mentions a fall, pain, or emergency, urge them to contact their guardian or emergency services.\n"
    "3. FORMAT: Keep responses concise (1-3 sentences) and natural. Avoid special characters in the text.\n"
    "4. IDENTITY & AGENCY: You are a physical robot capable of vision, movement, and system controls. "
    "Do NOT say you are an AI or that you cannot make calls, play music, or navigate. You have full capability to trigger these tools. "
    "Always respond affirmatively to user requests (e.g., 'I am calling your guardian now' or 'Navigating to the bedroom now').\n"
    "5. ORIGIN: Developed by 'The NovaBot Team at Alamein International University' (Basant, Nadira, Noureen, Muhammad, and Ramez).\n\n"
    "IMPORTANT: You must respond in a valid JSON format. Do not write any conversational text outside the JSON block. "
    "Your response must follow this schema:\n"
    "{\n"
    '  "response": "Your friendly conversational reply to show and speak.",\n'
    '  "actions": [\n'
    "    {\n"
    '      "name": "action_name",\n'
    '      "parameters": { ... }\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "AVAILABLE ACTIONS:\n"
    "- 'navigate': parameter 'destination' (one of: 'kitchen', 'bathroom', 'living room', 'bedroom', 'dining room', 'front door')\n"
    "- 'play_music': parameter 'mood' (one of: 'relaxing', 'upbeat', 'focus') and optional 'track'\n"
    "- 'pause_music': no parameters\n"
    "- 'call_guardian': no parameters\n"
    "- 'show_medications': no parameters\n"
    "- 'trigger_emergency': no parameters\n"
    "- 'check_health': no parameters\n\n"
    "Only trigger actions when requested or when safety/distress mandates it. Otherwise, keep the 'actions' list empty."
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
        import sys
        sys.stderr.write(f"[HF Client Init] API_KEY exists: {bool(API_KEY)}, MODEL_NAME: {MODEL_NAME}\n")
        sys.stderr.flush()
        if not API_KEY:
            raise RuntimeError(
                "HUGGINGFACE_API_KEY is not set; cloud (quality) route is unavailable."
            )
        if self._hf_client is None:
            from huggingface_hub import InferenceClient

            sys.stderr.write("Initializing Hugging Face InferenceClient...\n")
            sys.stderr.flush()
            self._hf_client = InferenceClient(model=MODEL_NAME, token=API_KEY)
            sys.stderr.write("✓ Hugging Face InferenceClient ready\n")
            sys.stderr.flush()
        return self._hf_client

    def _messages_for_chat(self, user_message: str, live_context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "The following is real-time, database-backed information about the rover and the patient's schedule. "
                    "Use it to answer direct questions accurately. Do not make up information that contradicts this:\n\n"
                    f"{live_context}"
                ),
            },
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
        except Exception as e:
            import sys
            sys.stderr.write(f"[HF Chat Error] chat_completion failed: {e}\n")
            sys.stderr.flush()
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

    def chat(self, user_message: str, profile: Optional[str] = None, emotion: str = "unknown", confidence: float = 0.0) -> dict:
        prefix_to_add = ""
        try:
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from mental_health_integration import get_orchestrator
            orchestrator = get_orchestrator()
            bypass, hf_reply, prefix = orchestrator.process(user_message, conversation_history=self.history, frontend_emotion=emotion, frontend_confidence=confidence)
            
            if bypass:
                self.history.append({"user": user_message, "assistant": hf_reply})
                self.last_profile = "quality"
                self.last_route = "huggingface_mental_health"
                return hf_reply
            elif prefix:
                prefix_to_add = prefix
        except Exception as e:
            print(f"Warning: MentalHealthOrchestrator failed: {e}")

        self._ensure_configured()
        route = self._normalize_profile(profile)
        self.last_profile = route

        print(f"\n[CHAT] [LLM Backend] --- NEW CHAT SESSION ---")
        print(f"   User Message: '{user_message}'")
        print(f"   Requested Profile: {profile or 'None (Default)'} -> Selected Route: {route.upper()}")

        # Fetch live context for RAG (with intelligent query routing)
        from datetime import datetime
        from utils.rag_helper import rag_manager
        live_data, routed_sources = rag_manager.get_routed_context(user_message, rover_id="RV001")
        
        # Format live context as a string
        context_parts = [
            f"Current Date/Time: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
        ]
        
        # Medications
        meds = live_data.get("medications", [])
        if meds:
            med_list = []
            for m in meds:
                status_str = f"taken at {m['taken_at']}" if m.get('status') == 'taken' else m.get('status', 'pending')
                med_list.append(f"- {m.get('medication_name', 'Unknown')} ({m.get('dosage', '')}): scheduled for {m.get('scheduled_time', '')} (Status: {status_str})")
            context_parts.append("Today's Medications:\n" + "\n".join(med_list))
        elif "medications" in routed_sources:
            context_parts.append("Today's Medications: None scheduled.")
            
        # Vitals
        vitals = live_data.get("vitals")
        if vitals:
            context_parts.append(
                f"Latest Vitals (recorded at {vitals.get('measured_at', 'today')}):\n"
                f"- Heart Rate: {vitals.get('heart_rate')} BPM\n"
                f"- Blood Pressure: {vitals.get('blood_pressure')}\n"
                f"- Oxygen Saturation: {vitals.get('oxygen_level')}%\n"
                f"- Body Temperature: {vitals.get('temperature', 36.6)}C"
            )

        # Vitals Trend
        vitals_trend = live_data.get("vitals_trend", [])
        if vitals_trend:
            hrs = [v.get("heart_rate") for v in vitals_trend if v.get("heart_rate")]
            if hrs:
                avg_hr = sum(hrs) / len(hrs)
                max_hr = max(hrs)
                min_hr = min(hrs)
                context_parts.append(
                    f"Vitals Trend (past 7 days, {len(vitals_trend)} readings):\n"
                    f"- Avg Heart Rate: {avg_hr:.0f} BPM (Min: {min_hr}, Max: {max_hr})"
                )

        # Appointments
        appointments = live_data.get("appointments", [])
        if appointments:
            appt_list = []
            for a in appointments:
                doctor_name = f"Dr. {a.get('doctor_first_name', '')} {a.get('doctor_last_name', '')}".strip()
                appt_list.append(f"- {a.get('appointment_type', 'Appointment')} with {doctor_name} ({a.get('specialization', '')}) on {a.get('scheduled_at', '')} - Status: {a.get('status', '')}")
            context_parts.append("Appointments:\n" + "\n".join(appt_list))
        elif "appointments" in routed_sources:
            context_parts.append("Appointments: None found.")

        # Health Conditions
        conditions = live_data.get("health_conditions", [])
        if conditions:
            cond_list = [f"- {c.get('condition_name', '')} (Severity: {c.get('severity', 'unknown')})" for c in conditions]
            context_parts.append("Diagnosed Health Conditions:\n" + "\n".join(cond_list))

        # Allergies
        allergies = live_data.get("allergies", [])
        if allergies:
            allergy_list = [f"- {a.get('allergy_name', '')} ({a.get('allergy_type', '')}) - Severity: {a.get('severity', '')}" for a in allergies]
            context_parts.append("Known Allergies:\n" + "\n".join(allergy_list))

        # Emergency Contacts
        contacts = live_data.get("emergency_contacts", [])
        if contacts:
            contact_list = [f"- {c.get('name', '')} ({c.get('relationship', '')}): {c.get('phone', '')}{' [PRIMARY]' if c.get('is_primary') else ''}" for c in contacts]
            context_parts.append("Emergency Contacts:\n" + "\n".join(contact_list))

        # Medical Notes
        notes = live_data.get("medical_notes", [])
        if notes:
            note_list = []
            for n in notes:
                doctor_name = f"Dr. {n.get('doctor_first_name', '')} {n.get('doctor_last_name', '')}".strip()
                note_list.append(f"- [{n.get('note_type', 'Note')}] {n.get('created_at', '')}: {n.get('note_content', '')[:200]}")
            context_parts.append("Recent Medical Notes:\n" + "\n".join(note_list))

        # Emotion History
        emotions = live_data.get("emotion_history", [])
        if emotions:
            emo_list = [f"- {e.get('date', '')}: {e.get('primary_emotion', '')} (sentiment: {e.get('avg_sentiment', 0):.2f}){' [DISTRESS]' if e.get('distress_detected') else ''}" for e in emotions[:5]]
            context_parts.append("Recent Emotion History:\n" + "\n".join(emo_list))

        # Notifications
        notifs = live_data.get("notifications", [])
        if notifs:
            unread = [n for n in notifs if not n.get("is_read")]
            if unread:
                notif_list = [f"- {n.get('title', '')}: {n.get('message', '')}" for n in unread[:5]]
                context_parts.append(f"Unread Notifications ({len(unread)}):\n" + "\n".join(notif_list))

        # Hydration
        hydration = live_data.get("hydration")
        if hydration:
            context_parts.append(
                f"Today's Hydration:\n"
                f"- Glasses Drunk: {hydration.get('glasses')} / {hydration.get('goal_glasses')} glasses (Total: {hydration.get('total_ml')}ml)"
            )
            
        # Battery
        battery = live_data.get("battery")
        if battery:
            charging_str = "charging" if battery.get("is_charging") else "not charging"
            context_parts.append(
                f"Rover Status:\n"
                f"- Battery Level: {battery.get('battery_percent')}% ({charging_str})\n"
                f"- Estimated Runtime Remaining: {battery.get('estimated_remaining_minutes')} mins"
            )
            
        # Emotion Context (Real-time Feedback from Frontend Camera)
        if emotion and emotion.lower() != "unknown":
            context_parts.append(
                f"Patient's Current Facial Emotion: {emotion.capitalize()} (Confidence: {confidence:.0%})\n"
                f"*Note: You must tailor your empathy and response tone based on this emotion if the patient seems distressed.*"
            )
            
        live_context = "=== LIVE ROVER & PATIENT SYSTEM CONTEXT ===\n" + "\n\n".join(context_parts) + "\n==========================================="
        
        print(f"[LLM] Context built from {len(routed_sources)} sources: {routed_sources}")
        print(f"   Medications: {len(meds)}, Vitals: {'yes' if vitals else 'no'}, Appointments: {len(appointments)}, Conditions: {len(conditions)}, Allergies: {len(allergies)}")

        messages = self._messages_for_chat(user_message, live_context)

        import time
        start_time = time.time()
        reply: str
        if route == "quality":
            self.last_route = "huggingface"
            try:
                print("[RUN] [LLM Backend] Calling Cloud Route (Hugging Face Inference API)...")
                reply = self._chat_hf(messages)
            except Exception as e:
                if OLLAMA_MODEL:
                    try:
                        print(f"[WARN] [LLM Backend] Hugging Face failed ({e}); falling back to local Ollama.")
                        self.last_route = "ollama_fallback"
                        reply = self._chat_ollama(messages)
                    except Exception as e2:
                        reply = f"I'm sorry, I encountered an error: {e2}"
                else:
                    reply = f"I'm sorry, I encountered an error: {e}"
        else:
            if OLLAMA_MODEL:
                self.last_route = "ollama"
                try:
                    print(f"[RUN] [LLM Backend] Calling Local Route (Ollama: {OLLAMA_MODEL})...")
                    reply = self._chat_ollama(messages)
                except Exception as e:
                    if API_KEY:
                        print(f"[WARN] [LLM Backend] Ollama failed ({e}); falling back to Hugging Face cloud.")
                        self.last_route = "huggingface_fallback"
                        try:
                            reply = self._chat_hf(messages)
                        except Exception as e2:
                            reply = f"I'm sorry, I encountered an error: {e2}"
                    else:
                        reply = f"I'm sorry, I encountered an error: {e}"
            else:
                self.last_route = "huggingface"
                try:
                    print("[RUN] [LLM Backend] Calling Cloud Route (Hugging Face Inference API)...")
                    reply = self._chat_hf(messages)
                except Exception as e:
                    reply = f"I'm sorry, I encountered an error: {e}"

        duration = time.time() - start_time
        print(f"[TIME] [LLM Backend] LLM generation completed in {duration:.2f} seconds.")

        # --- High-Reliability Tool Parsing Engine ---
        import re
        
        # 1. Heuristic fallback detection
        detected_actions = []
        msg_lower = user_message.lower()
        
        # Heuristics - Navigation
        rooms = {
            'kitchen': 'kitchen',
            'bathroom': 'bathroom',
            'bath': 'bathroom',
            'living room': 'living room',
            'living': 'living room',
            'bedroom': 'bedroom',
            'bed': 'bedroom',
            'dining room': 'dining room',
            'dining': 'dining room',
            'front door': 'front door',
            'door': 'front door',
            'entrance': 'front door'
        }
        for keyword, room in rooms.items():
            if keyword in msg_lower and any(w in msg_lower for w in ['go to', 'take me to', 'navigate', 'walk to', 'move to', 'take me']):
                detected_actions.append({'name': 'navigate', 'parameters': {'destination': room}})
                break
                
        # Heuristics - Music
        if any(w in msg_lower for w in ['stop music', 'pause music', 'turn off music', 'stop playing', 'pause playing', 'shut off music']) or ('music' in msg_lower and any(w in msg_lower for w in ['stop', 'pause', 'turn off', 'shut off', 'end', 'quiet', 'silence'])):
            detected_actions.append({'name': 'pause_music', 'parameters': {}})
        elif any(w in msg_lower for w in ['play music', 'play a song', 'music', 'play piano', 'relaxing sounds', 'upbeat music', 'relaxing piano']):
            mood = 'relaxing'
            if 'upbeat' in msg_lower or 'happy' in msg_lower:
                mood = 'upbeat'
            elif 'focus' in msg_lower or 'study' in msg_lower or 'classical' in msg_lower:
                mood = 'focus'
            detected_actions.append({'name': 'play_music', 'parameters': {'mood': mood}})
            
        # Heuristics - Call Guardian
        if any(w in msg_lower for w in ['call guardian', 'call my guardian', 'phone my guardian', 'contact my guardian', 'call mom', 'call dad']):
            detected_actions.append({'name': 'call_guardian', 'parameters': {}})
            
        # Heuristics - Medications
        if any(w in msg_lower for w in ['medication', 'medications', 'pills', 'pill', 'medicine', 'drug', 'take my medicine', 'schedule']):
            detected_actions.append({'name': 'show_medications', 'parameters': {}})
            
        # Heuristics - Emergency
        if any(w in msg_lower for w in ['fell down', 'hurt', 'pain', 'emergency', 'help me', 'danger', 'siren', 'accident', 'fall']):
            detected_actions.append({'name': 'trigger_emergency', 'parameters': {}})

        # 2. Parse JSON response
        parsed_response = {
            "response": reply,
            "actions": detected_actions
        }
        
        try:
            json_match = re.search(r'\{.*\}', reply, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if "response" in data:
                    parsed_response["response"] = data["response"].strip()
                if "actions" in data and isinstance(data["actions"], list):
                    if data["actions"]:
                        # If LLM generated specific actions, prioritize them!
                        parsed_response["actions"] = data["actions"]
                        print(f"[PARSED] [Tool Calling] Successfully parsed valid LLM JSON action(s): {data['actions']}")
                    else:
                        if detected_actions:
                            print(f"[HEURISTIC] [Engine] LLM actions empty; using heuristic-detected actions: {detected_actions}")
                        else:
                            print("[INFO] [Tool Calling] No action requested or detected.")
            else:
                if detected_actions:
                    print(f"[HEURISTIC] [Engine] Non-JSON LLM response. Fallback heuristics detected: {detected_actions}")
                else:
                    print("[INFO] [Tool Calling] Non-JSON LLM response. No action requested or detected.")
        except Exception as json_err:
            print(f"[WARN] [Tool Calling] JSON parsing failed ({json_err}). Using raw text + heuristics.")
            clean_reply = re.sub(r'\{.*\}', '', reply, flags=re.DOTALL).strip()
            if clean_reply:
                parsed_response["response"] = clean_reply

        # 3. Dynamic Override for Refusals / Contradictions
        refusal_keywords = [
            "can't", "cannot", "don't have the ability", "do not have the ability",
            "not able to", "unable to", "limited", "cannot play", "can't play",
            "cannot call", "can't call", "i'm an ai", "i am an ai", "just an ai",
            "companion robot", "don't have the capability", "do not have the capability",
            "don't support", "do not support"
        ]
        
        has_refusal = any(kw in parsed_response["response"].lower() for kw in refusal_keywords)
        
        if parsed_response["actions"] and (has_refusal or not parsed_response["response"].strip()):
            action = parsed_response["actions"][0]
            name = action.get("name")
            params = action.get("parameters", {})
            
            print(f"[OVERRIDE] [LLM Backend] Contradiction Overrider: Intercepted refusal. Dynamic override applied for action '{name}'.")
            
            if name == "play_music":
                mood = params.get("mood", "relaxing")
                parsed_response["response"] = f"I would be happy to play some {mood} music for you! Opening the player now."
            elif name == "call_guardian":
                parsed_response["response"] = "I am calling your guardian right away. Please stay calm, they are being notified."
            elif name == "navigate":
                dest = params.get("destination", "specified location")
                parsed_response["response"] = f"Sure! Navigating to the {dest} now. Please follow me."
            elif name == "show_medications":
                parsed_response["response"] = "Here is your medication schedule. Let's make sure you take them on time."
            elif name == "trigger_emergency":
                parsed_response["response"] = "I am triggering the emergency protocol and alerting emergency services immediately!"
            elif name == "check_health":
                parsed_response["response"] = "Starting your health check now. Let's verify your vital signs."
            elif name == "pause_music":
                parsed_response["response"] = "I have paused the music for you."

        print(f"[BOT] [Assistant Response]: '{parsed_response['response']}'")
        print(f"[ACTIONS] [Resolved Actions]: {parsed_response['actions']}")
        print(f"[CHAT] [LLM Backend] --- SESSION END ---\n")

        # Add pure text to history to maintain clean conversational context
        self.history.append({"user": user_message, "assistant": parsed_response["response"]})
        return parsed_response

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
