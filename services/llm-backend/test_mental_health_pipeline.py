"""
Integration Test Suite -- Mental Health Therapy Pipeline
========================================================
Tests each stage of the pipeline independently, then runs end-to-end scenarios.
Works with or without API keys (reports what's available vs. missing).

Usage:
    python test_mental_health_pipeline.py
"""

import json
import os
import sys
import time

# Ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(current_dir, ".env"), override=True)

# -- Colour helpers --
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[OK] {msg}{RESET}")
def fail(msg): print(f"  {RED}[FAIL] {msg}{RESET}")
def warn(msg): print(f"  {YELLOW}[WARN] {msg}{RESET}")
def info(msg): print(f"  {CYAN}[INFO] {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'='*60}\n  {msg}\n{'='*60}{RESET}")
def subheader(msg): print(f"\n{BOLD}  -- {msg} --{RESET}")

passed = 0
failed = 0
skipped = 0

# Delay between API calls to avoid Gemini free-tier 429
API_DELAY = 5  # seconds


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        ok(f"{name}")
        if detail:
            info(f"  -> {detail}")
        passed += 1
    else:
        fail(f"{name}")
        if detail:
            info(f"  -> {detail}")
        failed += 1


def api_pause():
    """Small pause between API calls to respect free-tier rate limits."""
    time.sleep(API_DELAY)


# ===========================================================================
# 0. Environment & Import Checks
# ===========================================================================
header("0. Environment & Import Checks")

groq_key = os.getenv("GROQ_API_KEY", "")
gemini_key = os.getenv("GEMINI_API_KEY", "")
hf_key = os.getenv("HUGGINGFACE_API_KEY", "")

info(f"GROQ_API_KEY:       {'SET (' + groq_key[:8] + '...)' if groq_key else 'NOT SET'}")
info(f"GEMINI_API_KEY:     {'SET (' + gemini_key[:8] + '...)' if gemini_key else 'NOT SET'}")
info(f"HUGGINGFACE_API_KEY: {'SET (' + hf_key[:8] + '...)' if hf_key else 'NOT SET'}")

has_groq = bool(groq_key)
has_gemini = bool(gemini_key)
has_any_key = has_groq or has_gemini

if not has_any_key:
    warn("No GROQ_API_KEY or GEMINI_API_KEY found!")
    warn("Pipeline will operate in fallback mode only.")
    warn("To fully test, add keys to services/llm-backend/.env")
    print()

# Test imports
try:
    from mental_health_pipeline import (
        get_pipeline, MentalHealthPipeline, PipelineResult, RiskLevel,
        stage_pattern_recognition, stage_risk_assessment,
        stage_therapeutic_response, stage_validate_response,
        _groq_chat, _gemini_chat,
        GROQ_API_KEY as LOADED_GROQ, GEMINI_API_KEY as LOADED_GEMINI,
    )
    test("Import mental_health_pipeline", True)
except Exception as e:
    test("Import mental_health_pipeline", False, str(e))
    print(f"\n{RED}Cannot continue without pipeline module.{RESET}")
    sys.exit(1)

try:
    from mental_health_integration import MentalHealthOrchestrator, get_orchestrator
    test("Import mental_health_integration", True)
except Exception as e:
    test("Import mental_health_integration", False, str(e))

# Pipeline singleton
pipeline = get_pipeline()
test("Pipeline singleton created", pipeline is not None)
test("Pipeline.is_available", pipeline.is_available == has_any_key,
     f"available={pipeline.is_available}")


# ===========================================================================
# 1. Test Low-Level API Helpers
# ===========================================================================
header("1. Low-Level API Connectivity")

subheader("Groq API")
if has_groq:
    t0 = time.time()
    result = _groq_chat(
        [{"role": "user", "content": "Say hello in one word."}],
        model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=10,
    )
    dt = time.time() - t0
    if result is not None:
        test("Groq chat-completions reachable", True, f"{dt:.2f}s -- reply: {result!r}")
    else:
        warn(f"Groq returned None (403/key issue) -- {dt:.2f}s. Gemini will be used as fallback.")
        skipped += 1
else:
    warn("SKIPPED -- no GROQ_API_KEY")
    skipped += 1

subheader("Groq Llama Guard")
if has_groq:
    t0 = time.time()
    result = _groq_chat(
        [{"role": "user", "content": "What is the weather today?"}],
        model="llama-guard-3-8b", temperature=0.0, max_tokens=20,
    )
    dt = time.time() - t0
    if result is not None:
        test("Llama Guard reachable", True, f"{dt:.2f}s -- reply: {result!r}")
    else:
        warn(f"Llama Guard returned None (403/key issue) -- {dt:.2f}s. Gemini will be used as fallback.")
        skipped += 1
else:
    warn("SKIPPED -- no GROQ_API_KEY")
    skipped += 1

subheader("Gemini API")
if has_gemini:
    t0 = time.time()
    result = _gemini_chat(
        [{"role": "user", "content": "Say hello in one word."}],
        temperature=0.0, max_tokens=10,
    )
    dt = time.time() - t0
    test("Gemini generateContent reachable", result is not None, f"{dt:.2f}s -- reply: {result!r}")
    api_pause()
else:
    warn("SKIPPED -- no GEMINI_API_KEY")
    skipped += 1


# ===========================================================================
# 2. Stage-by-Stage Tests
# ===========================================================================
header("2. Stage-by-Stage Pipeline Tests")

# -- Stage 1: Pattern Recognition --
subheader("Stage 1 -- Pattern Recognition")

test_messages = [
    ("I feel so empty and hopeless, nothing matters anymore", True, "depression"),
    ("I can't sleep at night, my mind keeps racing with worry", True, "anxiety"),
    ("What's the weather like today?", False, "none"),
    ("I'm having a great day!", False, "none"),
]

if has_any_key:
    for msg, expect_detected, expect_pattern in test_messages:
        t0 = time.time()
        result = stage_pattern_recognition(msg)
        dt = time.time() - t0
        detected = result.get("detected", False)
        pattern = result.get("pattern", "none")
        confidence = result.get("confidence", 0.0)

        if expect_detected:
            test(
                f"Detects signal in: '{msg[:45]}...'",
                detected == True,
                f"pattern={pattern}, conf={confidence:.2f}, time={dt:.2f}s"
            )
        else:
            test(
                f"No false positive: '{msg[:45]}...'",
                detected == False or confidence < 0.4,
                f"detected={detected}, pattern={pattern}, conf={confidence:.2f}, time={dt:.2f}s"
            )
        api_pause()
else:
    warn("SKIPPED Stage 1 tests -- no API keys")
    skipped += 4

# -- Stage 2: Risk Assessment --
subheader("Stage 2 -- Risk Assessment")

risk_messages = [
    ("I want to hurt myself", RiskLevel.HIGH),
    ("I love spending time with my family", RiskLevel.NONE),
]

if has_any_key:
    for msg, min_expected in risk_messages:
        t0 = time.time()
        risk_level, raw = stage_risk_assessment(msg)
        dt = time.time() - t0

        if min_expected == RiskLevel.NONE:
            test(
                f"Safe message rated safe: '{msg[:40]}...'",
                risk_level == RiskLevel.NONE,
                f"risk={risk_level.value}, raw={raw[:80] if raw else '(empty)'!r}, time={dt:.2f}s"
            )
        else:
            test(
                f"Risky message flagged: '{msg[:40]}...'",
                risk_level in (RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRISIS),
                f"risk={risk_level.value}, raw={raw[:80] if raw else '(empty)'!r}, time={dt:.2f}s"
            )
        api_pause()
else:
    warn("SKIPPED Stage 2 tests -- no API keys")
    skipped += 2

# -- Stage 3: Therapeutic Response --
subheader("Stage 3 -- Therapeutic Response Generation")

if has_any_key:
    t0 = time.time()
    reply, route = stage_therapeutic_response(
        user_message="I've been feeling really anxious lately and can't focus on anything",
        pattern="anxiety",
        risk_level=RiskLevel.NONE,
        emotion="neutral",
    )
    dt = time.time() - t0
    test(
        f"Therapy response generated via {route}",
        bool(reply) and len(reply) > 20,
        f"time={dt:.2f}s, len={len(reply)} chars"
    )
    info(f"  Response preview: {reply[:150]}...")
    api_pause()

    # Test crisis response
    t0 = time.time()
    reply2, route2 = stage_therapeutic_response(
        user_message="I don't want to be alive anymore",
        pattern="self_harm",
        risk_level=RiskLevel.CRISIS,
        emotion="sad",
    )
    dt = time.time() - t0
    test(
        f"Crisis response generated via {route2}",
        bool(reply2) and len(reply2) > 20,
        f"time={dt:.2f}s"
    )
    info(f"  Crisis preview: {reply2[:150]}...")
    api_pause()
else:
    warn("SKIPPED Stage 3 tests -- no API keys")
    skipped += 2

# -- Stage 4: Response Validation --
subheader("Stage 4 -- Response Validation")

if has_any_key:
    safe_response = "I hear you, and your feelings are completely valid. It's okay to feel this way."
    t0 = time.time()
    is_safe, raw = stage_validate_response(safe_response)
    dt = time.time() - t0
    test("Safe therapeutic response passes validation", is_safe, f"time={dt:.2f}s, raw={raw[:80] if raw else '(empty)'!r}")
    api_pause()
else:
    warn("SKIPPED Stage 4 tests -- no API keys")
    skipped += 1


# ===========================================================================
# 3. End-to-End Pipeline Tests
# ===========================================================================
header("3. End-to-End Pipeline Integration")

if has_any_key:
    # Reset session
    pipeline.clear_session()

    e2e_scenarios = [
        {
            "name": "Depression signal via text",
            "message": "I feel so empty inside, like nothing matters anymore and I can't find joy in anything",
            "emotion": "neutral",
            "emotion_conf": 0.0,
            "expect_triggered": True,
        },
        {
            "name": "Anxiety with sad facial emotion",
            "message": "I keep worrying about everything and my chest feels tight",
            "emotion": "sad",
            "emotion_conf": 0.9,
            "expect_triggered": True,
        },
        {
            "name": "Normal conversation (no trigger)",
            "message": "Can you tell me about the weather forecast for tomorrow?",
            "emotion": "neutral",
            "emotion_conf": 0.0,
            "expect_triggered": False,
        },
        {
            "name": "Emotion-only trigger (happy text, sad face)",
            "message": "I'm fine, everything is great",
            "emotion": "sad",
            "emotion_conf": 0.85,
            "expect_triggered": True,
        },
    ]

    for scenario in e2e_scenarios:
        subheader(f"Scenario: {scenario['name']}")
        t0 = time.time()
        result = pipeline.process(
            user_message=scenario["message"],
            emotion=scenario["emotion"],
            emotion_confidence=scenario["emotion_conf"],
        )
        dt = time.time() - t0

        test(
            f"triggered={result.triggered} (expected={scenario['expect_triggered']})",
            result.triggered == scenario["expect_triggered"],
            f"pattern={result.pattern}, risk={result.risk_level}, route={result.route}, time={dt:.2f}s"
        )

        if result.stages_log:
            info(f"  Stages: {' | '.join(result.stages_log)}")
        if result.response:
            preview = result.response.replace('\n', ' ')[:120]
            info(f"  Response: {preview}...")
        api_pause()

    # Session summary
    subheader("Session Summary")
    summary = pipeline.get_session_summary()
    info(f"Total triggers: {summary['total_triggers']}")
    info(f"Pattern counts: {summary['pattern_counts']}")
    info(f"Most common: {summary['most_common']}")
    test("Session tracking works", summary["total_triggers"] > 0)
else:
    warn("SKIPPED E2E tests -- no API keys configured")
    skipped += 5


# ===========================================================================
# 4. Orchestrator Bridge Integration
# ===========================================================================
header("4. Orchestrator Bridge (mental_health_integration.py)")

try:
    orchestrator = get_orchestrator()
    test("Orchestrator singleton created", orchestrator is not None)
    test("Orchestrator has pipeline", orchestrator._pipeline is not None,
         f"pipeline.is_available={orchestrator._pipeline.is_available if orchestrator._pipeline else 'N/A'}")

    if has_any_key:
        api_pause()
        # Test process() returns correct tuple format
        bypass, reply, prefix = orchestrator.process("I feel very sad and lonely today")
        test(
            "process() returns correct tuple",
            isinstance(bypass, bool) and isinstance(reply, str) and isinstance(prefix, str),
            f"bypass={bypass}, reply_len={len(reply)}, prefix_len={len(prefix)}"
        )
        if bypass:
            test("Pipeline bypass triggered for sad message", True, "route via pipeline")
            info(f"  Reply preview: {reply[:120]}...")
        else:
            info(f"  No bypass -- prefix: {prefix!r}")

        api_pause()
        # Test normal message doesn't trigger
        bypass2, reply2, prefix2 = orchestrator.process("What time is it?")
        test("Normal message does NOT trigger bypass", not bypass2,
             f"bypass={bypass2}, prefix={prefix2!r}")
    else:
        warn("SKIPPED orchestrator E2E -- no API keys")
        skipped += 3
except Exception as e:
    fail(f"Orchestrator test failed: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# ===========================================================================
# 5. Flask API Endpoint Tests (simulated)
# ===========================================================================
header("5. Flask API Endpoint Tests")

try:
    sys.path.insert(0, current_dir)
    from api_server import app
    test("Flask app imports successfully", True)

    with app.test_client() as client:
        # Health check
        resp = client.get("/health")
        test("GET /health returns 200", resp.status_code == 200)

        # Mental health health endpoint
        resp = client.get("/api/mental-health/health")
        data = resp.get_json()
        test("GET /api/mental-health/health returns JSON", data is not None,
             f"status={data.get('status')}, groq={data.get('groq_configured')}, gemini={data.get('gemini_configured')}")

        # Mental health session endpoint
        resp = client.get("/api/mental-health/session")
        data = resp.get_json()
        test("GET /api/mental-health/session returns JSON", data is not None,
             f"total_triggers={data.get('total_triggers')}")

        if has_any_key:
            api_pause()
            # Mental health analyze endpoint
            resp = client.post("/api/mental-health/analyze",
                               json={"message": "I've been feeling very depressed lately"},
                               content_type="application/json")
            data = resp.get_json()
            test(
                "POST /api/mental-health/analyze processes message",
                resp.status_code == 200 and data.get("status") == "success",
                f"triggered={data.get('triggered')}, pattern={data.get('pattern')}, route={data.get('route')}"
            )

            api_pause()
            # Test chat endpoint integration
            resp = client.post("/api/chat",
                               json={"message": "I feel really anxious and scared"},
                               content_type="application/json")
            data = resp.get_json()
            test(
                "POST /api/chat with mental health message",
                resp.status_code == 200,
                f"status={data.get('status')}, route={data.get('llm_route')}, has_mh_meta={'mental_health' in data}"
            )
        else:
            resp = client.post("/api/mental-health/analyze",
                               json={"message": "test"},
                               content_type="application/json")
            test("POST /api/mental-health/analyze returns 503 without keys",
                 resp.status_code == 503)

except Exception as e:
    fail(f"Flask API tests failed: {e}")
    import traceback
    traceback.print_exc()
    failed += 1


# ===========================================================================
# Summary
# ===========================================================================
header("TEST SUMMARY")

total = passed + failed + skipped
print(f"""
  {GREEN}Passed:  {passed}{RESET}
  {RED}Failed:  {failed}{RESET}
  {YELLOW}Skipped: {skipped}{RESET}
  -----------
  Total:   {total}
""")

if not has_any_key:
    print(f"""{YELLOW}
  To run FULL integration tests, add API keys to .env:

    GROQ_API_KEY=gsk_your_key_here
    GEMINI_API_KEY=your_gemini_key_here

  Free keys:
    Groq   -> https://console.groq.com
    Gemini -> https://aistudio.google.com
{RESET}""")

if failed > 0:
    print(f"  {RED}{BOLD}{failed} test(s) failed.{RESET}")
    sys.exit(1)
else:
    print(f"  {GREEN}{BOLD}All tests passed!{RESET}")
    sys.exit(0)
