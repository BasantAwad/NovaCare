"""Test which API backends are available for the mental health pipeline."""
import urllib.request, json, os, io, sys, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'), override=True)


def test_openai_compat(name, base_url, api_key, model):
    if not api_key:
        print(f"  [SKIP] {name}: No API key set")
        return False
    url = f"{base_url}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say OK in one word."}],
        "max_tokens": 10, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
    }, method="POST")
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=20)
        dt = time.time() - t0
        data = json.loads(resp.read().decode())
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"  [OK] {name} ({model}): {text!r} in {dt:.2f}s")
        return True
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:150]
        print(f"  [FAIL] {name} ({model}): HTTP {e.code} -> {msg}")
        return False
    except Exception as e:
        print(f"  [FAIL] {name} ({model}): {e}")
        return False


def test_gemini(api_key):
    if not api_key:
        print(f"  [SKIP] Gemini: No API key set")
        return False
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
                       "generationConfig": {"maxOutputTokens": 5}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=15)
        dt = time.time() - t0
        data = json.loads(resp.read().decode())
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        print(f"  [OK] Gemini (gemini-2.0-flash): {text!r} in {dt:.2f}s")
        return True
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:150]
        print(f"  [FAIL] Gemini: HTTP {e.code} -> {msg}")
        return False


print("=" * 50)
print("  API Backend Availability Test")
print("=" * 50)

results = {}

print("\n--- OpenAI-Compatible Backends ---")
results["Groq"] = test_openai_compat(
    "Groq", "https://api.groq.com/openai/v1",
    os.getenv("GROQ_API_KEY", ""), "llama-3.3-70b-versatile")

results["Cerebras"] = test_openai_compat(
    "Cerebras", "https://api.cerebras.ai/v1",
    os.getenv("CEREBRAS_API_KEY", ""), "llama-3.3-70b")

results["SambaNova"] = test_openai_compat(
    "SambaNova", "https://api.sambanova.ai/v1",
    os.getenv("SAMBANOVA_API_KEY", ""), "Meta-Llama-3.3-70B-Instruct")

print("\n--- Gemini Backend ---")
results["Gemini"] = test_gemini(os.getenv("GEMINI_API_KEY", ""))

print("\n--- Summary ---")
working = [k for k, v in results.items() if v]
failed = [k for k, v in results.items() if not v]

if working:
    print(f"  Working: {', '.join(working)}")
else:
    print("  No backends currently available!")
    print("  Add at least one API key to .env (see .env.example)")

if failed:
    print(f"  Unavailable: {', '.join(failed)}")

# Quick pipeline test if anything works
if working:
    print("\n--- Quick Pipeline Test ---")
    from mental_health_pipeline import get_pipeline, stage_pattern_recognition
    p = get_pipeline()
    print(f"  Pipeline available: {p.is_available}")
    
    time.sleep(2)
    pat = stage_pattern_recognition("I feel so empty and hopeless, nothing matters anymore")
    detected = pat.get("detected", False)
    pattern = pat.get("pattern", "none")
    conf = pat.get("confidence", 0)
    print(f"  Pattern test: detected={detected}, pattern={pattern}, confidence={conf}")
    
    if detected:
        print(f"\n  Pipeline is WORKING via: {', '.join(working)}")
    else:
        print(f"\n  Pattern detection returned no result (API may be warming up)")
