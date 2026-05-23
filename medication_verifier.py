"""
medication_verifier.py
======================
NovaCare - AI-Powered Medication Dataset Verifier & Corrector

WHAT THIS DOES
--------------
Reads master_pharmaceutical_dataset_v4.csv (or any version), sends each drug
record to Claude claude-sonnet-4-20250514 in batches, and for each record:

  1. VERIFIES   — is the generic_name a real drug? are the other fields correct?
  2. CORRECTS   — fills null/Unknown/wrong fields with accurate data
  3. FLAGS      — marks records it cannot verify as 'unverifiable'
  4. REPORTS    — outputs per-field accuracy %, correction counts, a full
                  accuracy_report.json, and uploads the clean CSV to PostgreSQL

STRATEGY
--------
  - Records are sent in batches of 20 to Claude with a strict JSON schema
  - Claude knows the full medication_catalog schema and what each field means
  - For each batch, Claude returns a JSON array of corrected records
  - Fields that were already correct are left unchanged
  - Fields that were null/wrong/Unknown get filled with real pharmaceutical data
  - Records that are not real drugs (e.g., leftover IUPAC junk) get flagged for removal

COST ESTIMATE (rough)
---------------------
  ~52,000 records / 20 per batch = ~2,600 API calls
  Each call ~500 tokens in + ~800 tokens out = ~1,300 tokens
  Total: ~3.4M tokens at claude-sonnet-4-20250514 pricing (~$5–$10 for the full run)
  Use MAX_RECORDS config below to do a test run on a subset first.

INSTALL
-------
  pip install pandas numpy requests sqlalchemy psycopg2-binary python-dotenv tqdm

RUN
---
  python scripts/medication_verifier.py

  # Or to test on first 500 records only:
  MAX_RECORDS=500 python scripts/medication_verifier.py
"""

import os, sys, json, time, logging, re, math
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    print("Run:  pip install requests")
    sys.exit(1)

try:
    from sqlalchemy import create_engine, text
except ImportError:
    print("Run:  pip install sqlalchemy psycopg2-binary")
    sys.exit(1)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("verifier.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\Pc\NovaCare-1\infrastructure\database\medication datasets"
INPUT_CSV   = os.path.join(DATA_DIR, "master_pharmaceutical_dataset_v4.csv")
OUTPUT_CSV  = os.path.join(DATA_DIR, "master_pharmaceutical_verified.csv")
REPORT_PATH = os.path.join(DATA_DIR, "accuracy_report.json")

# DB URL — reads from .env DATABASE_URL, swaps asyncpg for psycopg2
_raw_url = os.getenv("DATABASE_URL", "postgresql://postgres:admin123@localhost:5432/novacare_db")
DB_URL   = re.sub(r"postgresql(\+\w+)?://", "postgresql+psycopg2://", _raw_url)

# Anthropic API — reads from .env ANTHROPIC_API_KEY
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"

# Processing config
BATCH_SIZE   = int(os.getenv("BATCH_SIZE",   "20"))    # records per Claude call
MAX_RECORDS  = int(os.getenv("MAX_RECORDS",  "0"))     # 0 = all records
RETRY_LIMIT  = 3
RETRY_DELAY  = 5   # seconds between retries
RATE_SLEEP   = 0.5 # seconds between batches (avoid API rate limit)

# DB columns that match medication_catalog (no med_id — that's BIGSERIAL auto)
DB_COLS = [
    "brand_name", "generic_name", "manufacturer", "therapeutic_class",
    "is_controlled_substance", "product_external_id", "product_type",
    "route", "marketing_category", "substance_name",
    "active_strength", "strength_unit", "pharm_classes",
    "dea_schedule", "dosage_form",
]

# Fields that will be verified/corrected by Claude
VERIFIABLE_FIELDS = [
    "brand_name", "generic_name", "manufacturer", "therapeutic_class",
    "product_type", "route", "marketing_category", "substance_name",
    "pharm_classes", "dea_schedule", "dosage_form",
    "is_controlled_substance",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_null_or_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    s = str(val).strip()
    return s in ("", "None", "nan", "NaN", "Unknown", "unknown", "N/A", "n/a")


def normalize_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    return s in ("true", "1", "yes")


def clean_record_for_prompt(row: dict) -> dict:
    """Prepare a single row for inclusion in the Claude prompt."""
    out = {}
    for col in VERIFIABLE_FIELDS + ["active_strength", "strength_unit",
                                      "product_external_id", "_source"]:
        val = row.get(col)
        out[col] = None if is_null_or_empty(val) else val
    # active_strength: convert to float or None
    try:
        out["active_strength"] = float(row.get("active_strength")) if not is_null_or_empty(row.get("active_strength")) else None
    except (ValueError, TypeError):
        out["active_strength"] = None
    return out


# ── Claude API call ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a pharmaceutical data expert and drug database curator.
Your job is to verify and correct medication records for a clinical database called NovaCare.

For each batch of drug records you receive, you must:

1. CHECK if generic_name is a real pharmaceutical drug (not a food, supplement, IUPAC formula, or gibberish).
   - If it is NOT a real drug, set "_keep": false and leave other fields as-is.
   - If it IS a real drug, set "_keep": true and proceed.

2. VERIFY each field against your pharmaceutical knowledge.
   - If a field is null, empty, "Unknown", or clearly wrong — FILL IT IN with the correct value.
   - If a field already has a correct value — leave it exactly as-is.
   - Do NOT fabricate data you are not confident about — leave those fields null.

3. CORRECT the following fields using accurate pharmaceutical knowledge:
   - generic_name: the INN (International Nonproprietary Name) — correct spelling, Title Case
   - brand_name: the most well-known brand name for this drug (e.g. Tylenol for acetaminophen)
   - manufacturer: primary manufacturer or "Various" for generics with many makers
   - therapeutic_class: ATC level-2 or clinical description (e.g. "Beta-Adrenergic Blocker", "Antidiabetic")
   - product_type: one of: "Human Prescription Drug", "OTC", "Biologic", "Vaccine", "Veterinary", "Diagnostic"
   - route: one of: Oral, Intravenous, Topical, Inhalation, Subcutaneous, Intramuscular, Transdermal, Ophthalmic, Nasal, Rectal, Sublingual, Otic, Vaginal, Intrathecal
   - dosage_form: one of: Tablet, Capsule, Solution, Injection, Cream, Ointment, Gel, Powder, Suspension, Syrup, Spray, Patch, Lotion, Drops, Suppository, Inhaler, Film, Granules, Implant
   - marketing_category: one of: NDA, ANDA, BLA, OTC Monograph, WHO Prequalified, Approved
   - substance_name: the active ingredient name (usually same as generic_name for single-ingredient drugs)
   - pharm_classes: pharmacological class string, e.g. "Cyclooxygenase Inhibitor [EPC]; Anti-Inflammatory Agents [CS]"
   - dea_schedule: null if not controlled, else one of: CI, CII, CIII, CIV, CV
   - is_controlled_substance: true/false boolean
   - active_strength: numeric value only (e.g. 500 for 500mg) — use most common dose if multiple exist, or null
   - strength_unit: the unit (mg, mcg, g, ml, %, IU, mEq) — or null

4. Add a "_confidence" field (0.0 to 1.0) indicating how confident you are in the corrected record overall.
   - 1.0 = well-known drug, all fields verified with certainty
   - 0.7 = mostly known drug, minor fields inferred
   - 0.5 = less common drug, some fields are best-guess
   - 0.3 = obscure or uncertain drug
   - 0.0 = could not verify at all

5. Add a "_corrections_made" array listing the field names you changed or filled in.

CRITICAL RULES:
- Never invent data you are not genuinely confident about. Leave those fields null.
- Do not change fields that are already correct.
- Output ONLY valid JSON. No markdown, no explanation, no backticks.
- Output a JSON array, one object per input record, in the same order.
- Each output object must contain ALL the fields from the input plus _keep, _confidence, _corrections_made.
"""

def call_claude(batch: list[dict]) -> Optional[list[dict]]:
    """Send a batch of records to Claude for verification + correction."""
    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Verify and correct these {len(batch)} drug records. "
                    f"Return a JSON array of {len(batch)} corrected objects.\n\n"
                    + json.dumps(batch, ensure_ascii=False, indent=None)
                ),
            }
        ],
    }

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data    = resp.json()
            content = data.get("content", [])
            text    = "".join(block.get("text", "") for block in content if block.get("type") == "text")

            # Strip markdown fences if Claude wrapped in them
            text = re.sub(r"^```json\s*", "", text.strip())
            text = re.sub(r"```\s*$",     "", text.strip())

            result = json.loads(text)
            if isinstance(result, list):
                return result
            log.warning(f"  Claude returned non-list JSON: {str(result)[:200]}")
            return None

        except json.JSONDecodeError as e:
            log.warning(f"  Attempt {attempt}: JSON parse error: {e}. Raw: {text[:300]}")
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait = 30 * attempt
                log.warning(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif resp.status_code == 529:
                wait = 60
                log.warning(f"  API overloaded. Waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"  HTTP {resp.status_code}: {e}")
                if attempt >= RETRY_LIMIT:
                    return None
        except Exception as e:
            log.warning(f"  Attempt {attempt}: {e}")
            if attempt >= RETRY_LIMIT:
                return None
        time.sleep(RETRY_DELAY * attempt)

    return None


# ── Merge corrected record back into the DataFrame row ────────────────────────

def apply_correction(original: dict, corrected: dict) -> dict:
    """
    Merge a corrected record from Claude back into the original row.
    Tracks which fields changed for reporting.
    """
    result   = dict(original)
    changed  = corrected.get("_corrections_made", [])

    for field in VERIFIABLE_FIELDS + ["active_strength", "strength_unit"]:
        new_val = corrected.get(field)
        old_val = original.get(field)

        # Only update if:
        # (a) the new value is non-null AND
        # (b) the old value was null/empty OR Claude explicitly listed it in corrections_made
        if new_val is not None and not is_null_or_empty(str(new_val)):
            if is_null_or_empty(str(old_val) if old_val is not None else "") or field in changed:
                result[field] = new_val

    result["_keep"]             = corrected.get("_keep", True)
    result["_confidence"]       = corrected.get("_confidence", 0.5)
    result["_corrections_made"] = json.dumps(corrected.get("_corrections_made", []))
    return result


# ── Accuracy reporter ─────────────────────────────────────────────────────────

def build_accuracy_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Compare the dataset before and after verification to produce a detailed
    accuracy report showing:
      - Per-field null rate before and after
      - Per-field correction count
      - Records removed as non-drugs
      - Average confidence score
      - Overall data completeness score
    """
    report = {
        "total_records_before":   len(df_before),
        "total_records_after":    len(df_after),
        "records_removed":        len(df_before) - len(df_after),
        "average_confidence":     round(df_after["_confidence"].mean(), 3) if "_confidence" in df_after else None,
        "per_field": {},
    }

    for col in VERIFIABLE_FIELDS + ["active_strength", "strength_unit"]:
        null_before = df_before[col].apply(lambda v: is_null_or_empty(str(v) if v is not None else "")).sum() if col in df_before.columns else len(df_before)
        null_after  = df_after[col].apply( lambda v: is_null_or_empty(str(v) if v is not None else "")).sum() if col in df_after.columns else len(df_after)
        filled_in   = max(0, null_before - null_after)
        total       = len(df_after)

        pct_complete_before = round((1 - null_before / len(df_before)) * 100, 1) if len(df_before) > 0 else 0
        pct_complete_after  = round((1 - null_after  / total)          * 100, 1) if total > 0 else 0

        report["per_field"][col] = {
            "null_count_before":     int(null_before),
            "null_count_after":      int(null_after),
            "fields_filled_in":      int(filled_in),
            "pct_complete_before":   pct_complete_before,
            "pct_complete_after":    pct_complete_after,
            "improvement":           round(pct_complete_after - pct_complete_before, 1),
        }

    # Count total corrections made
    if "_corrections_made" in df_after.columns:
        all_corrections = df_after["_corrections_made"].dropna().apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )
        correction_counts = {}
        for corr_list in all_corrections:
            for field in corr_list:
                correction_counts[field] = correction_counts.get(field, 0) + 1
        report["correction_counts_by_field"] = correction_counts
        report["total_corrections"] = sum(correction_counts.values())

    # Overall completeness score
    field_scores = [v["pct_complete_after"] for v in report["per_field"].values()]
    report["overall_completeness_pct"] = round(sum(field_scores) / len(field_scores), 1)

    return report


def print_accuracy_report(report: dict):
    log.info("=" * 68)
    log.info("  ACCURACY & CORRECTION REPORT")
    log.info("=" * 68)
    log.info(f"  Records before: {report['total_records_before']:,}")
    log.info(f"  Records after:  {report['total_records_after']:,}")
    log.info(f"  Removed (non-drugs): {report['records_removed']:,}")
    if report.get("average_confidence"):
        log.info(f"  Avg confidence: {report['average_confidence']:.3f} / 1.000")
    log.info(f"  Overall completeness: {report['overall_completeness_pct']}%")
    log.info(f"  Total corrections made: {report.get('total_corrections', 0):,}")
    log.info("")
    log.info(f"  {'Field':<30} {'Before':>8} {'After':>8} {'Filled':>8} {'Improvement':>12}")
    log.info(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
    for field, stats in sorted(report["per_field"].items(),
                                key=lambda x: -x[1]["improvement"]):
        log.info(
            f"  {field:<30} {stats['pct_complete_before']:>7.1f}% "
            f"{stats['pct_complete_after']:>7.1f}% "
            f"{stats['fields_filled_in']:>8,} "
            f"{stats['improvement']:>+11.1f}%"
        )
    log.info("=" * 68)


# ── DB connection ─────────────────────────────────────────────────────────────

def get_engine():
    for url in [DB_URL, DB_URL.replace("psycopg2", "pg8000")]:
        driver = "psycopg2" if "psycopg2" in url else "pg8000"
        try:
            eng = create_engine(url, pool_pre_ping=True)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info(f"  DB connected ({driver}) -> {url.split('@')[-1]}")
            return eng
        except Exception as e:
            if "No module" in str(e):
                pkg = "psycopg2-binary" if driver == "psycopg2" else "pg8000"
                log.warning(f"  [{driver}] not installed. Run:  pip install {pkg}")
            else:
                log.warning(f"  [{driver}] failed: {e}")
    return None


# ── Checkpoint save/load (so you can resume if interrupted) ───────────────────

def checkpoint_path(batch_idx: int) -> Path:
    return Path(DATA_DIR) / "verifier_checkpoints" / f"batch_{batch_idx:06d}.json"


def save_checkpoint(batch_idx: int, corrected_rows: list):
    p = checkpoint_path(batch_idx)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(corrected_rows, f, ensure_ascii=False)


def load_checkpoint(batch_idx: int) -> Optional[list]:
    p = checkpoint_path(batch_idx)
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def already_done_batches() -> set:
    p = Path(DATA_DIR) / "verifier_checkpoints"
    if not p.exists():
        return set()
    return {int(f.stem.split("_")[1]) for f in p.glob("batch_*.json")}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 68)
    log.info("  NovaCare Medication Verifier + Corrector")
    log.info("=" * 68)

    if not ANTHROPIC_API_KEY:
        log.error("ANTHROPIC_API_KEY is not set!")
        log.error("Add this line to your .env file:")
        log.error("  ANTHROPIC_API_KEY=your_key_here")
        log.error("Get your key at: https://console.anthropic.com/")
        sys.exit(1)

    # Load input CSV
    if not os.path.exists(INPUT_CSV):
        # Try v3 or v2 as fallback
        for alt in ["v3", "v2"]:
            alt_path = INPUT_CSV.replace("v4", alt)
            if os.path.exists(alt_path):
                log.warning(f"v4 CSV not found. Using {alt_path}")
                INPUT_CSV_ACTUAL = alt_path
                break
        else:
            log.error(f"Input CSV not found: {INPUT_CSV}")
            log.error("Run medication_pipeline_v4.py first.")
            sys.exit(1)
    else:
        INPUT_CSV_ACTUAL = INPUT_CSV

    log.info(f"[*] Loading {INPUT_CSV_ACTUAL}...")
    df = pd.read_csv(INPUT_CSV_ACTUAL, dtype=str)

    # Ensure all DB cols exist
    for col in DB_COLS + ["_source"]:
        if col not in df.columns:
            df[col] = None

    if MAX_RECORDS > 0:
        log.info(f"[*] MAX_RECORDS={MAX_RECORDS} — processing subset only.")
        df = df.head(MAX_RECORDS)

    log.info(f"[*] {len(df):,} records to verify.")
    df_before = df.copy()

    # Prepare output columns
    df["_keep"]             = True
    df["_confidence"]       = 0.5
    df["_corrections_made"] = "[]"

    # Figure out which batches are already done (checkpoint resume)
    done_batches = already_done_batches()
    if done_batches:
        log.info(f"[*] Resuming — {len(done_batches)} batches already done.")

    records   = df.to_dict("records")
    n_batches = math.ceil(len(records) / BATCH_SIZE)

    corrected_records = list(records)  # will be updated in-place

    log.info(f"[*] Processing {n_batches:,} batches of {BATCH_SIZE}...")

    for batch_idx in tqdm(range(n_batches), desc="Verifying batches"):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(records))
        batch = records[start:end]

        # Check checkpoint
        cached = load_checkpoint(batch_idx)
        if cached is not None and batch_idx in done_batches:
            for i, corrected_row in enumerate(cached):
                corrected_records[start + i] = apply_correction(batch[i], corrected_row)
            continue

        # Prepare records for Claude (only send verifiable fields to save tokens)
        batch_for_claude = [clean_record_for_prompt(r) for r in batch]

        corrected = call_claude(batch_for_claude)
        if corrected is None or len(corrected) != len(batch):
            log.warning(f"  Batch {batch_idx}: Claude returned unexpected result, keeping originals.")
            # Mark as low confidence but don't change data
            for i in range(start, end):
                corrected_records[i]["_confidence"] = 0.0
                corrected_records[i]["_corrections_made"] = "[]"
            continue

        # Apply corrections
        for i, corrected_row in enumerate(corrected):
            corrected_records[start + i] = apply_correction(batch[i], corrected_row)

        # Save checkpoint
        save_checkpoint(batch_idx, corrected)
        time.sleep(RATE_SLEEP)

    # Build final DataFrame
    log.info("[*] Building final DataFrame...")
    df_out = pd.DataFrame(corrected_records)

    # Remove records Claude flagged as non-drugs
    n_before_filter = len(df_out)
    df_out = df_out[df_out["_keep"].apply(normalize_bool)]
    log.info(f"  Removed {n_before_filter - len(df_out):,} non-drug records.")

    # Coerce types
    df_out["active_strength"]         = pd.to_numeric(df_out.get("active_strength"), errors="coerce")
    df_out["is_controlled_substance"] = df_out.get("is_controlled_substance", False).apply(normalize_bool)
    df_out["_confidence"]             = pd.to_numeric(df_out.get("_confidence"), errors="coerce").fillna(0.5)

    # Ensure NOT NULL columns
    df_out["brand_name"]   = df_out["brand_name"].fillna(df_out["generic_name"])
    df_out["generic_name"] = df_out["generic_name"].fillna("Unknown")
    df_out = df_out[df_out["generic_name"] != "Unknown"]

    # Truncate to column max lengths
    str_limits = {
        "brand_name": 255, "generic_name": 255, "manufacturer": 255,
        "therapeutic_class": 255, "product_external_id": 50,
        "product_type": 100, "route": 100, "marketing_category": 100,
        "strength_unit": 50, "dea_schedule": 10, "dosage_form": 100,
    }
    for col, limit in str_limits.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].apply(
                lambda v: str(v)[:limit] if v is not None and not pd.isna(v) else None
            )

    # ── Accuracy Report ──
    report = build_accuracy_report(df_before, df_out)
    print_accuracy_report(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"[*] Accuracy report saved -> {REPORT_PATH}")

    # ── Save verified CSV ──
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"[*] Verified CSV saved -> {OUTPUT_CSV}")

    # ── Upload to PostgreSQL ──
    log.info("[*] Uploading to PostgreSQL...")
    engine = get_engine()
    if not engine:
        log.warning("[!] No DB connection. CSV saved, import manually:")
        log.warning(f"    psql -d novacare_db -c \"\\copy medication_catalog({','.join(DB_COLS)}) FROM '{OUTPUT_CSV}' CSV HEADER\"")
        return

    df_db = df_out[DB_COLS].copy()
    df_db["active_strength"]         = pd.to_numeric(df_db["active_strength"], errors="coerce")
    df_db["is_controlled_substance"] = df_db["is_controlled_substance"].fillna(False).astype(bool)

    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE medication_catalog RESTART IDENTITY CASCADE"))
        df_db.to_sql(
            "medication_catalog", con=engine,
            if_exists="append", index=False,
            chunksize=2000, method="multi",
        )
        log.info(f"[*] Uploaded {len(df_db):,} verified records to PostgreSQL!")
    except Exception as e:
        log.error(f"[!] DB insert failed: {e}")

    log.info("=" * 68)
    log.info("  Verification complete!")
    log.info("=" * 68)


if __name__ == "__main__":
    main()
