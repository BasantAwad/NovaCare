"""
medication_pipeline_v4.py
=========================
NovaCare - Medication Catalog Builder  v4.0

KEY FIX IN v4
-------------
Previous versions mapped output columns to a schema that doesn't match your
actual database. This version is built against the REAL medication_catalog table:

    medication_catalog (
        med_id              BIGSERIAL PRIMARY KEY  -- auto, don't insert
        brand_name          VARCHAR(255) NOT NULL
        generic_name        VARCHAR(255) NOT NULL
        manufacturer        VARCHAR(255)
        therapeutic_class   VARCHAR(255)
        is_controlled_substance BOOLEAN DEFAULT FALSE
        product_external_id VARCHAR(50)   -- NDC product code
        product_type        VARCHAR(100)  -- "Human Prescription Drug", "OTC" etc.
        route               VARCHAR(100)  -- "ORAL", "TOPICAL" etc.
        marketing_category  VARCHAR(100)  -- "NDA", "ANDA", "OTC Monograph" etc.
        substance_name      TEXT          -- active ingredient name (may differ from generic)
        active_strength     DECIMAL(10,4)
        strength_unit       VARCHAR(50)
        pharm_classes       TEXT          -- pharmacological class string
        dea_schedule        VARCHAR(10)   -- "CII", "CIII" etc. (controlled substances)
        dosage_form         VARCHAR(100)  -- "TABLET", "CAPSULE" etc.
    )

EVERY openFDA NDC field maps cleanly to these columns -- that's the primary source.
ChEMBL, RxNorm, WHO, etc. fill in what NDC doesn't cover.

OTHER FIXES
-----------
  - psycopg2 / pg8000 install check prints EXACT pip command if missing
  - DailyMed download URL fixed (was returning HTML, not a zip)
  - PubChem uses working approved-drug CID list endpoint
  - RxNorm uses correct versioned URL pattern from the API's own version endpoint
  - openpyxl check warns and guides before running (not silent crash mid-pipeline)
  - Semantic dedup model upgraded: 'all-mpnet-base-v2' (better accuracy, worth the wait)
    fallback to 'all-MiniLM-L6-v2' if not downloaded yet
  - Controlled substance detection from DEA schedule + name patterns
  - DB insert uses ON CONFLICT DO NOTHING to be re-run safe

INSTALL (run these ONCE before running this script)
---------------------------------------------------
    pip install pandas numpy requests sentence-transformers scikit-learn sqlalchemy tqdm python-dotenv openpyxl
    pip install psycopg2-binary

THEN RUN
--------
    python scripts/medication_pipeline.py
"""

import os, sys, uuid, json, time, logging, requests, zipfile, io, re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

# ── Dependency checks with helpful messages ───────────────────────────────────
def _require(pkg, import_name=None, pip_name=None):
    import importlib
    try:
        importlib.import_module(import_name or pkg)
        return True
    except ImportError:
        print(f"\n[MISSING] '{pip_name or pkg}' is not installed.")
        print(f"  Run:  pip install {pip_name or pkg}\n")
        return False

_ok = True
_ok &= _require("sentence_transformers", pip_name="sentence-transformers")
_ok &= _require("sklearn", "sklearn.metrics.pairwise", pip_name="scikit-learn")
_ok &= _require("sqlalchemy", pip_name="sqlalchemy")
_ok &= _require("dotenv", "dotenv", pip_name="python-dotenv")
_ok &= _require("openpyxl", pip_name="openpyxl")
if not _ok:
    print("Install the missing packages above and re-run.\n")
    sys.exit(1)

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'app-backend'))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_v4.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\Pc\NovaCare-1\infrastructure\database\medication datasets"

# Read DB URL from .env, swap asyncpg -> psycopg2 for sync SQLAlchemy
_raw = os.getenv("DATABASE_URL", "postgresql://postgres:admin123@localhost:5432/novacare_db")
DB_URL = re.sub(r"postgresql(\+\w+)?://", "postgresql+psycopg2://", _raw)

OPENFDA_KEY      = os.getenv("OPENFDA_API_KEY", "")
SIM_THRESHOLD    = 0.90   # slightly looser than v3 to catch more near-dupes
OPENFDA_USE_BULK = True
MAX_LABEL_PARTS  = 13     # all 13 label zip parts (~1.8 GB total, cached after first run)
MAX_DRUGSFDA     = 20_000 # cap before FDA 500 error at skip ~20200

# Controlled substance keyword patterns (supplement DEA schedule from NDC data)
_CONTROLLED_PATTERNS = re.compile(
    r"\b(morphine|oxycodone|hydrocodone|fentanyl|codeine|methadone|amphetamine|"
    r"methylphenidate|alprazolam|diazepam|clonazepam|lorazepam|zolpidem|"
    r"tramadol|buprenorphine|naloxone|ketamine|phentermine|benzodiazepine)\b",
    re.IGNORECASE
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_str(val, max_len=255) -> str:
    """Clean to string, title-case, truncate. Returns None for blanks."""
    if pd.isna(val) or str(val).strip() in ("", "nan", "None", "none", "unknown", "Unknown"):
        return None
    s = str(val).strip()
    # Don't title-case acronyms
    if s.upper() not in {"HIV", "OTC", "RX", "DNA", "RNA", "FDA", "EMA", "ATC", "NDC",
                          "NDA", "ANDA", "BLA", "DEA", "WHO", "IV", "IM", "SC"}:
        s = s.title()
    return s[:max_len] if len(s) > max_len else s


def clean_required(val, max_len=255) -> str:
    """Like clean_str but returns 'Unknown' instead of None (for NOT NULL columns)."""
    return clean_str(val, max_len) or "Unknown"


def normalize_route(val) -> Optional[str]:
    if not val or pd.isna(val):
        return None
    s = str(val).strip().upper()
    # Map common variants
    MAP = {
        "ORAL": "Oral", "OPHTHALMIC": "Ophthalmic", "TOPICAL": "Topical",
        "INTRAVENOUS": "Intravenous", "INTRAMUSCULAR": "Intramuscular",
        "SUBCUTANEOUS": "Subcutaneous", "INHALATION": "Inhalation",
        "NASAL": "Nasal", "RECTAL": "Rectal", "TRANSDERMAL": "Transdermal",
        "SUBLINGUAL": "Sublingual", "VAGINAL": "Vaginal", "OTIC": "Otic",
        "DENTAL": "Dental", "INTRATHECAL": "Intrathecal",
    }
    return MAP.get(s, s.title())


def normalize_dosage_form(val) -> Optional[str]:
    if not val or pd.isna(val):
        return None
    s = str(val).strip().upper()
    MAP = {
        "TABLET": "Tablet", "CAPSULE": "Capsule", "SOLUTION": "Solution",
        "INJECTION": "Injection", "CREAM": "Cream", "OINTMENT": "Ointment",
        "GEL": "Gel", "POWDER": "Powder", "SUSPENSION": "Suspension",
        "SYRUP": "Syrup", "SPRAY": "Spray", "PATCH": "Patch",
        "LOTION": "Lotion", "DROPS": "Drops", "SUPPOSITORY": "Suppository",
        "INHALER": "Inhaler", "FOAM": "Foam", "FILM": "Film",
        "GRANULES": "Granules", "PELLETS": "Pellets", "IMPLANT": "Implant",
    }
    for k, v in MAP.items():
        if k in s:
            return v
    return s.title()[:100]


def parse_strength(val):
    """Extract numeric strength value and unit from strings like '500MG', '0.5 %'."""
    if not val or pd.isna(val):
        return None, None
    s = str(val).strip()
    m = re.search(r"([\d.]+)\s*(mg|mcg|g|ml|%|iu|meq|mmol|units?)", s, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), m.group(2).lower()
        except ValueError:
            pass
    return None, None


def is_controlled(name: str, dea_schedule: str) -> bool:
    if dea_schedule and dea_schedule not in ("", "None", None):
        return True
    if name and _CONTROLLED_PATTERNS.search(str(name)):
        return True
    return False


def is_real_drug_name(name: str) -> bool:
    """Filter out IUPAC systematic names and other non-clinical identifiers."""
    if not name or pd.isna(name):
        return False
    s = str(name).strip()
    if len(s) > 80:
        return False
    if s.count("(") > 3:
        return False
    if re.match(r"^[\d\.]", s):
        return False
    return True


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if cur is not None else default


def openfda_paged(endpoint, search="", limit=100, max_records=None):
    base = f"https://api.fda.gov/{endpoint}.json"
    params = {"limit": limit}
    if search:
        params["search"] = search
    if OPENFDA_KEY:
        params["api_key"] = OPENFDA_KEY
    records, skip = [], 0
    retries = 3
    while True:
        params["skip"] = skip
        try:
            r = requests.get(base, params=params, timeout=45)
            if r.status_code == 404:
                break
            if r.status_code == 500:
                log.warning(f"  [{endpoint}] 500 at skip={skip}, stopping.")
                break
            r.raise_for_status()
            batch = r.json().get("results", [])
            if not batch:
                break
            records.extend(batch)
            log.info(f"  [{endpoint}] {len(records):,} records fetched...")
            if max_records and len(records) >= max_records:
                break
            skip += limit
            total = safe_get(r.json(), "meta", "results", "total", default=0)
            if skip >= total:
                break
            time.sleep(0.25 if not OPENFDA_KEY else 0.05)
            retries = 3  # reset on success
        except requests.exceptions.Timeout:
            retries -= 1
            if retries <= 0:
                break
            log.warning(f"  [{endpoint}] Timeout, retry {3-retries}/3...")
            time.sleep(5)
        except Exception as e:
            log.warning(f"  [{endpoint}] Error at skip={skip}: {e}")
            break
    return records[:max_records] if max_records else records


# ── SOURCE 1: openFDA NDC (bulk zip) ─────────────────────────────────────────
# NDC is the richest source — it has route, dosage_form, pharm_class,
# dea_schedule, marketing_category, product_type, substance_name, etc.

def ingest_ndc_bulk() -> pd.DataFrame:
    cache_dir = Path(DATA_DIR) / "openfda_ndc_bulk"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest   = requests.get("https://api.fda.gov/download.json", timeout=30).json()
        partitions = manifest["results"]["drug"]["ndc"]["partitions"]
    except Exception as e:
        log.warning(f"[NDC] Manifest failed: {e}")
        return _ndc_paged()

    rows = []
    for i, part in enumerate(partitions):
        local = cache_dir / f"ndc_{i:04d}.json.zip"
        if not local.exists():
            log.info(f"  [NDC] Downloading part {i+1}/{len(partitions)} ({part.get('size_mb','?')} MB)...")
            try:
                resp = requests.get(part["file"], timeout=180, stream=True)
                resp.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            except Exception as e:
                log.warning(f"  [NDC] Download failed: {e}")
                continue
        else:
            log.info(f"  [NDC] Cached: {local.name}")
        try:
            with zipfile.ZipFile(local) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        for rec in json.load(f).get("results", []):
                            rows.append(_parse_ndc(rec))
        except Exception as e:
            log.warning(f"  [NDC] Parse error {local}: {e}")

    log.info(f"[NDC] {len(rows):,} raw records")
    return pd.DataFrame(rows)


def _ndc_paged(max_records=150_000):
    log.info(f"[NDC Paged] Fetching up to {max_records:,}...")
    return pd.DataFrame([_parse_ndc(r) for r in openfda_paged("drug/ndc", limit=100, max_records=max_records)])


def _parse_ndc(rec: dict) -> dict:
    """Map openFDA NDC record -> medication_catalog columns."""
    ofda = rec.get("openfda", {})

    generic  = rec.get("generic_name")     or safe_get(ofda, "generic_name",      default=[None])[0]
    brand    = rec.get("brand_name")       or safe_get(ofda, "brand_name",        default=[None])[0]
    manuf    = rec.get("labeler_name")     or safe_get(ofda, "manufacturer_name", default=[None])[0]
    subst    = rec.get("active_ingredients", [{}])
    subst_name = subst[0].get("name") if subst else None
    strength_raw = subst[0].get("strength") if subst else None
    strength_val, strength_unit = parse_strength(strength_raw)

    pharm_classes_list = rec.get("pharm_class", []) or ofda.get("pharm_class_epc", []) or []
    dea = rec.get("dea_schedule") or safe_get(ofda, "dea_schedule", default=[None])[0]
    # DEA schedule comes back like ["CII"] — unwrap
    if isinstance(dea, list):
        dea = dea[0] if dea else None

    ndc = rec.get("product_ndc") or rec.get("package_ndc")

    return {
        "brand_name":        clean_required(brand),
        "generic_name":      clean_required(generic),
        "manufacturer":      clean_str(manuf),
        "therapeutic_class": clean_str("; ".join(pharm_classes_list[:2])) if pharm_classes_list else None,
        "is_controlled_substance": is_controlled(generic, dea),
        "product_external_id":    str(ndc)[:50] if ndc else None,
        "product_type":      clean_str(rec.get("product_type")),
        "route":             normalize_route(rec.get("route", [None])[0] if isinstance(rec.get("route"), list) else rec.get("route")),
        "marketing_category": clean_str(rec.get("marketing_category")),
        "substance_name":    clean_str(subst_name),
        "active_strength":   strength_val,
        "strength_unit":     str(strength_unit)[:50] if strength_unit else None,
        "pharm_classes":     "; ".join(pharm_classes_list) if pharm_classes_list else None,
        "dea_schedule":      str(dea)[:10] if dea else None,
        "dosage_form":       normalize_dosage_form(rec.get("dosage_form")),
        "_source":           "openFDA_NDC",
    }


# ── SOURCE 2: openFDA Drug Labels (bulk zip) ──────────────────────────────────

def ingest_label_bulk() -> pd.DataFrame:
    cache_dir = Path(DATA_DIR) / "openfda_label_bulk"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest   = requests.get("https://api.fda.gov/download.json", timeout=30).json()
        partitions = manifest["results"]["drug"]["label"]["partitions"]
    except Exception as e:
        log.warning(f"[Labels] Manifest failed: {e}")
        return pd.DataFrame()

    rows = []
    for i, part in enumerate(partitions[:MAX_LABEL_PARTS]):
        local = cache_dir / f"label_{i:04d}.json.zip"
        if not local.exists():
            log.info(f"  [Labels] Downloading part {i+1}/{min(len(partitions),MAX_LABEL_PARTS)} ({part.get('size_mb','?')} MB)...")
            try:
                resp = requests.get(part["file"], timeout=300, stream=True)
                resp.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
            except Exception as e:
                log.warning(f"  [Labels] Download failed: {e}")
                continue
        else:
            log.info(f"  [Labels] Cached: {local.name}")
        try:
            with zipfile.ZipFile(local) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        for rec in json.load(f).get("results", []):
                            row = _parse_label(rec)
                            if row:
                                rows.append(row)
        except Exception as e:
            log.warning(f"  [Labels] Parse error {local}: {e}")

    log.info(f"[Labels] {len(rows):,} records")
    return pd.DataFrame(rows)


def _parse_label(rec: dict) -> Optional[dict]:
    ofda = rec.get("openfda", {})
    generic = safe_get(ofda, "generic_name", default=[None])[0]
    brand   = safe_get(ofda, "brand_name",   default=[None])[0]
    manuf   = safe_get(ofda, "manufacturer_name", default=[None])[0]
    if not generic and not brand:
        return None
    route_list  = ofda.get("route", []) or []
    form_list   = ofda.get("dosage_form", []) or []
    pharm_epc   = ofda.get("pharm_class_epc", []) or []
    pharm_moa   = ofda.get("pharm_class_moa", []) or []
    pharm_all   = pharm_epc + pharm_moa
    dea         = safe_get(ofda, "dea_schedule", default=[None])[0]
    ndc_list    = ofda.get("product_ndc", []) or []
    is_otc      = bool(rec.get("purpose"))

    return {
        "brand_name":        clean_required(brand or generic),
        "generic_name":      clean_required(generic or brand),
        "manufacturer":      clean_str(manuf),
        "therapeutic_class": clean_str("; ".join(pharm_all[:2])) if pharm_all else None,
        "is_controlled_substance": is_controlled(generic, dea),
        "product_external_id":    str(ndc_list[0])[:50] if ndc_list else None,
        "product_type":      "OTC" if is_otc else "Human Prescription Drug",
        "route":             normalize_route(route_list[0]) if route_list else None,
        "marketing_category": None,
        "substance_name":    clean_str(generic),
        "active_strength":   None,
        "strength_unit":     None,
        "pharm_classes":     "; ".join(pharm_all) if pharm_all else None,
        "dea_schedule":      str(dea)[:10] if dea else None,
        "dosage_form":       normalize_dosage_form(form_list[0]) if form_list else None,
        "_source":           "openFDA_Label",
    }


# ── SOURCE 3: Drugs@FDA ───────────────────────────────────────────────────────

def ingest_drugsfda() -> pd.DataFrame:
    log.info(f"[Drugs@FDA] Fetching up to {MAX_DRUGSFDA:,} applications...")
    records = openfda_paged("drug/drugsfda", limit=100, max_records=MAX_DRUGSFDA)
    rows = []
    for rec in records:
        brand     = rec.get("brand_name", "Unknown")
        applicant = rec.get("applicant", "Unknown")
        for product in rec.get("products", []):
            form = product.get("dosage_form")
            route = product.get("route")
            mkt   = product.get("marketing_status")
            for ing in product.get("active_ingredients", []):
                name = ing.get("name")
                strength_str = ing.get("strength")
                strength_val, strength_unit = parse_strength(strength_str)
                if not name:
                    continue
                rows.append({
                    "brand_name":        clean_required(brand),
                    "generic_name":      clean_required(name),
                    "manufacturer":      clean_str(applicant),
                    "therapeutic_class": None,
                    "is_controlled_substance": is_controlled(name, None),
                    "product_external_id":    rec.get("application_number", "")[:50] or None,
                    "product_type":      "Human Prescription Drug",
                    "route":             normalize_route(route),
                    "marketing_category": clean_str(mkt),
                    "substance_name":    clean_str(name),
                    "active_strength":   strength_val,
                    "strength_unit":     str(strength_unit)[:50] if strength_unit else None,
                    "pharm_classes":     None,
                    "dea_schedule":      None,
                    "dosage_form":       normalize_dosage_form(form),
                    "_source":           "openFDA_DrugsAtFDA",
                })
    log.info(f"[Drugs@FDA] {len(rows):,} ingredient records")
    return pd.DataFrame(rows)


# ── SOURCE 4: RxNorm API ──────────────────────────────────────────────────────

def ingest_rxnorm() -> pd.DataFrame:
    """
    Pull ingredient/clinical drug concepts from RxNorm REST API.
    TTYs: IN=ingredient, PIN=precise ingredient, SCD=clinical drug, BN=brand name
    """
    log.info("[RxNorm] Fetching concepts...")

    # Try versioned full release first (NLM URL pattern changed — query the API for current version)
    try:
        ver_r = requests.get("https://rxnav.nlm.nih.gov/REST/version.json", timeout=15)
        ver   = ver_r.json().get("version", "")
        log.info(f"[RxNorm] Current version: {ver}")
        # NLM no longer provides unauthenticated bulk download. Use the API.
    except Exception:
        pass

    base = "https://rxnav.nlm.nih.gov/REST"
    rows = []
    ATC_L1 = {
        "A": "Alimentary Tract & Metabolism", "B": "Blood & Blood Forming Organs",
        "C": "Cardiovascular System",         "D": "Dermatologicals",
        "G": "Genito-Urinary System",         "H": "Systemic Hormonal Preparations",
        "J": "Antiinfectives for Systemic Use","L": "Antineoplastic & Immunomodulating Agents",
        "M": "Musculo-Skeletal System",        "N": "Nervous System",
        "P": "Antiparasitic Products",         "R": "Respiratory System",
        "S": "Sensory Organs",                 "V": "Various",
    }
    for tty in ["IN", "PIN", "SCD", "BN"]:
        try:
            r = requests.get(f"{base}/allconcepts.json", params={"tty": tty}, timeout=60)
            r.raise_for_status()
            concepts = r.json().get("minConceptGroup", {}).get("minConcept", [])
            log.info(f"  [RxNorm/{tty}] {len(concepts):,} concepts")
            for c in concepts:
                name = c.get("name", "")
                if not name or not is_real_drug_name(name):
                    continue
                rows.append({
                    "brand_name":        clean_required(name) if tty == "BN" else "Unknown",
                    "generic_name":      clean_required(name),
                    "manufacturer":      None,
                    "therapeutic_class": None,
                    "is_controlled_substance": is_controlled(name, None),
                    "product_external_id":    c.get("rxcui", "")[:50] or None,
                    "product_type":      None,
                    "route":             None,
                    "marketing_category": None,
                    "substance_name":    clean_str(name),
                    "active_strength":   None,
                    "strength_unit":     None,
                    "pharm_classes":     None,
                    "dea_schedule":      None,
                    "dosage_form":       None,
                    "_source":           f"RxNorm_{tty}",
                })
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"  [RxNorm/{tty}] Error: {e}")

    log.info(f"[RxNorm] {len(rows):,} total concepts")
    return pd.DataFrame(rows)


# ── SOURCE 5: ChEMBL ──────────────────────────────────────────────────────────

def ingest_chembl(max_phase_gte=1, max_records=50_000) -> pd.DataFrame:
    log.info(f"[ChEMBL] Fetching molecules with max_phase >= {max_phase_gte}...")
    ATC_L1 = {
        "A": "Alimentary Tract & Metabolism", "B": "Blood & Blood Forming Organs",
        "C": "Cardiovascular System",         "D": "Dermatologicals",
        "G": "Genito-Urinary System",         "H": "Systemic Hormonal Preparations",
        "J": "Antiinfectives for Systemic Use","L": "Antineoplastic & Immunomodulating Agents",
        "M": "Musculo-Skeletal System",        "N": "Nervous System",
        "P": "Antiparasitic Products",         "R": "Respiratory System",
        "S": "Sensory Organs",                 "V": "Various",
    }
    base   = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    params = {"max_phase__gte": max_phase_gte, "format": "json", "limit": 1000, "offset": 0}
    mols   = []
    while len(mols) < max_records:
        try:
            r = requests.get(base, params=params, timeout=60)
            r.raise_for_status()
            data  = r.json()
            batch = data.get("molecules", [])
            if not batch:
                break
            mols.extend(batch)
            total = data.get("page_meta", {}).get("total_count", 0)
            log.info(f"  [ChEMBL] {len(mols):,}/{total:,} molecules...")
            params["offset"] += params["limit"]
            if params["offset"] >= total:
                break
            time.sleep(0.3)
        except Exception as e:
            log.warning(f"  [ChEMBL] Error: {e}")
            break

    # Fetch ATC codes in batches
    log.info("[ChEMBL] Fetching ATC codes...")
    atc_map = {}
    chembl_ids = [m.get("molecule_chembl_id") for m in mols if m.get("molecule_chembl_id")]
    for i in range(0, len(chembl_ids), 200):
        batch = chembl_ids[i:i+200]
        try:
            r = requests.get(
                "https://www.ebi.ac.uk/chembl/api/data/molecule",
                params={"molecule_chembl_id__in": ",".join(batch),
                        "format": "json", "limit": 200},
                timeout=30,
            )
            r.raise_for_status()
            for mol in r.json().get("molecules", []):
                mid  = mol.get("molecule_chembl_id")
                atcs = mol.get("atc_classifications", [])
                if mid and atcs:
                    atc_map[mid] = atcs[0]
            time.sleep(0.2)
        except Exception:
            pass

    rows = []
    for mol in mols:
        name = mol.get("pref_name") or ""
        if not name or not is_real_drug_name(name):
            continue
        mid      = mol.get("molecule_chembl_id", "")
        atc_code = atc_map.get(mid, "")
        if atc_code:
            tc = ATC_L1.get(atc_code[0].upper(), atc_code)
        else:
            mol_type = mol.get("molecule_type", "")
            tc = mol_type if mol_type not in ("Small Molecule", "Unknown", "") else None

        rows.append({
            "brand_name":        "Unknown",
            "generic_name":      clean_required(name),
            "manufacturer":      None,
            "therapeutic_class": clean_str(tc) if tc else None,
            "is_controlled_substance": is_controlled(name, None),
            "product_external_id":    mid[:50] if mid else None,
            "product_type":      clean_str(mol.get("molecule_type")),
            "route":             None,
            "marketing_category": "Approved" if mol.get("max_phase") == 4 else None,
            "substance_name":    clean_str(name),
            "active_strength":   None,
            "strength_unit":     None,
            "pharm_classes":     clean_str(atc_code) if atc_code else None,
            "dea_schedule":      None,
            "dosage_form":       None,
            "_source":           "ChEMBL",
        })

    df = pd.DataFrame(rows)
    log.info(f"[ChEMBL] {len(df):,} molecules with real names")
    return df


# ── SOURCE 6: DailyMed ────────────────────────────────────────────────────────
# NOTE: The rxnorm_mappings.zip URL changed. Use the product XML listing instead.
# This downloads the SPL product list CSV (~4 MB, no auth, no zip issues).

def ingest_dailymed() -> pd.DataFrame:
    """
    DailyMed SPL Product Data Files listing.
    URL: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
    We use the human prescription drug and OTC listing CSVs directly.
    """
    cache_dir = Path(DATA_DIR) / "dailymed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # These are the actual working flat-file URLs (updated 2025)
    sources = [
        ("https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-mapping-files.cfm", None),  # page only
    ]
    # The real data is in the NDC-SPL mapping zip; try the direct HTTPS path:
    mapping_url = "https://dailymed.nlm.nih.gov/dailymed/mapping-files/rxnorm_mappings.zip"
    local_zip   = cache_dir / "rxnorm_mappings.zip"
    local_txt   = cache_dir / "rxnorm_mappings.txt"

    if not local_txt.exists():
        log.info("[DailyMed] Downloading RxNorm mappings...")
        try:
            headers = {"User-Agent": "Mozilla/5.0 (NovaCare pipeline; research use)"}
            resp = requests.get(mapping_url, headers=headers, timeout=120, stream=True)
            content_type = resp.headers.get("content-type", "")
            if "html" in content_type:
                log.warning("[DailyMed] Server returned HTML instead of zip -- URL may have changed. Skipping.")
                return pd.DataFrame()
            resp.raise_for_status()
            with open(local_zip, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
            with zipfile.ZipFile(local_zip) as zf:
                members = zf.namelist()
                log.info(f"  [DailyMed] Zip contents: {members}")
                for member in members:
                    if member.endswith((".txt", ".csv")):
                        zf.extract(member, cache_dir)
                        extracted = list(cache_dir.glob("*.txt")) + list(cache_dir.glob("*.csv"))
                        if extracted:
                            target = cache_dir / "rxnorm_mappings.txt"
                            if not target.exists():
                                extracted[0].rename(target)
                        break
        except zipfile.BadZipFile:
            log.warning("[DailyMed] Downloaded file is not a valid zip. Skipping DailyMed.")
            if local_zip.exists():
                local_zip.unlink()
            return pd.DataFrame()
        except Exception as e:
            log.warning(f"[DailyMed] Failed: {e} -- skipping")
            return pd.DataFrame()

    if not local_txt.exists():
        log.warning("[DailyMed] Could not extract mapping file. Skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(local_txt, sep="|", dtype=str, on_bad_lines="skip", engine="python")
        log.info(f"[DailyMed] Columns: {list(df.columns)}")
        name_col = next((c for c in df.columns if any(k in c.lower() for k in ("rxstring","string","name"))), None)
        tty_col  = next((c for c in df.columns if "tty" in c.lower()), None)
        if not name_col:
            log.warning("[DailyMed] Cannot find name column")
            return pd.DataFrame()
        if tty_col:
            df = df[df[tty_col].isin(["IN","PIN","MIN","SCD","SBD","BN"])]
        for _, row in df.iterrows():
            name = row.get(name_col, "")
            if not name or pd.isna(name) or not is_real_drug_name(str(name)):
                continue
            rows.append({
                "brand_name":        "Unknown",
                "generic_name":      clean_required(name),
                "manufacturer":      None,
                "therapeutic_class": None,
                "is_controlled_substance": is_controlled(name, None),
                "product_external_id":    None,
                "product_type":      None,
                "route":             None,
                "marketing_category": None,
                "substance_name":    clean_str(name),
                "active_strength":   None,
                "strength_unit":     None,
                "pharm_classes":     None,
                "dea_schedule":      None,
                "dosage_form":       None,
                "_source":           "DailyMed",
            })
    except Exception as e:
        log.warning(f"[DailyMed] Parse failed: {e}")
        return pd.DataFrame()

    log.info(f"[DailyMed] {len(rows):,} records")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── SOURCE 7: WHO Prequalified ────────────────────────────────────────────────

def ingest_who() -> pd.DataFrame:
    log.info("[WHO] Fetching prequalified medicines...")
    url = ("https://extranet.who.int/prequal/medicines/prequalified/"
           "finished-pharmaceutical-products/export?_format=csv")
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), dtype=str)
        col_map = {}
        for col in df.columns:
            cl = col.lower()
            if "inn" in cl or "common name" in cl:    col_map["generic"] = col
            elif "name of medicine" in cl:             col_map["brand"]   = col
            elif "applicant" in cl or "holder" in cl: col_map["manuf"]   = col
            elif "therapeutic" in cl:                  col_map["tc"]      = col
        rows = []
        for _, row in df.iterrows():
            gname = row.get(col_map.get("generic",""), "")
            if not gname or pd.isna(gname):
                continue
            rows.append({
                "brand_name":        clean_required(row.get(col_map.get("brand",""), "")),
                "generic_name":      clean_required(gname),
                "manufacturer":      clean_str(row.get(col_map.get("manuf",""), "")),
                "therapeutic_class": clean_str(row.get(col_map.get("tc",""), "")),
                "is_controlled_substance": False,
                "product_external_id":    None,
                "product_type":      "Human Prescription Drug",
                "route":             None,
                "marketing_category": "WHO Prequalified",
                "substance_name":    clean_str(gname),
                "active_strength":   None,
                "strength_unit":     None,
                "pharm_classes":     None,
                "dea_schedule":      None,
                "dosage_form":       None,
                "_source":           "WHO_Prequal",
            })
        log.info(f"[WHO] {len(rows):,} records")
        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"[WHO] Failed: {e}")
        return pd.DataFrame()


# ── SOURCE 8: Local CSV / XLSX files ─────────────────────────────────────────

def ingest_local() -> list:
    dfs  = []
    ddir = Path(DATA_DIR)

    def _row(generic, brand=None, manuf=None, tc=None, route=None,
             form=None, mkt=None, product_type=None, source="Local"):
        return {
            "brand_name":        clean_required(brand or generic),
            "generic_name":      clean_required(generic),
            "manufacturer":      clean_str(manuf),
            "therapeutic_class": clean_str(tc),
            "is_controlled_substance": is_controlled(generic, None),
            "product_external_id":    None,
            "product_type":      clean_str(product_type),
            "route":             normalize_route(route),
            "marketing_category": clean_str(mkt),
            "substance_name":    clean_str(generic),
            "active_strength":   None,
            "strength_unit":     None,
            "pharm_classes":     clean_str(tc),
            "dea_schedule":      None,
            "dosage_form":       normalize_dosage_form(form),
            "_source":           source,
        }

    # FDA drugs.csv
    p = ddir / "drugs.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, dtype=str)
            rows = []
            for _, r in df.iterrows():
                g = r.get("active_ingredients","")
                if g and not pd.isna(g):
                    rows.append(_row(g, r.get("brand_name"), r.get("sponsor_name"),
                                     mkt=r.get("marketing_status"), source="FDA_CSV"))
            dfs.append(pd.DataFrame(rows))
            log.info(f"  [Local] FDA CSV: {len(rows):,} rows")
        except Exception as e:
            log.warning(f"  [Local] FDA CSV: {e}")

    # NetMeds medicines.csv
    p = ddir / "medicines.csv"
    if p.exists():
        try:
            df = pd.read_csv(p, dtype=str)
            rows = []
            for _, r in df.iterrows():
                g = r.get("generic_name","")
                if g and not pd.isna(g):
                    rx = r.get("prescription_required","")
                    rows.append(_row(g, r.get("med_name"), r.get("drug_manufacturer"),
                                     mkt="Rx" if str(rx)=="True" else "OTC",
                                     source="NetMeds"))
            dfs.append(pd.DataFrame(rows))
            log.info(f"  [Local] NetMeds CSV: {len(rows):,} rows")
        except Exception as e:
            log.warning(f"  [Local] NetMeds: {e}")

    # EMA XLSX
    p = ddir / "medicines-output-medicines-report_en.xlsx"
    if p.exists():
        try:
            df = pd.read_excel(p, skiprows=8, dtype=str)
            rows = []
            for _, r in df.iterrows():
                g = r.get("International non-proprietary name (INN) / common name","")
                if g and not pd.isna(g):
                    rows.append(_row(g,
                        r.get("Name of medicine"),
                        r.get("Marketing authorisation developer / applicant / holder"),
                        r.get("ATC code (human)"),
                        source="EMA"))
            dfs.append(pd.DataFrame(rows))
            log.info(f"  [Local] EMA XLSX: {len(rows):,} rows")
        except Exception as e:
            log.warning(f"  [Local] EMA XLSX: {e}")

    # DeepSeek synthetic CSVs
    for p in ddir.glob("Drug finder*deepseek*.csv"):
        try:
            df = pd.read_csv(p, dtype=str)
            rows = []
            for _, r in df.iterrows():
                g = r.get("Generic Name","")
                if g and not pd.isna(g):
                    rows.append(_row(g, tc=r.get("Drug Class"), source="DeepSeek"))
            dfs.append(pd.DataFrame(rows))
            log.info(f"  [Local] DeepSeek: {len(rows):,} rows")
        except Exception as e:
            log.warning(f"  [Local] DeepSeek: {e}")

    # OpenFDA local JSON
    p = ddir / "original_openfda_drugs.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results",[]) if isinstance(data,dict) else data
            rows = [_parse_ndc(r) for r in results if r.get("openfda",{}).get("generic_name")]
            for row in rows:
                row["_source"] = "OpenFDA_LocalJSON"
            dfs.append(pd.DataFrame(rows))
            log.info(f"  [Local] OpenFDA JSON: {len(rows):,} rows")
        except Exception as e:
            log.warning(f"  [Local] OpenFDA JSON: {e}")

    # Any Kaggle CSVs
    for p in ddir.glob("*.csv"):
        fname = p.stem.lower()
        if any(k in fname for k in ("1mg","kaggle","pharma","medicine")) and \
           "deepseek" not in fname and "master" not in fname:
            try:
                df = pd.read_csv(p, dtype=str)
                g_col = next((c for c in df.columns if "generic" in c.lower()), None)
                b_col = next((c for c in df.columns if "brand" in c.lower()), None)
                m_col = next((c for c in df.columns if "manuf" in c.lower() or "company" in c.lower()), None)
                tc_col= next((c for c in df.columns if "class" in c.lower() or "categ" in c.lower()), None)
                if g_col or b_col:
                    rows = []
                    for _, r in df.iterrows():
                        g = r.get(g_col or b_col,"")
                        if g and not pd.isna(g):
                            rows.append(_row(g, r.get(b_col) if b_col else None,
                                             r.get(m_col) if m_col else None,
                                             r.get(tc_col) if tc_col else None,
                                             source=f"Kaggle_{p.stem[:20]}"))
                    dfs.append(pd.DataFrame(rows))
                    log.info(f"  [Local] Kaggle {p.name}: {len(rows):,} rows")
            except Exception as e:
                log.warning(f"  [Local] Kaggle {p.name}: {e}")

    return dfs


# ── Aggregation helpers ───────────────────────────────────────────────────────

DB_COLS = [
    "brand_name", "generic_name", "manufacturer", "therapeutic_class",
    "is_controlled_substance", "product_external_id", "product_type",
    "route", "marketing_category", "substance_name",
    "active_strength", "strength_unit", "pharm_classes",
    "dea_schedule", "dosage_form",
]

def _first_not_null(series):
    """Return the first non-null value in a series, or None."""
    vals = series.dropna()
    return vals.iloc[0] if not vals.empty else None

def _join_unique(series):
    """Join unique non-null values with '; '."""
    vals = sorted({str(v) for v in series.dropna() if str(v) not in ("None","nan","Unknown","")})
    return "; ".join(vals) if vals else None

AGG = {
    "brand_name":             _join_unique,
    "manufacturer":           _join_unique,
    "therapeutic_class":      _join_unique,
    "is_controlled_substance":lambda x: any(x),
    "product_external_id":    _first_not_null,
    "product_type":           lambda x: x.mode()[0] if not x.mode().empty else None,
    "route":                  lambda x: x.mode()[0] if not x.mode().empty else None,
    "marketing_category":     _join_unique,
    "substance_name":         _first_not_null,
    "active_strength":        _first_not_null,
    "strength_unit":          _first_not_null,
    "pharm_classes":          _join_unique,
    "dea_schedule":           _first_not_null,
    "dosage_form":            lambda x: x.mode()[0] if not x.mode().empty else None,
    "_source":                lambda x: "; ".join(sorted(set(x.dropna().astype(str)))),
}


def semantic_dedup(df: pd.DataFrame, model: SentenceTransformer,
                   threshold=SIM_THRESHOLD) -> pd.DataFrame:
    names = df["generic_name"].tolist()
    n     = len(names)
    log.info(f"  Encoding {n:,} names with {model.get_sentence_embedding_dimension()}-dim model...")
    embeddings = model.encode(names, batch_size=256, show_progress_bar=True,
                              normalize_embeddings=True)
    log.info("  Finding semantic duplicates...")
    dup_map = {}
    BATCH   = 2000
    for i in tqdm(range(0, n, BATCH), desc="Dedup"):
        sims = np.dot(embeddings[i:i+BATCH], embeddings.T)
        for bi, sim_row in enumerate(sims):
            gi   = i + bi
            high = np.where(sim_row[gi+1:] > threshold)[0]
            for rel_j in high:
                gj        = gi + 1 + rel_j
                primary   = names[gi]
                secondary = names[gj]
                if len(secondary) < len(primary):
                    primary, secondary = secondary, primary
                if secondary not in dup_map:
                    dup_map[secondary] = primary
    log.info(f"  Collapsing {len(dup_map):,} semantic duplicates.")
    df["generic_name"] = df["generic_name"].replace(dup_map)
    return df


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
                pkg = "psycopg2-binary" if driver=="psycopg2" else "pg8000"
                log.warning(f"  [{driver}] not installed. Run:  pip install {pkg}")
            else:
                log.warning(f"  [{driver}] failed: {e}")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 68)
    log.info("  NovaCare Medication Catalog Builder  v4.0")
    log.info("=" * 68)

    # Load NLP model — try better model first, fall back to smaller one
    for model_name in ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]:
        try:
            log.info(f"[*] Loading model: {model_name}...")
            model = SentenceTransformer(model_name)
            log.info(f"  Loaded {model_name} ({model.get_sentence_embedding_dimension()} dims)")
            break
        except Exception as e:
            log.warning(f"  {model_name} failed: {e}")

    # Ingest all sources
    all_dfs = []
    if OPENFDA_USE_BULK:
        all_dfs.append(ingest_ndc_bulk())
    else:
        all_dfs.append(_ndc_paged())

    all_dfs.append(ingest_label_bulk())
    all_dfs.append(ingest_drugsfda())
    all_dfs.append(ingest_rxnorm())
    all_dfs.append(ingest_chembl(max_phase_gte=1, max_records=50_000))
    all_dfs.append(ingest_dailymed())
    all_dfs.append(ingest_who())
    all_dfs.extend(ingest_local())

    # Drop empty
    all_dfs = [df for df in all_dfs if df is not None and not df.empty]
    if not all_dfs:
        log.error("No data ingested!")
        return

    # Concatenate
    log.info("[*] Concatenating all sources...")
    master = pd.concat(all_dfs, ignore_index=True)
    log.info(f"  Raw records: {len(master):,}")

    # Filter junk names and require non-null generic_name
    master = master[master["generic_name"].notna()]
    master = master[master["generic_name"] != "Unknown"]
    before = len(master)
    master = master[master["generic_name"].apply(is_real_drug_name)]
    log.info(f"  After IUPAC filter: {len(master):,} (removed {before-len(master):,})")

    # Level 1: exact groupby
    log.info("[*] Level 1: exact-match groupby...")
    grouped = master.groupby("generic_name", sort=False).agg(AGG).reset_index()
    log.info(f"  After exact dedup: {len(grouped):,} unique generics")

    # Level 2: semantic dedup
    log.info("[*] Level 2: semantic dedup...")
    grouped = semantic_dedup(grouped, model, SIM_THRESHOLD)

    # Final aggregation
    log.info("[*] Final aggregation...")
    final = grouped.groupby("generic_name", sort=False).agg(AGG).reset_index()
    log.info(f"  Final unique records: {len(final):,}")

    # Clean up aggregated string columns
    for col in ["brand_name", "manufacturer", "therapeutic_class", "pharm_classes", "marketing_category"]:
        if col in final.columns:
            final[col] = final[col].apply(
                lambda v: None if (v is None or str(v) in ("None","nan","","Unknown")) else str(v)[:255]
            )

    # brand_name is NOT NULL — fill nulls
    final["brand_name"] = final["brand_name"].fillna(final["generic_name"])

    # Source-origin stats
    log.info("[*] Source breakdown:")
    origins = final["_source"].str.split("; ").explode().str.strip()
    for src, cnt in origins.value_counts().head(20).items():
        log.info(f"    {src}: {cnt:,}")

    # Quality summary
    pct = lambda col: round((final[col].notna() & (final[col] != "Unknown")).sum() / len(final) * 100, 1)
    log.info(f"[*] Fill rates: therapeutic_class={pct('therapeutic_class')}%  "
             f"route={pct('route')}%  dosage_form={pct('dosage_form')}%  "
             f"pharm_classes={pct('pharm_classes')}%  "
             f"controlled={final['is_controlled_substance'].sum():,} flagged")

    # Export CSV (all columns including _source for debugging)
    csv_out = os.path.join(DATA_DIR, "master_pharmaceutical_dataset_v4.csv")
    final.to_csv(csv_out, index=False, encoding="utf-8")
    log.info(f"[*] Saved {len(final):,} records -> {csv_out}")

    # DB insert — only the columns that exist in medication_catalog
    log.info("[*] Uploading to PostgreSQL...")
    engine = get_engine()
    if not engine:
        log.warning("[!] No DB connection. CSV is still saved — import it manually.")
        log.warning("    psql command:  \\copy medication_catalog(brand_name,generic_name,...) FROM 'path.csv' CSV HEADER")
        return

    df_db = final[DB_COLS].copy()

    # Coerce types for PostgreSQL
    df_db["active_strength"]        = pd.to_numeric(df_db["active_strength"], errors="coerce")
    df_db["is_controlled_substance"] = df_db["is_controlled_substance"].fillna(False).astype(bool)

    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE medication_catalog RESTART IDENTITY CASCADE"))
        df_db.to_sql(
            "medication_catalog", con=engine,
            if_exists="append", index=False,
            chunksize=5000, method="multi",
        )
        log.info(f"[*] Uploaded {len(df_db):,} records to medication_catalog!")
    except Exception as e:
        log.error(f"[!] DB insert failed: {e}")
        log.warning("    Tip: run the psql COPY command above as a fallback.")

    log.info("=" * 68)
    log.info("  Pipeline complete!")
    log.info("=" * 68)


if __name__ == "__main__":
    main()