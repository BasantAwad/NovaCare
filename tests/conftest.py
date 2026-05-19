import asyncio
import os
from pathlib import Path
import uuid

import asyncpg
import pytest
from dotenv import load_dotenv


def _load_env() -> None:
    # Prefer .env.test at repo root; fallback to .env
    repo_root = Path(__file__).resolve().parents[1]
    env_test = repo_root / ".env.test"
    env_default = repo_root / ".env"

    if env_test.exists():
        load_dotenv(env_test, override=True)
    elif env_default.exists():
        load_dotenv(env_default, override=True)


_load_env()


def _require(name: str, default=None):
    v = os.getenv(name, default)
    if v in (None, ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return v


@pytest.fixture(scope="session")
def event_loop():
    # pytest-asyncio needs an explicit loop fixture sometimes.
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def db_pool():
    # Use a dedicated test DB.
    # Example DATABASE_URL in .env.test:
    # postgresql+asyncpg://user:pass@localhost:5432/novacare_test_db
    # However, asyncpg needs a native URL.
    database_url = _require("DATABASE_URL")
    # If SQLAlchemy-style prefix exists, strip to asyncpg compatible URL.
    # asyncpg accepts postgresql://... (with user/pass).
    database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")

    pool = await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=5)
    return pool


@pytest.fixture(scope="function")
async def db_conn(db_pool):
    async with db_pool.acquire() as conn:
        yield conn


@pytest.fixture(scope="function")
async def tx(db_conn):
    async with db_conn.transaction():
        yield


# Seed IDs: keep within VARCHAR(36) exactly.
@pytest.fixture(scope="function")
def test_rover_id():
    return "RV_TST_0000000000000000000000000"  # 35/36 depends; ensure <= 36


@pytest.fixture(scope="function")
def test_doctor_id():
    return "DOC_TST_0000000000000000000000000"


@pytest.fixture(scope="function")
def test_caregiver_id():
    return "CG_TST_0000000000000000000000000"



@pytest.fixture(scope="function")
async def seed_entities(db_conn, tx, test_rover_id, test_doctor_id, test_caregiver_id):
    # The schema is large; seed only what FK checks require for our tests.
    # Use user rows so rovers/doctors/caregivers reference users.
    user_rover = "U_TST_ROVER_00000000000000000000000"  # <=36
    user_doc = "U_TST_DOC_00000000000000000000000"      # <=36
    user_cg = "U_TST_CG_00000000000000000000000"        # <=36

    # Insert users (idempotent)
    await db_conn.execute(
        """
        INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, is_email_verified, is_active, created_at, updated_at)
        VALUES ($1,$2,NULL,NULL,'Test','Rover',TRUE,TRUE, NOW(), NOW())
        ON CONFLICT (id) DO NOTHING
        """,
        user_rover,
        "rover.tst@example.com",
    )
    await db_conn.execute(
        """
        INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, is_email_verified, is_active, created_at, updated_at)
        VALUES ($1,$2,NULL,NULL,'Test','Doctor',TRUE,TRUE, NOW(), NOW())
        ON CONFLICT (id) DO NOTHING
        """,
        user_doc,
        "doctor.tst@example.com",
    )
    await db_conn.execute(
        """
        INSERT INTO users (id, email, hashed_password, google_id, first_name, last_name, is_email_verified, is_active, created_at, updated_at)
        VALUES ($1,$2,NULL,NULL,'Test','Caregiver',TRUE,TRUE, NOW(), NOW())
        ON CONFLICT (id) DO NOTHING
        """,
        user_cg,
        "caregiver.tst@example.com",
    )


    # Minimal lookup rows for enums/FKs
    # verification_statuses: pick 'pending' row if exists, else insert.
    vs = await db_conn.fetchrow("SELECT id FROM verification_statuses WHERE status_name = 'pending' LIMIT 1")
    if not vs:
        await db_conn.execute(
            "INSERT INTO verification_statuses (id, status_name, display_label) VALUES ($1,'pending','Verification Pending')",
            "V_TST_PENDING_0000000000000000000000000001",
        )
        vs_id = "V_TST_PENDING_0000000000000000000000000001"
    else:
        vs_id = vs["id"]

    # addresses are optional for FK unless specified; we won't set address_id fields.

    # caregivers
    await db_conn.execute(
        """
        INSERT INTO caregivers (id, user_id, phone_number, address_id, government_id_number, verification_status_id)
        VALUES ($1,$2,'+1000000000',NULL,'ID-TST',$3)
        """,
        test_caregiver_id,
        user_cg,
        vs_id,
    )

    # rovers: primary_caregiver_id expects FK to caregivers.id
    await db_conn.execute(
        """
        INSERT INTO rovers (id, user_id, date_of_birth, gender, address_id, primary_caregiver_id)
        VALUES ($1,$2,'1990-01-01','male',NULL,$3)
        """,
        test_rover_id,
        user_rover,
        test_caregiver_id,
    )

    # specializations optional but doctors.specialization_id can be NULL
    # doctors
    await db_conn.execute(
        """
        INSERT INTO doctors (id, user_id, medical_license_num, specialization_id, verification_status_id)
        VALUES ($1,$2,'LIC-TST',NULL,$3)
        """,
        test_doctor_id,
        user_doc,
        vs_id,
    )

    # relationship_types + assignment optional for caregiver<->rover; our tests use primary_caregiver_id.
    # Insert sessions minimal for online checks if needed.

    return {
        "user_rover": user_rover,
        "user_doc": user_doc,
        "user_cg": user_cg,
    }


