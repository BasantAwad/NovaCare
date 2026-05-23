"""Async SQLAlchemy CRUD smoke test.

Runs a minimal read against the live database using:
- infrastructure.database.connection.get_db()
- infrastructure.database.crud.fetch_one / fetch_all / execute

Usage:
    python scripts/crud_smoke_test.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from infrastructure.database.connection import get_db
from infrastructure.database.crud import fetch_one, fetch_all, execute




async def main() -> None:
    async for session in get_db():
        # 1) Read test: does a known table exist?
        row = await fetch_one(
            session,
            "SELECT to_regclass('public.users') AS tbl",
        )
        print("users table present?:", row.get("tbl") if row else None)

        # 2) Read test: fetch a small sample.
        users = await fetch_all(
            session,
            "SELECT id, email FROM users ORDER BY created_at DESC NULLS LAST LIMIT 3",
        )
        print("sample users:", users)

        # 3) Write test: insert a row into configuration then delete it.
        # NOTE: assumes table `configuration(key, value)` exists (from novacare_db.sql).
        key = "crud_smoke_test_key"
        await execute(
            session,
            "INSERT INTO configuration (key, value) VALUES (:key, :val) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            {"key": key, "val": "1"},
        )

        verify = await fetch_one(
            session,
            "SELECT key, value FROM configuration WHERE key=:key",
            {"key": key},
        )
        print("configuration write verify:", verify)

        await execute(
            session,
            "DELETE FROM configuration WHERE key=:key",
            {"key": key},
        )

        print("CRUD smoke test finished OK")


if __name__ == "__main__":
    asyncio.run(main())

