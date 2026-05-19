"""infrastructure.database.crud

Centralized CRUD helpers for async SQLAlchemy.

This module is designed to be consumed by backend services/routes so they fetch
real data from the database instead of using static placeholders.

Notes:
- Your project currently has both Flask/MySQL controllers and SQLAlchemy/async
  infrastructure. This CRUD module targets the SQLAlchemy async engine exposed
  by `infrastructure.database.connection.get_db()`.
- Add further domain-specific CRUD functions as your routes need them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _bind_params(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return params or {}


async def fetch_one(session: AsyncSession, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single row as a dict; returns None if no row."""
    stmt = text(query)
    result = await session.execute(stmt, _bind_params(params))
    row = result.mappings().first()
    return dict(row) if row is not None else None


async def fetch_all(session: AsyncSession, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Fetch all rows as list of dicts."""
    stmt = text(query)
    result = await session.execute(stmt, _bind_params(params))
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def execute(session: AsyncSession, query: str, params: Optional[Dict[str, Any]] = None) -> int:
    """Execute a write query (INSERT/UPDATE/DELETE) and return affected rows."""
    stmt = text(query)
    result = await session.execute(stmt, _bind_params(params))
    return int(result.rowcount or 0)


async def upsert_example(
    session: AsyncSession,
    table: str,
    pk_col: str,
    pk_value: Any,
    update_cols: Dict[str, Any],
) -> None:
    """Generic UPSERT example for Postgres.

    Domain-specific upserts should be created as needed.

    This uses Postgres' ON CONFLICT; other DBs may require different syntax.
    """

    if not update_cols:
        return

    cols = [pk_col] + list(update_cols.keys())
    insert_cols_sql = ", ".join(cols)
    insert_values_sql = ", ".join([f":{c}" for c in cols])
    set_sql = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols.keys()])

    query = f"""
        INSERT INTO {table} ({insert_cols_sql})
        VALUES ({insert_values_sql})
        ON CONFLICT ({pk_col}) DO UPDATE
        SET {set_sql}
    """

    params: Dict[str, Any] = {pk_col: pk_value, **update_cols}
    await session.execute(text(query), params)

