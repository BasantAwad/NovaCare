import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text   # <-- ADD THIS IMPORT


def load_database_url() -> str:
    # mirrors infrastructure/database/connection.py
    root_env_path = Path(__file__).resolve().parents[1] / '.env'
    if not root_env_path.exists():
        # fallback: repo root .env
        root_env_path = Path(__file__).resolve().parents[2] / '.env'
    load_dotenv(root_env_path)

    url = os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError('DATABASE_URL is not set (check repo .env)')
    return url


async def main():
    database_url = load_database_url()
    print('Testing DATABASE_URL:', database_url)

    engine = create_async_engine(database_url, echo=False)
    try:
        async with engine.connect() as conn:
            # Wrap raw SQL in text()
            res = await conn.execute(text('SELECT 1'))
            val = res.scalar_one()
            print('DB connectivity OK. SELECT 1 =', val)
    finally:
        await engine.dispose()


if __name__ == '__main__':
    asyncio.run(main())