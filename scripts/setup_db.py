import asyncio
import asyncpg
import os

DB_URL = "postgresql://postgres:admin123@localhost:5432/postgres"
TARGET_DB = "novacare_db"

async def setup_db():
    print("Connecting to default postgres database...")
    try:
        sys_conn = await asyncpg.connect(DB_URL)
        # Check if database exists
        exists = await sys_conn.fetchval(f"SELECT 1 FROM pg_database WHERE datname='{TARGET_DB}'")
        if exists:
            print(f"Dropping database {TARGET_DB}...")
            await sys_conn.execute(f"DROP DATABASE {TARGET_DB} WITH (FORCE)")
            
        print(f"Creating database {TARGET_DB}...")
        await sys_conn.execute(f"CREATE DATABASE {TARGET_DB}")

        await sys_conn.close()

        print(f"Connecting to {TARGET_DB}...")
        conn = await asyncpg.connect(f"postgresql://postgres:admin123@localhost:5432/{TARGET_DB}")
        
        sql_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "novacare_db.sql")
        print(f"Reading SQL file: {sql_file}")
        with open(sql_file, "r", encoding="utf-8") as f:
            sql = f.read()

        print("Executing SQL script (this may take a moment)...")
        await conn.execute(sql)
        print("Database setup complete!")
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(setup_db())
