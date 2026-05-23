#!/usr/bin/env python3
"""
Import pharmaceutical CSV dataset into PostgreSQL database.
"""

import pandas as pd
from sqlalchemy import create_engine, text, inspect

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_FILE_PATH = r"C:\Users\Pc\NovaCare-1\infrastructure\database\medication datasets\master_pharmaceutical_dataset_v4.csv"
DATABASE_URL = "postgresql+psycopg2://postgres:admin123@localhost:5432/novacare_db"
TABLE_NAME = "pharmaceutical_products"
CHUNK_SIZE = 5000
IF_EXISTS = "replace"  # 'replace', 'append', 'fail'

# =============================================================================
# SCRIPT
# =============================================================================

def main():
    engine = create_engine(DATABASE_URL, echo=False)

    # Test connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ Connected to database: {DATABASE_URL}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # Read CSV in chunks
    try:
        total_rows = 0
        first_chunk = True

        # Use 'c' engine with proper quoting (faster, handles most cases)
        # For problematic CSV with embedded newlines, use engine='python' but remove low_memory
        for i, chunk in enumerate(pd.read_csv(
            CSV_FILE_PATH,
            chunksize=CHUNK_SIZE,
            encoding='utf-8',
            engine='c',          # Use C engine for speed; fallback to 'python' if issues
            # If you must use 'python', remove low_memory and set:
            # engine='python', low_memory=None (or omit)
            on_bad_lines='skip', # Skip problematic lines if any
        )):
            # Replace NaN with None
            chunk = chunk.where(pd.notnull(chunk), None)

            if first_chunk:
                if IF_EXISTS == 'replace':
                    chunk.to_sql(TABLE_NAME, engine, if_exists='replace', index=False, method='multi')
                elif IF_EXISTS == 'append':
                    chunk.to_sql(TABLE_NAME, engine, if_exists='append', index=False, method='multi')
                elif IF_EXISTS == 'fail':
                    inspector = inspect(engine)
                    if inspector.has_table(TABLE_NAME):
                        print(f"❌ Table '{TABLE_NAME}' already exists and IF_EXISTS='fail'.")
                        return
                    else:
                        chunk.to_sql(TABLE_NAME, engine, if_exists='fail', index=False, method='multi')
                first_chunk = False
            else:
                chunk.to_sql(TABLE_NAME, engine, if_exists='append', index=False, method='multi')

            total_rows += len(chunk)
            print(f"📦 Chunk {i+1}: {len(chunk)} rows inserted (total {total_rows})")

        print(f"✅ Import completed: {total_rows} rows written to table '{TABLE_NAME}'.")

    except pd.errors.EmptyDataError:
        print("❌ CSV file is empty.")
    except FileNotFoundError:
        print(f"❌ CSV file not found: {CSV_FILE_PATH}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()