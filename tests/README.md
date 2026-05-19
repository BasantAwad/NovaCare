# NovaCare PostgreSQL Integration Tests

## Purpose
These tests verify that NovaCare components correctly **read from and write to PostgreSQL** and that the database is the **single source of truth**.

## Prerequisites
- Python 3.10+
- PostgreSQL installed
- A test database created (example: `novacare_test_db`)

## Environment variables
Create a `.env.test` in the repo root with at least:

- `DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/novacare_test_db`

## Install test dependencies
```bash
pip install -r test-requirements.txt
```

## Run
```bash
pytest -q
```

## Notes
- Tests are written as transaction-wrapped DB operations. Each test uses `asyncpg` and rolls back automatically.
- The repository schema is assumed to already exist.

