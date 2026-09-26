#!/bin/sh
# Container startup: create/migrate the SQLite database and seed the
# deterministic demo item bank *before* uvicorn starts accepting traffic.
# Idempotent: safe to run on every container start and alongside the
# FastAPI startup hook.
set -e

python -c "from app.database import init_db; init_db()"

exec "$@"
