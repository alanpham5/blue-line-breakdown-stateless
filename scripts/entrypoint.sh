#!/bin/sh
set -e

echo "Starting backend..."

PORT="${PORT:-8080}"
WORKERS="${GUNICORN_WORKERS:-1}"
TIMEOUT="${GUNICORN_TIMEOUT:-600}"

exec gunicorn --bind "0.0.0.0:${PORT}" --timeout "${TIMEOUT}" --workers "${WORKERS}" app:app
