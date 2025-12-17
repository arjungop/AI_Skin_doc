#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

echo "==> Setting up Python venv (.venv)"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "==> Installing backend requirements"
pip install -r requirements.txt >/dev/null

echo "==> Ensuring frontend dependencies"
pushd frontend-react >/dev/null
if [[ ! -d node_modules ]]; then
  npm install >/dev/null
fi
popd >/dev/null

export ADMIN_EMAIL="admin@example.com"
export ADMIN_USERNAME="admin"
export ADMIN_PASSWORD="Admin@12345"

BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"

echo "==> Starting backend (FastAPI) on http://${BACKEND_HOST}:${BACKEND_PORT}"
uvicorn backend.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" --reload &
BACK_PID=$!

cleanup() {
  echo "\n==> Shutting down (PID $BACK_PID)"
  kill "$BACK_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting frontend (Vite dev server)"
echo "    API base: http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "    Admin login: admin@example.com / Admin@12345"
pushd frontend-react >/dev/null
npm run dev
popd >/dev/null

