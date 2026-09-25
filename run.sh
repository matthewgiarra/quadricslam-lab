#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-8765}" --reload
