#!/usr/bin/env bash
# Start the API and the UI, and stop the container if either one dies.
#
# Both run as children rather than one being exec'd. `exec` would replace this
# shell, discarding the trap along with it, and a crashed UI would leave the
# API running as an orphan -- a container that looks alive while serving
# nothing on its public port.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
PORT="${PORT:-7860}"

cleanup() {
    kill "${API_PID:-}" "${UI_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --log-level warning &
API_PID=$!

for _ in $(seq 1 60); do
    # Checked with python rather than curl: the slim base image has no curl,
    # and installing one just to poll a local port is not worth the layer.
    if python -c "
import sys, urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:${API_PORT}/health', timeout=2)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "API failed to start" >&2
        exit 1
    fi
    sleep 1
done

export RAG_API_URL="http://127.0.0.1:${API_PORT}"

streamlit run ui/streamlit_app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &
UI_PID=$!

# Exit as soon as either process does, so the platform restarts the container
# instead of leaving half of it running.
wait -n
