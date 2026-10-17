#!/usr/bin/env bash
# Runs backend + frontend in one container (Render, Railway, etc.).
# The backend stays private on localhost; only Streamlit is public on $PORT.
set -e

uvicorn main:app --host 127.0.0.1 --port 8000 --no-access-log &

# Wait for the backend before serving the UI
for _ in $(seq 1 30); do
  curl -s http://127.0.0.1:8000/ > /dev/null && break
  sleep 1
done

exec streamlit run app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
