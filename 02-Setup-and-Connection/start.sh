#!/usr/bin/env bash
set -euo pipefail

echo "Starting Qdrant..."
/qdrant/qdrant &

echo "Waiting for Qdrant on localhost:6333..."
until curl -sf "http://localhost:6333/collections" >/dev/null; do
  sleep 1
done
echo "Qdrant is ready."

echo "Starting JupyterLab on port 8888..."
exec /workspace/venv/bin/jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.allow_origin='*' \
  --ServerApp.token='' \
  --ServerApp.password=''
