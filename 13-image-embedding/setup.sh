#!/bin/bash
# Local venv for 13-image-embedding (CLIP + ChromaDB + Jupyter).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME="image_embedding_demo"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
  if command -v "$cmd" &>/dev/null && "$cmd" -m venv --help &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "No suitable Python found. Install Python 3.10+."
  exit 1
fi

echo "Creating venv: $VENV_NAME ($PYTHON)"
"$PYTHON" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt

python -m ipykernel install --user --name="$VENV_NAME" --display-name="Python (image_embedding_demo)"

echo "Done. Open image_embedding_chroma.ipynb and select kernel Python (image_embedding_demo)."
