#!/bin/bash
# Setup script for ChromaDB Document Embedding Visualization
# Run this BEFORE opening the notebook, then select the "chroma_demo" kernel in Jupyter.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_NAME="chroma_demo"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"

# Find a Python that supports venv (try common versions)
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
  if command -v "$cmd" &>/dev/null && "$cmd" -m venv --help &>/dev/null; then
    PYTHON="$cmd"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "❌ No suitable Python found. Install Python 3.10+ with: brew install python"
  exit 1
fi

echo "📦 Creating virtual environment: $VENV_NAME (using $PYTHON)"
"$PYTHON" -m venv "$VENV_PATH"

if [[ ! -d "$VENV_PATH/bin" ]]; then
  echo "❌ Failed to create venv at $VENV_PATH"
  exit 1
fi

echo "📥 Activating and installing dependencies..."
echo "   (sentence-transformers + PyTorch can take 5–15 min on first run)"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip -q
pip install chromadb PyPDF2 sentence-transformers plotly jupyter ipykernel scikit-learn pandas

echo "🔧 Registering Jupyter kernel..."
python -m ipykernel install --user --name="$VENV_NAME" --display-name="Python (chroma_demo)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Launching Jupyter Notebook (browser will open)..."
echo "   Select 'Python (chroma_demo)' kernel if prompted."
echo ""
jupyter notebook chroma_embedding_visualization.ipynb
