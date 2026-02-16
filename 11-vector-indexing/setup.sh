#!/usr/bin/env bash
# =============================================================
#  Setup script for the LanceDB Indexing & Quantization Demo
#  Run once:  bash setup.sh
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
KERNEL_NAME="vector-db-demo"

echo "──────────────────────────────────────────────"
echo "  📦  Creating virtual environment (.venv) …"
echo "──────────────────────────────────────────────"
python3 -m venv "$VENV_DIR"

echo ""
echo "  📥  Installing dependencies …"
echo ""
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "  🧠  Registering Jupyter kernel: $KERNEL_NAME"
echo ""
python -m ipykernel install --user --name "$KERNEL_NAME" \
       --display-name "Vector DB Demo (Python)"

echo ""
echo "──────────────────────────────────────────────"
echo "  ✅  All done!"
echo ""
echo "  To use the notebook:"
echo "    1. Open indexing_and_quantization_demo.ipynb"
echo "    2. Select kernel → 'Vector DB Demo (Python)'"
echo "──────────────────────────────────────────────"
