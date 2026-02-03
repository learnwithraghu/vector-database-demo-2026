#!/bin/bash

# Vector Database Demo - Environment Setup Script
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "=========================================="
echo "Vector Database Demo - Environment Setup"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "✓ Found $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Register Jupyter kernel
echo "🔗 Registering Jupyter kernel..."
python -m ipykernel install --user --name=vector-db-demo --display-name "Python (Vector DB Demo)"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo ""
echo "Then select 'Python (Vector DB Demo)' as the kernel"
echo "=========================================="
