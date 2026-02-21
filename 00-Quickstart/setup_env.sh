#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up the VectorDB Streamlit App Environment..."

echo "We recommend running this application via Docker."
read -p "Do you want to build and run the Docker container now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Building Docker image 'lancedb-quickstart'..."
    docker build -t lancedb-quickstart .
    echo "Starting Docker container on port 8501..."
    docker run -p 8501:8501 lancedb-quickstart
    exit 0
fi

echo "Proceeding with local Python environment setup..."

# 1. Create a virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# 2. Activate the environment and install dependencies
source "$VENV_DIR/bin/activate"

echo "Installing dependencies using pip3 from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "✅ Local Setup Complete!"
echo "To start the Streamlit App locally:"
echo "1. Activate the environment: source .venv/bin/activate"
echo "2. Run the application: streamlit run streamlit_app/app.py"
