#!/bin/bash

# ZVec Query Latency Demo - Docker Runner
# This script builds and runs the Docker container with Jupyter Notebook

set -e

echo "=========================================="
echo "ZVec Query Latency Demo - Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker from: https://www.docker.com/get-started"
    exit 1
fi

echo "✓ Docker is installed"
echo ""

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "Error: docker-compose is not available."
    echo "Please install docker-compose or use Docker Desktop."
    exit 1
fi

echo "✓ Docker Compose is available"
echo ""

echo "Building Docker image..."
echo "This may take a few minutes..."
echo ""
echo "Note: Rebuilding from scratch to ensure latest version..."
echo ""

# Remove old image and rebuild from scratch
$COMPOSE_CMD down --rmi local 2>/dev/null || true
$COMPOSE_CMD build --no-cache

echo ""
echo "=========================================="
echo "Starting Jupyter Notebook..."
echo "=========================================="
echo ""
echo "The notebook will be available at:"
echo "  http://localhost:8888"
echo ""
echo "No token required - direct access enabled"
echo ""
echo "To stop the container:"
echo "  Press Ctrl+C or run: $COMPOSE_CMD down"
echo ""
echo "=========================================="
echo ""

# Run the container
$COMPOSE_CMD up
