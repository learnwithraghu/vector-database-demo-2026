# ZVec Query Latency Demo - Dockerfile
# Base image with Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY query_latency_demo.ipynb .
COPY pdf_doc/ ./pdf_doc/

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "========================================"\n\
echo "ZVec Query Latency Demo"\n\
echo "========================================"\n\
echo ""\n\
echo "Starting Jupyter Notebook..."\n\
echo ""\n\
echo "Access the notebook at:"\n\
echo "  http://localhost:8888"\n\
echo ""\n\
echo "The token will be displayed below."\n\
echo "========================================"\n\
echo ""\n\
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password=""\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run Jupyter Notebook
CMD ["/app/start.sh"]
