FROM qdrant/qdrant:latest

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m venv /workspace/venv && \
    /workspace/venv/bin/pip install --no-cache-dir -r /workspace/requirements.txt

COPY setup_and_connection.ipynb /workspace/setup_and_connection.ipynb
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

EXPOSE 6333 6334 8888

CMD ["/workspace/start.sh"]
