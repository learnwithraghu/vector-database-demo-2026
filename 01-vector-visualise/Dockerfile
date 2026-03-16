# Document Embedding Visualization with ChromaDB
# Jupyter notebook environment with baked-in sample PDF

FROM python:3.11-slim

WORKDIR /workspace

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy notebooks, guide, and PDF (baked into image)
COPY chroma_embedding_visualization.ipynb chroma_embedding_visualization_student.ipynb guide.md use_2025_budget.pdf .

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter - allow root for Docker, bind to all interfaces
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
