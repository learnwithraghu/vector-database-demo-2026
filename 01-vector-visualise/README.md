# Document Embedding Visualization with ChromaDB

This folder contains a notebook that demonstrates **loading a PDF, chunking text, embedding with sentence-transformers, storing in ChromaDB, and visualizing the embedding space in 3D with Plotly**.

## Notebook

- **`chroma_embedding_visualization.ipynb`** – End-to-end pipeline:
  1. Load a PDF and extract text (PyPDF2)
  2. Chunk text with overlap (500 chars, 50 overlap)
  3. Create a ChromaDB collection with sentence-transformers (`all-MiniLM-L6-v2`)
  4. Add chunks and metadata to ChromaDB
  5. Visualize embeddings in 3D with **Plotly** (PCA-reduced, coloured by chunk position)

## Setup

1. **Run the setup script** (creates a venv and installs dependencies):

   ```bash
   cd 01-vector-visualise
   ./setup.sh
   ```

2. **Select the kernel**  
   In Jupyter/VS Code, choose **Python (chroma_demo)**.

3. **Add a PDF**  
   Place your PDF in this folder (e.g. `use_2025_budget.pdf`) and set `PDF_PATH` in the notebook.

## Requirements

The setup script installs:

- `chromadb`
- `PyPDF2`
- `sentence-transformers`
- `plotly`
- `scikit-learn`
- `pandas`
- `jupyter`, `ipykernel`

First run may take ~30–60 s for imports and ~1–2 min for the embedding model download.
