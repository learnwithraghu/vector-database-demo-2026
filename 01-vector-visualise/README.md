# Document Embedding Visualization with ChromaDB

This folder contains a notebook that demonstrates **loading a PDF, chunking text, embedding with sentence-transformers, storing in ChromaDB, and visualizing the embedding space in 3D with Plotly**.

## Notebooks

- **`chroma_embedding_visualization.ipynb`** – Full demo (instructor version):
  1. Load a PDF and extract text (PyPDF2)
  2. Chunk text with overlap (500 chars, 50 overlap)
  3. Create a ChromaDB collection with sentence-transformers (`all-MiniLM-L6-v2`)
  4. Add chunks and metadata to ChromaDB
  5. Visualize embeddings in 3D with **Plotly** (PCA-reduced, coloured by chunk position)

- **`chroma_embedding_visualization_student.ipynb`** – Student exercise with gaps to fill. Use **`guide.md`** for hints and answers.

---

## Run with Docker (recommended)

Docker provides an isolated environment with all dependencies and a sample PDF baked in.

### Build the image

```bash
cd 01-vector-visualise
docker build -t chroma-embedding-demo .
```

### Run the container

```bash
docker run -p 8888:8888 chroma-embedding-demo
```

### Access Jupyter

1. Check the terminal output for a URL like:
   ```
   http://127.0.0.1:8888/?token=abc123...
   ```
2. Open that URL in your browser (replace `127.0.0.1` with `localhost` if needed).
3. Open `chroma_embedding_visualization.ipynb` for the demo, or `chroma_embedding_visualization_student.ipynb` for the exercise.

**Note:** The U.S. Budget 2025 PDF (`use_2025_budget.pdf`) is included in the image. No extra setup required.

---

## Local setup (without Docker)

1. **Run the setup script** (creates a venv and installs dependencies):

   ```bash
   cd 01-vector-visualise
   ./setup.sh
   ```

2. **Select the kernel**  
   In Jupyter/VS Code, choose **Python (chroma_demo)**.

3. **Add a PDF**  
   Place your PDF in this folder (e.g. `use_2025_budget.pdf`) and set `PDF_PATH` in the notebook. For the student exercise, you can use any PDF or create one.

---

## Requirements

- `chromadb`
- `PyPDF2`
- `sentence-transformers`
- `plotly`
- `scikit-learn`
- `pandas`
- `jupyter`, `ipykernel`

First run may take ~30–60 s for imports and ~1–2 min for the embedding model download.
