# Student Guide: Document Embedding Visualization

Complete the exercises in `chroma_embedding_visualization_student.ipynb` by filling in the gaps. Use this guide for hints and to unblock yourself.

---

## Exercise 1: Import PCA

**Task:** Add the missing import for PCA (Principal Component Analysis) from scikit-learn.

**Hint:** PCA is used for dimensionality reduction. It lives in `sklearn.decomposition`.

<details>
<summary>Answer</summary>

```python
from sklearn.decomposition import PCA
```

Add this line with the other imports (e.g., after `import plotly.express as px`).
</details>

---

## Exercise 2: PDF Reader Class

**Task:** Replace `XXX` with the correct PyPDF2 class for reading a PDF file.

**Hint:** The class reads a PDF from a file object. Its name suggests it "reads" a "PDF".

<details>
<summary>Answer</summary>

```python
reader = PyPDF2.PdfReader(f)
```

Replace `XXX` with `PdfReader`.
</details>

---

## Exercise 3: Extract Chunk from Text

**Task:** Add the missing line that extracts and strips the chunk from the text.

**Hint:** Use the `start` and `end` indices to slice the text, then strip whitespace. The variable should be named `chunk`.

<details>
<summary>Answer</summary>

```python
chunk = text[start:end].strip()
```

Add this line before `if chunk:`.
</details>

---

## Exercise 4: Embedding Model Name

**Task:** Set the `model_name` for the sentence-transformers embedding function.

**Hint:** We use a lightweight 384-dimensional model: `all-MiniLM-L6-v2`.

<details>
<summary>Answer</summary>

```python
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```

Replace `"FILL_MODEL_NAME"` with `"all-MiniLM-L6-v2"`.
</details>

---

## Exercise 5: PDF Path

**Task:** Set `PDF_PATH` to the PDF included in the Docker image.

**Hint:** The baked-in file is the U.S. Budget 2025 document: `use_2025_budget.pdf`.

<details>
<summary>Answer</summary>

```python
PDF_PATH = "use_2025_budget.pdf"
```

Replace `"FILL_PDF_PATH"` with `"use_2025_budget.pdf"`.
</details>

---

## Exercise 6: ChromaDB get() Include Parameter

**Task:** Pass the correct `include` list to `collection.get()` so we retrieve embeddings, documents, and metadata.

**Hint:** We need to fetch three types of data: `"embeddings"`, `"documents"`, and `"metadatas"`.

<details>
<summary>Answer</summary>

```python
results = collection.get(include=["embeddings", "documents", "metadatas"])
```

Replace `include=[]` with `include=["embeddings", "documents", "metadatas"]`.
</details>

---

## Multiple Choice Questions

### MCQ 1: Why do we use overlapping chunks instead of non-overlapping chunks?

**A)** Overlapping chunks use less memory  
**B)** Overlap helps preserve context at chunk boundaries, improving retrieval quality  
**C)** Overlapping chunks are faster to embed  
**D)** ChromaDB requires overlapping chunks

<details>
<summary>Answer</summary>

**B)** Overlap helps preserve context at chunk boundaries, improving retrieval quality.

When we split text at hard boundaries, important context can be lost at the edges. Overlap ensures that phrases spanning two chunks are still captured in at least one chunk.
</details>

---

### MCQ 2: What does PCA do in this notebook?

**A)** It increases the dimensionality of embeddings for better accuracy  
**B)** It reduces 384-dimensional embeddings to 3 dimensions for visualization  
**C)** It trains the embedding model  
**D)** It stores embeddings in ChromaDB

<details>
<summary>Answer</summary>

**B)** It reduces 384-dimensional embeddings to 3 dimensions for visualization.

PCA (Principal Component Analysis) projects high-dimensional data onto fewer dimensions while preserving as much variance as possible. We use it to visualize 384-D vectors in 3D space.
</details>

---

### MCQ 3: What is the role of ChromaDB in this pipeline?

**A)** It generates the embeddings from text  
**B)** It stores and retrieves vector embeddings, enabling similarity search  
**C)** It loads and parses PDF files  
**D)** It creates the 3D visualization

<details>
<summary>Answer</summary>

**B)** It stores and retrieves vector embeddings, enabling similarity search.

ChromaDB is a vector database. It stores embeddings (from sentence-transformers) and supports fast similarity search. PDF loading is done by PyPDF2, and visualization by Plotly.
</details>

---

## Quick Reference: Full Answers

| Exercise | Answer |
|----------|--------|
| 1 | `from sklearn.decomposition import PCA` |
| 2 | `PyPDF2.PdfReader` |
| 3 | `chunk = text[start:end].strip()` |
| 4 | `model_name="all-MiniLM-L6-v2"` |
| 5 | `PDF_PATH = "use_2025_budget.pdf"` |
| 6 | `include=["embeddings", "documents", "metadatas"]` |
