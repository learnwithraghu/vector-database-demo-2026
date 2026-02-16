# 🔍 Vector Database Internals — Indexing & Quantization Demo

A hands-on series of Jupyter notebooks demonstrating how **indexing** (IVF) and **quantization** (PQ / SQ) work inside a vector database, using [LanceDB](https://lancedb.com/) and 200K synthetic vectors.

## Quick Start

### Step 1: Run the setup script

```bash
cd /path/to/this/folder
bash setup.sh
```

This creates a Python virtual environment (`.venv`), installs all dependencies, and registers a Jupyter kernel named **"Vector DB Demo (Python)"**.

### Step 2: Attach the kernel to the notebooks

1. Open any of the notebooks below in **VS Code** or **Jupyter**
2. In the kernel picker (top-right in VS Code, or Kernel menu in Jupyter), select:
   > **Vector DB Demo (Python)**
3. Run the cells!

> ⚠️ **Important:** You must select the `Vector DB Demo (Python)` kernel, not your default Python kernel. This ensures the correct virtual environment and packages are used.

## Notebooks

Run them in order — each one builds on concepts from the previous:

| # | Notebook | Topic |
|---|----------|-------|
| 1 | `01_indexing.ipynb` | Brute-force baseline → IVF index → `nprobes` tradeoff |
| 2 | `02_product_quantization.ipynb` | PQ compression → refine factor → sub-vector tuning |
| 3 | `03_scalar_quantization.ipynb` | SQ compression → PQ vs SQ comparison → decision guide |

## Requirements

- Python 3.9+
- macOS / Linux
- ~500 MB disk space (vectors + indexes)
