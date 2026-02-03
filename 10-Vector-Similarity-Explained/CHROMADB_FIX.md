# ChromaDB API Fix - Applied to Notebooks

## Issue
The notebooks were using an outdated ChromaDB API with `Settings` class:
```python
from chromadb.config import Settings
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=None
))
```

This caused a `ValidationError` because newer ChromaDB versions don't accept `None` for `persist_directory`.

## Solution
Updated to the simpler, current API for in-memory mode:
```python
import chromadb
client = chromadb.Client()  # In-memory by default
```

## Files Fixed
- ✅ `notebooks/02_chromadb_storage.ipynb`
- ✅ `notebooks/03_similarity_metrics.ipynb`

## Changes Made
1. Removed `from chromadb.config import Settings` import
2. Simplified client initialization to `chromadb.Client()`
3. Cleared error outputs from notebook cells

The notebooks now work with ChromaDB 0.4.22 and will run without errors!
