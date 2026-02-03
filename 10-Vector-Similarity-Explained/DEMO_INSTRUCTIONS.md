# Vector Similarity Explained - Demo Instructions

## 🎯 Quick Start Guide

### Setup (One-time)
```bash
cd 10-Vector-Similarity-Explained
bash setup_env.sh
```

### Running the Demo
```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

### Notebook Sequence
1. **01_embeddings_introduction.ipynb** (8-10 min)
   - Generate embeddings for 30 fruits
   - Visualize in 2D with t-SNE
   - Calculate similarity scores

2. **02_chromadb_storage.ipynb** (8-10 min)
   - Store vectors in ChromaDB
   - Perform similarity searches
   - Use metadata filtering

3. **03_similarity_metrics.ipynb** (8-10 min)
   - Compare cosine, Euclidean, dot product
   - Show different results for same query
   - Visualize ranking differences

## 📊 What Students Will Learn

- How embeddings capture semantic meaning
- How to use vector databases (ChromaDB)
- Why different similarity metrics give different results
- When to use each metric in production

## 🔑 Key Demonstrations

### Notebook 1: "Aha Moment"
Students see fruits cluster by similarity in 2D visualization

### Notebook 2: "Practical Application"
Students learn to store and query vectors efficiently

### Notebook 3: "Critical Insight"
Students discover that metric choice significantly impacts results

---

**All notebooks run in-memory - no persistence needed!**
