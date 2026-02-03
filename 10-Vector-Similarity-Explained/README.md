# Vector Similarity Explained 📊

A comprehensive hands-on demonstration of vector databases using ChromaDB and Sentence Transformers. This module teaches MLOps engineers the fundamentals of vector embeddings, storage, and similarity search through practical examples using a fruit dataset.

## 📚 Learning Objectives

- Understand what embeddings are and how they represent semantic meaning
- Learn to generate embeddings using Sentence Transformers
- Store and retrieve vectors using ChromaDB
- Compare different similarity metrics (cosine, Euclidean, dot product)
- Understand how metric choice affects search results

## 📋 Prerequisites

- **Python**: 3.9, 3.10, or 3.11
- **Operating System**: macOS, Linux, or Windows with WSL
- **Basic Knowledge**: Python programming, basic ML concepts

## 🚀 Quick Start

### 1. Navigate to This Directory

```bash
cd 10-Vector-Similarity-Explained
```

### 2. Run the Setup Script

```bash
bash setup_env.sh
```

This script will:
- Create a virtual environment
- Install all dependencies
- Register a Jupyter kernel

### 3. Activate the Virtual Environment

```bash
source venv/bin/activate
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Select the Correct Kernel

In Jupyter, select **"Python (Vector DB Demo)"** as your kernel for all notebooks.

## 📓 Notebooks Overview

### Notebook 1: Introduction to Embeddings
**Duration**: 8-10 minutes  
**File**: `notebooks/01_embeddings_introduction.ipynb`

- What are embeddings and why use them?
- Generating embeddings for fruits using Sentence Transformers
- Visualizing embeddings in 2D space
- Understanding vector representations

### Notebook 2: ChromaDB Storage and Retrieval
**Duration**: 8-10 minutes  
**File**: `notebooks/02_chromadb_storage.ipynb`

- Introduction to ChromaDB vector database
- Storing fruit embeddings in ChromaDB
- Basic similarity search and retrieval
- Working with metadata and filtering

### Notebook 3: Similarity Metrics Comparison
**Duration**: 8-10 minutes  
**File**: `notebooks/03_similarity_metrics.ipynb`

- Understanding cosine similarity, Euclidean distance, and dot product
- Creating separate collections for each metric
- Comparing search results across metrics
- Visualizing differences in ranking
- When to use which metric

## 📁 Project Structure

```
10-Vector-Similarity-Explained/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_env.sh                       # Environment setup script
├── data/
│   └── fruits.json                    # Fruit dataset (30 fruits)
├── notebooks/
│   ├── 01_embeddings_introduction.ipynb
│   ├── 02_chromadb_storage.ipynb
│   └── 03_similarity_metrics.ipynb
└── venv/                              # Virtual environment (created by setup)
```

## 🔧 Manual Setup (Alternative)

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=vector-db-demo --display-name "Python (Vector DB Demo)"
```

## 🎯 Running the Notebooks

**Important**: Run all cells in each notebook sequentially from top to bottom.

1. Start with `01_embeddings_introduction.ipynb`
2. Then proceed to `02_chromadb_storage.ipynb`
3. Finally, run `03_similarity_metrics.ipynb`

Each notebook is self-contained but builds on concepts from previous ones.

## 🛠️ Troubleshooting

### Virtual Environment Not Activating

**macOS/Linux**:
```bash
source venv/bin/activate
```

**Windows (WSL)**:
```bash
source venv/bin/activate
```

### Kernel Not Showing in Jupyter

```bash
python -m ipykernel install --user --name=vector-db-demo --display-name "Python (Vector DB Demo)"
```

Then restart Jupyter.

### Import Errors

Make sure the virtual environment is activated:
```bash
which python  # Should show path to venv/bin/python
```

If not, activate it:
```bash
source venv/bin/activate
```

### ChromaDB Errors

ChromaDB runs in-memory mode in this demo. If you encounter errors, restart the Jupyter kernel:
- In Jupyter: `Kernel` → `Restart Kernel`

## 📦 Dependencies

- **chromadb**: Vector database
- **sentence-transformers**: Embedding generation
- **numpy**: Numerical operations
- **pandas**: Data manipulation
- **matplotlib & seaborn**: Visualizations
- **jupyter**: Notebook environment

See `requirements.txt` for specific versions.

## 🎓 Key Concepts Covered

- **Embeddings**: Dense vector representations of text
- **Vector Databases**: Specialized databases for similarity search
- **Cosine Similarity**: Measures angle between vectors (0-1, higher is more similar)
- **Euclidean Distance**: Measures straight-line distance (lower is more similar)
- **Dot Product**: Measures vector alignment (higher is more similar)
- **Semantic Search**: Finding similar items based on meaning, not keywords

## 📝 Notes

- All ChromaDB collections use **in-memory** storage (data is not persisted)
- Each notebook run starts fresh
- The fruit dataset contains 30 fruits with descriptions and metadata
- Embeddings are generated using the `all-MiniLM-L6-v2` model (384 dimensions)

## 🤝 For Instructors

Each notebook is designed for 8-10 minute demonstrations:
- Clear explanations before each code cell
- Detailed print statements during execution
- Visualizations to reinforce concepts
- Progressive complexity across notebooks

---

**Happy Learning! 🚀**
