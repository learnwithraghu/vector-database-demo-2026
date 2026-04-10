# Vector Database Demos

A collection of hands-on labs covering vector database concepts, from basic embeddings to advanced cloud integrations. Each module is self-contained with Docker support.

## Course Prerequisites

- **Docker** (for containerized demos)
- **Python 3.8+**
- **Basic Python knowledge**

---

## Labs Overview

### Getting Started

#### 00-Quickstart
Streamlit app for quick exploration using Warren Buffett shareholder letters. Demonstrates document ingestion, embedding, and free-text querying with LanceDB.

#### 01-vector-visualise
Interactive visualization of ChromaDB embeddings. Explains what vectors are and how similarity works through 2D projection.

---

### Core Concepts

#### 02-Setup-and-Connection
Qdrant running in Docker with JupyterLab. Verify your setup and run your first connection.

#### 03-query-latency-vector-db
Two-notebook flow: build embeddings from a PDF, then measure query latency and throughput (p50/p95/p99).

---

### Text Embedding

#### 04-Text-Embedding
Text embedding with LanceDB using airline policy documents. Learn chunking, embedding generation, and vector storage. Includes student notebook with fill-in exercises.

#### 05-vectordb-normadb-search
Side-by-side comparison of SQL (Postgres) and vector search (LanceDB) on IT support tickets. Demonstrates semantic vs keyword search.

---

### Search Techniques

#### 06-Semantic-Search
Fruit similarity demos showing how embeddings capture meaning. Compare results across different similarity metrics.

#### 07-Metadata-Filtering
Theory and practice of metadata filtering. Learn how filters transform "vibe check" searches into precise data queries.

#### 08-Hybrid-Search
Combines dense (semantic) and sparse (keyword) vectors for production-quality search. Covers BM25, SPLADE, and Reciprocal Rank Fusion.

---

### Specialized Domains

#### 09-Image-Embedding-LanceDB
Image embeddings using CLIP and LanceDB. Encode PNG images into vectors and search by visual similarity.

#### 10-AWS-S3-Integration
AWS S3 Vectors for cloud-native vector storage. Connect to S3 Vector Buckets, upload policy document embeddings, and run semantic queries.

#### 11-Vector-Similarity-Explained
Deep dive into similarity metrics: cosine similarity, Euclidean distance, and dot product. Compare results across ChromaDB collections.

#### 12-vector-indexing
Vector indexing concepts with ChromaDB. Visualize the difference between linear scan and indexed search.

#### 13-image-embedding
Image embedding using CLIP and ChromaDB. Learn the pipeline from pixels to vectors to database storage.

#### 14-visual-learning
Interactive visual demos and lessons for visual learners. Explore vector concepts through animations and interactive examples.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/learnwithraghu/vector-database-demo-2026.git
cd vector-database-demo-2026

# Run any lab with Docker
cd <lab-folder>
docker build -t <lab-name> .
docker run -p 8888:8888 <lab-name>
```

Open `http://localhost:8888` and run the notebook.

---

## Shared Datasets

The `datasets/` folder contains shared data files used across multiple labs:

- `airline_policy_dataset.json` - Airline policy documents for text embedding labs

---

## Lab Structure

Each lab folder follows a consistent pattern:

- `README.md` - Setup and run instructions
- `*.ipynb` - Instructor notebook (complete)
- `*_student.ipynb` - Student notebook (fill-in exercises)
- `guide.md` - Step-by-step instructions for students
- `Dockerfile` - Container definition
- `requirements.txt` - Python dependencies

---

## Future Topics

- RAG (Retrieval Augmented Generation) with LLMs
- Multimodal search (text + image combined)
- Real-time indexing and updates
- Evaluation metrics and benchmarking
