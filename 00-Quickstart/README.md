# 🐳 LanceDB Dockerized Streamlit App - Lab Engineer Guide

This folder contains a fully Dockerized interactive Streamlit application to explore vector databases using five years of Warren Buffett's shareholder letters (2020-2024).

*(Note: The visualization notebook has been moved to the `01-vector-visualise` directory).*

## 🛠️ Lab Engineer Setup (Docker Recommended)
We recommend running this lab using Docker to avoid environment conflicts.

1. Ensure Docker is running.
2. Build and start the container:
```bash
docker build -t lancedb-quickstart .
docker run -p 8501:8501 lancedb-quickstart
```
*(Alternatively, run `bash setup_env.sh` and follow the prompts).*

## 🚀 Running the Demonstration
Once the container is running or the local environment is active, navigate to `http://localhost:8501` to view the Streamlit Application.

## 🎓 Student Experience
### 1. The Streamlit App (`app.py`)
- **Document Viewer**: Students can browse all 5 shareholder letters using an embedded PDF reader.
- **Smart Ingestion**: Clicking "Store in Vector DB" chunks the 5 letters, applies embeddings, adds metadata (year/source), and saves them to a local LanceDB instance instantly. *A progress bar visually tracks the chunking and embedding phases.*
- **Dynamic Queries**: The previous static queries have been replaced with a **free-text input**. Students can type in whatever they want to ask Warren Buffett, or select from an auto-complete list of suggestions (like "impact of COVID-19"). Results are instantly returned, showcasing semantic match rather than simple keyword matches.


### The "Aha!" Moment
Students will experience the magic of dragging and dropping raw PDFs, tracking the vectorize process with a progress bar, and freely querying the data to instantly retrieve intelligent answers with exact source citations—all inside an easy-to-use Dockerized UI.
