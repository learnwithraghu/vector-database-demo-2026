# Lab: VectorDB Quickstart — Warren Buffett Letters

This lab runs an interactive Streamlit app that explores vector databases using Warren Buffett's shareholder letters (2020–2024). You will load PDFs, store them in LanceDB, and run semantic queries.

---

## Prerequisites

- Docker installed and running  
  *(or Python 3.10+ for local setup)*

---

## Step 1: Navigate to the Project Folder

```bash
cd 00-Quickstart
```

---

## Step 2: Build the Docker Image

```bash
docker build -t lancedb-quickstart .
```

This builds an image that includes:
- Python dependencies (LanceDB, Streamlit, sentence-transformers)
- The Streamlit app and shareholder letters
- Pre-loaded embeddings (done during build)

---

## Step 3: Start the Container

```bash
docker run -p 8501:8501 lancedb-quickstart
```

- **8501** is the Streamlit port.
- Leave this terminal running.

---

### MCQ 1: Why does the Docker build run `preload_db.py`?

**A)** To start the Streamlit server  
**B)** To chunk the PDFs, generate embeddings, and pre-populate LanceDB so the app loads quickly  
**C)** To download the shareholder letters from the internet  
**D)** To install Python packages

<details>
<summary>Answer</summary>

**B)** To chunk the PDFs, generate embeddings, and pre-populate LanceDB so the app loads quickly.

Embedding generation is slow. Running it during the Docker build means the database is ready when the container starts, so the app responds quickly when you click "Store in Vector DB".
</details>

---

## Step 4: Open the App in Your Browser

Go to: **http://localhost:8501**

You should see the VectorDB Quickstart app with Warren Buffett letters (2020–2024).

---

## Step 5: Use the App

1. **Document Viewer** — Browse the shareholder letters.
2. **Store in Vector DB** — Click to confirm the pre-loaded data is ready (embeddings are already in the image).
3. **Dynamic Queries** — Type a question (e.g. "impact of COVID-19") or pick a suggestion. Results show semantically similar passages with source citations.

---

### MCQ 2: What makes the query results "semantic" rather than keyword-based?

**A)** The app uses regex to find exact word matches  
**B)** Text is converted to vector embeddings; similar meaning maps to similar vectors, so the search finds conceptually related passages  
**C)** The app searches only in the document titles  
**D)** LanceDB stores raw text and matches by character count

<details>
<summary>Answer</summary>

**B)** Text is converted to vector embeddings; similar meaning maps to similar vectors, so the search finds conceptually related passages.

Embeddings capture meaning. Queries like "COVID impact" can match passages about "pandemic" or "virus" even if those exact words are not used, because the vectors are close in embedding space.
</details>

