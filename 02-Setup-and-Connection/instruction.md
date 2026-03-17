# Lab: Setup and Connection to Qdrant

This lab walks you through setting up Qdrant (a vector database) and completing `connect_lab.py` to establish a connection from Python.

---

## Prerequisites

- Docker installed and running
- Python 3.8+

---

## Step 1: Start Qdrant Server

In a terminal, run:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

- **6333**: HTTP API and Dashboard
- **6334**: gRPC API
- **-v**: Persists data when the container stops

Leave this terminal running.

---

## Step 2: Verify Qdrant is Running

1. Open `http://localhost:6333/dashboard` in your browser.
2. You should see the Qdrant dashboard (empty collections).

---

## Step 3: Set Up Python Environment

In a **new** terminal:

```bash
cd 02-Setup-and-Connection
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install qdrant-client
```

---

## Step 4: Complete `connect_lab.py`

Open `connect_lab.py` and fill in the gaps marked with **FILL THE LINE** and **FILL THE WORD**.

### Gap 1: **FILL THE LINE** (top of file)

Add the import for the Qdrant client.

<details>
<summary>Answer</summary>

```python
from qdrant_client import QdrantClient
```
</details>

---

### Gap 2: **FILL THE WORD** — `FILL_URL`

Set the URL for the local Qdrant instance.

<details>
<summary>Answer</summary>

```python
client = QdrantClient(url="http://localhost:6333")
```

Replace `FILL_URL` with `"http://localhost:6333"`.
</details>

---

### Gap 3: **FILL THE WORD** — `FILL_METHOD`

Call the method that retrieves all collections from the server.

<details>
<summary>Answer</summary>

```python
collections = client.get_collections()
```

Replace `FILL_METHOD` with `get_collections`.
</details>

---

## Step 5: Run the Script

```bash
python connect_lab.py
```

**Expected output:**
```
✅ Connection established successfully!
📚 Current Collections: collections=[]
```

---

## Multiple Choice Questions

### MCQ 1: What does the Qdrant server need to be running for?

**A)** To compile Python code  
**B)** To store and serve vector embeddings; the Python client connects to it over the network  
**C)** To install Python packages  
**D)** To run the Jupyter notebook

<details>
<summary>Answer</summary>

**B)** To store and serve vector embeddings; the Python client connects to it over the network.

Qdrant is a vector database server. The Python client sends requests to it over HTTP/gRPC. If the server is not running, connection calls will fail.
</details>

---

### MCQ 2: What does `client.get_collections()` do?

**A)** Creates a new collection in the database  
**B)** Sends a request to the server and returns the list of existing collections  
**C)** Deletes all collections  
**D)** Connects the client to the server

<details>
<summary>Answer</summary>

**B)** Sends a request to the server and returns the list of existing collections.

`get_collections()` performs a network call to the Qdrant server and returns metadata about all collections. It is often used to verify the connection is working.
</details>
