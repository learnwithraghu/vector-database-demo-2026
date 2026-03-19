# 02 - Setup and Connection (Step-by-Step, No Docker for Notebook)

This guide shows a simple, classroom-friendly flow:

1. Create a Python virtual environment
2. Start **Qdrant** in Docker (vector DB only)
3. Open the **Qdrant UI** in the browser
4. Run a Python script to **connect** (`connect.py`)
5. Run a Python script to **insert dummy vectors** (`insert_dummy_vectors.py`)

---

## 1. Create and activate virtual environment

In a terminal:

```bash
cd 02-Setup-and-Connection
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install qdrant-client
```

You now have an isolated environment just for this lab.

---

## 2. Start Qdrant (vector database) in Docker

In the **same terminal or another one** (your choice), start Qdrant:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

- Port **6333**: HTTP API + Dashboard
- Port **6334**: gRPC API
- The volume keeps data on disk between container restarts.

Leave this Docker command running while you do the rest of the lab.

---

## 3. Open the Qdrant UI

In your browser, open:

- `http://localhost:6333/dashboard`

You should see the Qdrant web UI with **no collections yet**.

---

## 4. Run the connection script

Back in the terminal with the virtual environment activated:

```bash
cd 02-Setup-and-Connection   # if not already there
python connect.py
```

What this does:

- Uses `QdrantClient(url="http://localhost:6333")`
- Calls `get_collections()` to verify the server is reachable
- Prints the list of collections (likely `[]` the first time)

If this succeeds, you have a **working client–server connection**.

---

## 5. Insert some dummy vectors

Next, run the insert script:

```bash
python insert_dummy_vectors.py
```

What this script does:

- Connects to `http://localhost:6333`
- Creates a small demo collection called `demo_vectors`
- Inserts 3 dummy vectors with simple payloads
- Performs a quick scroll to print a few records back

You should see console output confirming the collection and a few records.

---

## 6. Confirm in the UI

Go back to the Qdrant dashboard in your browser and **refresh**.

- You should now see the `demo_vectors` collection
- Clicking into it should show 3 points with the payloads from the script

This completes the story:

1. **Setup**: venv + Docker Qdrant
2. **Open UI**: `localhost:6333/dashboard`
3. **Connect**: `connect.py`
4. **Insert records**: `insert_dummy_vectors.py`

