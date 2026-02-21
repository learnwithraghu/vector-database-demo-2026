# Demo 1: Setup and Connection (The Foundation)

## 🎯 Setup: Where the Journey Begins
Before we can build "Brains" or "Search Engines", we need a place to store our memories. In this module, we will stand up our **Vector Database (Qdrant)** and establish our first communication line with it.

*   **Previous Step**: (None - this is Day 1)
*   **Next Step**: Sending actual data.

## 🛠️ Pre-flight Check (Do this every time!)
Before running any code, the instructor must ensure the "Engine" is running.

```bash
# 1. Check if Qdrant is running
docker ps 
# (Look for 'qdrant/qdrant' on port 6333)

# 2. Setup Python Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install this lesson's requirements
pip install qdrant-client
```

## 📝 Steps for the Instructor

### 1. The Concept: "The Brain in a Jar"
Explain: 
*   **The Server (Docker)** is like a "Brain in a Jar" sitting on a shelf. It has the **capacity** to remember things, but it has no eyes or ears. It's just waiting.
*   **The Client (Python)** is like the **Scientist** connecting wires to that brain.
*   Today, we are simply plugging in the wires to make sure the light turns green. We aren't teaching it anything yet; we are just verifying it's alive.

### 2. Launch Qdrant (The Server)

Run this command in your terminal:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
*   **-p 6333:6333**: Exposes the HTTP API (and the Dashboard!).
*   **-p 6334:6334**: Exposes the gRPC API (faster).
*   **-v ...**: Persists data so it isn't lost when you stop the container.

### 2. Verify with Dashboard
Open your browser to `http://localhost:6333/dashboard`.
*Show them the empty dashboard. This is a great visual aid to prove the server is running.*

### 3. Python Client Setup
Create a virtual environment and install the client:
```bash
python3 -m venv venv
source venv/bin/activate
pip install qdrant-client
```

### 4. Code Walkthrough: `connect.py`

This script is very simple. It verifies that your Python environment can talk to the Qdrant server running in Docker.

### Phase 1: Connection (Line 5)
*   `client = QdrantClient(url="http://localhost:6333")`
*   This creates the object we use to talk to the database.
*   **`localhost:6333`**: This is the address of the Qdrant API. If this line doesn't crash, it means the Python library is installed correctly, but it doesn't verify the server exists yet.

### Phase 2: Verification (Line 10)
*   `client.get_collections()`
*   This actually sends a request 📡 to the server.
*   If the Docker container is running, it will reply with a list of active collections.
*   If the Docker container is **Stopped**, this line will crash with a connection error.

### 5. Run the script
```bash
python connect.py
```
**Expected Output:**
> Connection established!
> Collections: collections=[]

## 💡 Key Takeaway
"We now have a running Vector Brain 🧠 ready to store our data!"
