# Demo 6: Admin & Operations (The "Pro" Level)

## 🎯 Level Up: From Developer to Architect
Congratulations! You have built a full vector application (Demos 1-5).
But a real engineer knows that **building** is only half the battle. Now we must **operate** it. This lesson covers how to keep your vector database healthy, upgrade it without downtime, and back it up.

*   **Previous Step**: The Chatbot.
*   **Next Step**: Course Completion!

## 🛠️ Pre-flight Check
```bash
# 1. Check if Qdrant is running
docker ps 

# 2. Setup Python Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install qdrant-client
```

## 📝 Steps for the Instructor

### 1. The Analogy: "The F1 Pit Crew"
*   **Developers** are the drivers. They want to go fast.
*   **Admins (Ops)** are the Pit Crew.
    *   **Observability**: Checking the engine temperature (Collection Status).
    *   **Snapshots**: Having a spare car ready if this one crashes (Backups).
    *   **Aliases**: Changing tires while the car is still moving (Zero-downtime updates).
*   Without the Pit Crew, the Driver crashes.

### 2. Observability (Collection Info)
Explain: "You don't just dump data and hope for the best. You need to verify."
*   We'll learn to check `points_count`, `vector_size`, and status (Green/Red).

### 2. Zero-Downtime Deployment (Aliases)
**The Problem**: You have a collection `movies_v1`. You want to release `movies_v2` with better embeddings. You don't want to change your backend code url every time.
**The Solution**: **Aliases**.
*   You point the alias `processing_db` -> `movies_v1`.
*   Your app only talks to `processing_db`.
*   When `v2` is ready, you simple "switch" the alias. The app never notices.

### 3. Safety (Snapshots)
Data is valuable. One line of code can back up your entire vector index.

## 💻 Code Walkthrough: `admin_ops.py`

### Phase 1: Setup & Data Generation (Lines 5-21)
We create a temporary collection `my_app_v1` and fill it with dummy data just so we have something to manage.
*   **Line 15**: `vectors_config` is standard setup.

### Phase 2: Inspection (Lines 24-29)
*   **`client.get_collection(collection_name)`**
*   This returns a rich object containing:
    *   `points_count`: How many vectors?
    *   `status`: Is it optimizing? Is it ready? (Green)
    *   `vectors_count`: Total number of vectors.

### Phase 3: The Alias Switch (Lines 35-49)
This is the "Hero" move. 🦸
*   **Step A**: We create an alias `production_alias` pointing to `my_app_v1`.
*   **Step B**: We verify that searching `production_alias` actually searches `my_app_v1`.
*   *Teacher Note*: Explain that in real life, you would build `my_app_v2` in the background, and then run an update command to switch the pointer.

### Phase 4: Backups (Lines 52-54)
*   **`client.create_snapshot(...)`**
*   This triggers the server to save the entire index to disk.
*   It returns a URL or path where the snapshot is stored.
