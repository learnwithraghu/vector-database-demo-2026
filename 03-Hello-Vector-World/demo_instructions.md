# Demo 2: Hello Vector World (The "RGB" Concept)

## 🎯 Level Up: Manual Coordinates
**Congratulations!** You have a running database from Demo 1.
Now, we need to understand *what* exactly we are storing. Before we let AI do the work, we will manually insert "Points" into the database to understand the relationship between **Vectors** and **Payloads**.

## 🛠️ Pre-flight Check
You need Qdrant running and the python client installed.
```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Install Client
pip install qdrant-client
```

## 📝 The Concept: RGB Colors
An "embedding" is just a list of numbers. The best real-world example of this is **Digital Color**.
Every color on your screen is a **Vector of 3 numbers**: `[Red, Green, Blue]`.

*   **Red**: `[255, 0, 0]` (Mostly Red)
*   **Green**: `[0, 255, 0]` (Mostly Green)
*   **Yellow**: `[255, 255, 0]` (Mix of Red and Green)

A **Vector Database** calculates which colors are "similar" by measuring the distance between these numbers.

### 1. Code Walkthrough (`hello_vector.py`)

**Step A: Define Dimensions**
We create a collection for vectors of **Size 3**.
Dimensions: `[Red, Green, Blue]` (Scaled 0.0 to 1.0)

**Step B: Insert Points**
We insert 4 colors into the database:
1.  **Red**: `[1.0, 0.05, 0.05]`
2.  **Green**: `[0.05, 1.0, 0.05]`
3.  **Blue**: `[0.05, 0.05, 1.0]`
4.  **Yellow**: `[1.0, 1.0, 0.05]`

**Step C: Search**
We ask the database: *"I have a color mixed like this `[1.0, 0.2, 0.0]`. What is it?"*
*   This input is mostly Red, with a tiny bit of Green.
*   **Prediction**: It should be closest to **Red**. It should be somewhat close to **Yellow** (since Yellow has Red in it). It should be far from **Blue**.

### 2. Run the script & Check Dashboard
Run the script:
```bash
python3 hello_vector.py
```

Check the results in the terminal.
Then, go to `http://localhost:6333/dashboard`:
1.  Click **rgb_colors**.
2.  Click **Points**.
3.  You will see your colors stored as vectors!

## 🧠 Why this matters?
This is exactly how AI works.
*   Instead of `[Red, Green, Blue]`, AI models have thousands of dimensions like `[Is_Animal, Is_Happy, Is_Past_Tense, ...]`.
*   "King" and "Queen" are close in the vector space, just like "Red" and "Dark Red" are close.
