# Demo 4: Metadata Filtering (Precision)

## 🎯 Level Up: The Best of Both Worlds
In Demo 3, we successfully searched by *meaning*. But what if the user wants "Sci-Fi movies" (Meaning) that are also "Released after year 2000" (Exact Fact)?
In this module, we combine the power of **Vector Search** with standard **SQL-like filtering**.

*   **Previous Step**: Building a Semantic Search Engine.
*   **Next Step**: Building a Chatbot (RAG).

## 🛠️ Pre-flight Check
```bash
# 1. Check if Qdrant is running
docker ps 

# 2. Setup Python Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install qdrant-client sentence-transformers
```

## 📝 Steps for the Instructor

### 1. The Analogy: "The Venn Diagram"
*   **Circle A (The Vibe)**: "Movies about Robots" (Semantic Search). This includes Terminator, Wall-E, and Iron Man.
*   **Circle B (The Hard Fact)**: "Movies released after 2005" (Metadata Filter).
*   **The Intersection**: We only want the movies where **BOTH** are true.
*   Wall-E (2008) ✅
*   Terminator (1984) ❌ (Right Vibe, Wrong Year)
*   Qdrant is fast because it efficiently finds this intersection.

### 2. The Solution: Payloads & Filters
We will add `year` and `genre` to our movies.
*   **Filter**: `must` (AND), `should` (OR), `must_not` (NOT).

### 3. Code Walkthrough
### 3. Code Walkthrough
1.  **Search 1 (Vectors Only)**: Search for **"Future robots"**.
    *   *Result*: Finds **"The Terminator"** (1984) and **"Wall-E"** (2008).
    *   *Problem*: Terminator is from 1984. The vector model found "robots" but ignored the *implied* time constraint because it's just looking for *similar meaning*.
    *   *Question*: "Why didn't we just search for 'Future robots released after 2000'?"
    *   *Answer*: The model might think "released after 2000" is just more keywords. It might match a movie about time travel to the year 3000, even if the movie was made in 1990! **Vectors catch vibes; Filters catch facts.**
2.  **Search 2 (Vectors + Filter)**: Search for **"Future robots"** AND Filter `year > 2000`.
    *   *Result*: **Terminator disappears**. Only **Wall-E** remains.
    *   *Success*: High Precision!

## 💻 Code Walkthrough: `filtering.py`

### Phase 1: Setup & Data (Lines 20-33)
*   We load `movies.json`.
*   **Crucial Step**: `payload=doc`.
    *   We are NOT just storing the vector (the numbers).
    *   We are storing the **Metadata** (`year`, `genre`) *inside* the vector database.
    *   *Why?* Qdrant builds a separate "Payload Index" (like a traditional SQL index) for these fields. This allows it to ignore millions of irrelevant vectors *instantly* before it even starts the semantic search.

### Phase 2: The Filter Definition (Lines 44-49)
This is where we build our "SQL WHERE clause" equivalent.
*   **Line 44**: `Filter(must=[...])`
    *   `must`: Means **AND**. All conditions inside the list must be true.
    *   (You can also use `should` for OR, and `must_not` for NOT).
*   **Line 46**: `FieldCondition(key="genre", match=MatchValue(value="Sci-Fi"))`
    *   Translation: `genre == "Sci-Fi"`
*   **Line 47**: `FieldCondition(key="year", range=Range(gte=2000))`
    *   Translation: `year >= 2000`

### Phase 3: The Filtered Search (Lines 52-56)
*   **Line 55**: `query_filter=my_filter`
    *   We pass the filter object to the search command.
    *   **How it works**: Qdrant *first* narrows down the list of candidates to only post-2000 Sci-Fi movies. *Then* it calculates the distance score for those few movies. This is extremely fast even with billions of vectors.
