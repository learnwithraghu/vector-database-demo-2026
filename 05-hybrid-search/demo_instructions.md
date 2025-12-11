# Demo 5: Hybrid Search (The "Pro" Search)

## 🎯 Level Up: Keyword + Meaning
We have done **Semantic Search** (Demo 3), which finds "Meaning" (Vectors).
But sometimes, you need to find an **Exact Word** (Keyword), like a specific product code "XJ-900" or a rare name "Zylorphian". Semantic search might miss these if the "vibes" are off.

**Hybrid Search** combines:
1.  **Dense Vectors** (Understanding): "Find me a phone."
2.  **Sparse Vectors** (Keywords): "Find me 'iPhone 15'."

*   **Previous Step**: Metadata Filtering.
*   **Next Step**: Admin Operations (Production).

## 🛠️ Pre-flight Check
```bash
# 1. Check if Qdrant is running
docker ps 

# 2. Setup Python Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
# We need 'fastembed' to create Sparse Vectors easily
pip install fastembed
```

## 📝 Steps for the Instructor

### 1. The Analogy: "Sketch Artist vs. Fingerprint Scanner"
*   **Dense Vectors (Semantic)** are like a **Police Sketch Artist**.
    *   Witness says: *"The suspect looked like a pirate, tall, scary."*
    *   The sketch isn't perfect, but it captures the **essence**. It matches anyone who looks roughly like a pirate.
*   **Sparse Vectors (Keyword)** are like a **Fingerprint Scanner**.
    *   It doesn't care if the suspect looks like a pirate. It only cares: *"Does this exact pattern match?"*
*   **Hybrid Search**: We use BOTH. We want someone who looks like a pirate (Context) AND has this specific fingerprint (Exact Match).

### 2. Code Walkthrough (`hybrid_search.py`)

## 💻 Code Walkthrough: `hybrid_search.py`

### Phase 1: Setup Hybrid Collection (Lines 6-17)
*   We enable **Two** vector components on the same collection:
    *   `"text-dense"`: The standard semantic vector (384 numbers).
    *   `"text-sparse"`: A special Keyword index.

### Phase 2: Loading Data (Lines 25-40)
*   We load `movies.json` again.
*   **Magic**: We pass `documents=docs_text`. Qdrant + FastEmbed automatically creates:
    *   **Dense Vectors**: For the meanings.
    *   **Sparse Vectors**: For every unique word (like 'XJ-900').

### Phase 3: The Search (Lines 43-70)
*   **Query**: "Project XJ-900"
*   **A. Dense Only**: Fails. The model doesn't know what "XJ-900" means. It just sees "Project" and guesses random things.
*   **B. Hybrid Search**: Success!
    *   It finds the exact documentary.
    *   *Why?* Because the **Sparse Vector** for "XJ-900" had a perfect match, boosting the score to the top.
