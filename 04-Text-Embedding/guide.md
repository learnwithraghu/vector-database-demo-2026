## Lab Guide — Text Embeddings with LanceDB (KodeKloud Airlines)

### 1) What we will do in this lab
In this lab we will:
- Load a KodeKloud Airlines policy document
- Split it into chunks
- Generate **text embeddings** for each chunk (text → numbers)
- Store embeddings in **LanceDB**
- Ask questions and retrieve the most relevant policy chunks using **vector search**

**How to run a Jupyter cell**
- Click a cell and press **Shift + Enter** to run it (or **Cmd/Ctrl + Enter** to run without moving).
- Run cells **top-to-bottom** because later cells depend on earlier variables.

You will use: `text_embedding_lancedb_student.ipynb`

---

### 2) Fill-in #1 — Import LanceDB
**Why this matters**: Without importing LanceDB, you can’t create the database, table, or run vector search.

**What to write**: In the imports cell, replace the placeholder with an import statement for LanceDB.

**Solution (copy/paste)**:

```python
import lancedb
```

---

### MCQ 1 (Embeddings)
**Question**: What is the main reason we use embeddings for semantic search?

A. They store the original text more efficiently than strings  
B. They allow similarity comparison based on meaning, not exact words  
C. They encrypt text so only the database can read it  
D. They remove the need for a database

**Correct answer**: B

---

### 3) Fill-in #2 — Point to the policy document
**Why this matters**: The notebook needs the source text to embed and store.

**What to write**: Set `POLICY_PATH` to the markdown file in the same folder.

**Solution (copy/paste)**:

```python
BASE_DIR / "kodekloud_airlines_policy.md"
```

---

### 4) Fill-in #3 — Choose the embedding model
**Why this matters**: The model defines how text becomes vectors (dimension + semantics).

**What to write**: Set `MODEL_NAME` to the model used in the instructor notebook.

**Solution (copy/paste)**:

```python
"all-MiniLM-L6-v2"
```

---

### MCQ 2 (Vector search)
**Question**: Why do we chunk the policy document before embedding?

A. Because embeddings only work on markdown files  
B. To reduce storage size to zero  
C. To improve retrieval by matching smaller, focused pieces of text  
D. Because LanceDB cannot store long strings

**Correct answer**: C

---

### 5) Fill-in #4 — Embed the question for searching
**Why this matters**: Vector search compares **question embedding** to **chunk embeddings**.

**What to write**: Create `qvec` by embedding the question and normalizing, then convert to a Python list (LanceDB accepts list vectors).

**Solution (copy/paste)**:

```python
model.encode(question, normalize_embeddings=True).tolist()
```

---

### Notes for the lab designer
- The student notebook intentionally contains 4 placeholders marked as `<fill_in the line here>` so learners must type before running.
- Both instructor and student notebooks are included in the Docker build because the Dockerfile copies the entire `04-Text-Embedding/` directory into the image.

