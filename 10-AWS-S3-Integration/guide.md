## Lab Guide - AWS S3 Vectors Integration (Student)

### 1) What we will do in this lab
In this lab we will:
- Connect to AWS S3 Vectors using `boto3`
- Validate vector bucket and index access
- Load and chunk a local policy document
- Generate text embeddings using Sentence Transformers
- Upload vectors to S3 Vectors
- Run semantic search queries and inspect results

**How to run a Jupyter cell**
- Click a cell and press **Shift + Enter** to run it (or **Cmd/Ctrl + Enter** to run without moving).
- Run cells **top-to-bottom** because later cells depend on earlier variables.

You will use: `aws_s3_integration_student.ipynb`

---

### 2) Fill-in #1 - Import required libraries
**Why this matters**: You need AWS SDK and embedding model classes before any connection or vector generation can happen.

**What to write**: Import both `boto3` and `SentenceTransformer`.

**Solution (copy/paste)**:

```python
import boto3
from sentence_transformers import SentenceTransformer
```

---

### MCQ 1 (S3 Vectors)
**Question**: What is the purpose of `list_vector_buckets()` in this workflow?

A. It uploads embeddings to the configured bucket  
B. It checks which vector buckets your current IAM identity can access  
C. It creates a new vector bucket automatically  
D. It converts text paragraphs into vector embeddings

**Correct answer**: B

---

### 3) Fill-in #2 - Set AWS access key id
**Why this matters**: The session uses your access key to authenticate API calls.

**What to write**: Set `AWS_ACCESS_KEY_ID` as a quoted string.

**Solution (copy/paste)**:

```python
"REPLACE_WITH_YOUR_ACCESS_KEY_ID"
```

---

### 4) Fill-in #3 - Set AWS secret access key
**Why this matters**: The secret key pairs with the access key and signs AWS requests.

**What to write**: Set `AWS_SECRET_ACCESS_KEY` as a quoted string.

**Solution (copy/paste)**:

```python
"REPLACE_WITH_YOUR_SECRET_ACCESS_KEY"
```

---

### MCQ 2 (Embedding dimensions)
**Question**: Why do we compare `embedding_dim` with the index dimension from `get_index()`?

A. To ensure query speed is always under 10ms  
B. Because vector indexes require exactly matching dimensions for stored/query vectors  
C. To reduce AWS billing by compressing vectors  
D. To convert float32 vectors into strings

**Correct answer**: B

---

### 5) Fill-in #4 - Choose the embedding model
**Why this matters**: The model defines vector dimension and semantic quality.

**What to write**: Set `MODEL_NAME` to the same value used in the instructor notebook.

**Solution (copy/paste)**:

```python
"sentence-transformers/all-MiniLM-L6-v2"
```

---

### 6) Fill-in #5 - Point to the source document
**Why this matters**: The notebook must read source text before chunking and embedding.

**What to write**: Set `policy_path` to `airline_security_policy.txt` in the current lab folder.

**Solution (copy/paste)**:

```python
Path("airline_security_policy.txt")
```

---

### MCQ 3 (Semantic search)
**Question**: In vector search results, what does a smaller `distance` generally indicate?

A. Less relevant result  
B. More relevant semantic match  
C. Larger original paragraph size  
D. Older upload timestamp

**Correct answer**: B

---

### 7) Fill-in #6 - Embed the query text
**Why this matters**: Query vectors must be produced by the same model as stored document vectors.

**What to write**: Encode `query_text`, take the first vector, and convert it to a list.

**Solution (copy/paste)**:

```python
model.encode([query_text])[0].tolist()
```

---

### Notes for the lab designer
- The student notebook intentionally contains 6 placeholders marked as `<fill_in the line here>` so learners actively complete key setup and search steps.
- This student flow combines both connectivity validation and embedding upload/query in one notebook for a single end-to-end exercise.
