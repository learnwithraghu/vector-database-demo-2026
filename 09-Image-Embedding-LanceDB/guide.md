# 📖 Instructor/Student Guide: Image Search with LanceDB

This guide outlines the **3 core AI steps** to build an Image Similarity Search system using **CLIP** and **LanceDB**.

### 🌟 Key Concepts
1.  **CLIP**: A multi-modal model that understands both images and text in the same space.
2.  **LanceDB**: A fast, serverless vector database that pairs "meaning" (vectors) with "source" (image paths).

---

### 🛠️ Step 1: Load the CLIP Model
We use `sentence-transformers` because it provides a simple one-line way to load the CLIP visual model.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('clip-ViT-B-32')
```

### 🛠️ Step 2: Generate Vectors (Embeddings)
We convert each image file in the `images/` folder into a 512-dimension vector.
- The model ignores the file name and only processes the visual pixels.
- Similar visual content (e.g. two different cats) will result in vectors that are mathematically "close."

```python
# Assuming 'images' is a list of opened PIL images
vectors = model.encode(images)
```

### 🛠️ Step 3: Semantic Search
We search by embedding a **text query** into a vector and finding the closest image vector in our database.
- This is call **multi-modal search** (searching across different data types).
- We use `table.search(vector)` in LanceDB to find the nearest neighbors.

```python
query_vector = model.encode(["a photo of a cat"])[0]
results = table.search(query_vector).limit(1).to_pandas()
```

---

### Tips for Success
- **Pairing**: Always remember to pair your vectors with the original image path! LanceDB handles this using the `data` list of dictionaries.
- **Visualisation**: Use `matplotlib` and `PIL` to show the student that the search actually found the correct photo.
