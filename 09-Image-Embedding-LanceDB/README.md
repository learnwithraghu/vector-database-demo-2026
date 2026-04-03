# 09 — Image Similarity Search with LanceDB

Learn how to build a **text-to-image** and **image-to-image** search system using **CLIP** (Contrastive Language-Image Pre-training) and **LanceDB**.

### 🛠️ Prerequisites
- Docker (optional)
- Python 3.11+
- Some images in the `images/` folder (animal photos)

### 🚀 Running the Notebook
1.  **With Docker**:
    - Build: `docker build -t image-embedding-demo .`
    - Run: `docker run -p 8888:8888 image-embedding-demo`
2.  **Locally**:
    - Install dependencies: `pip install -r requirements.txt`
    - Start Jupyter: `jupyter lab --allow-root` (and follow the token link)

### 📖 Folder Contents
- `image_embedding_lancedb.ipynb`: Complete demo code.
- `image_embedding_lancedb_student.ipynb`: Blank version for practice.
- `images/`: Put your animal photos here!
