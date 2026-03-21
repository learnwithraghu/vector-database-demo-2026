# Image embedding with ChromaDB

This folder demonstrates **embedding a PNG with a CLIP model**, **inspecting** the vector, and **storing** it in **ChromaDB**. The notebook focuses on the pipeline (pixels → embedding → database), not on search queries.

## Before you run

The notebook defaults to `images/landscape.png` (privacy-computing industry landscape). To use another PNG, place it under `images/` and set `IMAGE_PATH` in the notebook.

## Notebook

- **`image_embedding_chroma.ipynb`** – Load the image, compute and inspect the CLIP embedding, upsert into ChromaDB, and a short “real world” wrap-up.

## Run with Docker (recommended)

### Build the image

```bash
cd 13-image-embedding
docker build -t chroma-image-embedding-demo .
```

### Run the container

If your PNG is already in `./images` on the host, mount it so the notebook can read it:

```bash
docker run --rm -p 8888:8888 -v "$(pwd)/images:/workspace/images" chroma-image-embedding-demo
```

### Open Jupyter

Copy the URL from the container logs (includes a token), open it in a browser, then run `image_embedding_chroma.ipynb`.

---

## Purge Docker and start clean

These commands affect **all** containers and images on your machine, not only this demo. Use a terminal where you are sure you want a full reset.

### Stop and remove every container

```bash
docker ps -q | xargs -r docker stop
docker ps -aq | xargs -r docker rm
```

On macOS, if `xargs -r` is unavailable, use:

```bash
docker stop $(docker ps -q) 2>/dev/null || true
docker rm $(docker ps -aq) 2>/dev/null || true
```

### Remove all images (and optionally build cache)

```bash
docker images -q | xargs -r docker rmi -f
```

macOS fallback:

```bash
docker rmi -f $(docker images -q) 2>/dev/null || true
```

### Remove unused networks, volumes, and build cache

```bash
docker system prune -af --volumes
```

### Rebuild and run this demo only

After the purge (or after removing just this image with `docker rmi chroma-image-embedding-demo`):

```bash
cd 13-image-embedding
docker build -t chroma-image-embedding-demo .
docker run --rm -p 8888:8888 -v "$(pwd)/images:/workspace/images" chroma-image-embedding-demo
```

---

## Local setup (without Docker)

```bash
cd 13-image-embedding
chmod +x setup.sh
./setup.sh
```

Then open the notebook and choose the **Python (image_embedding_demo)** kernel.

First run may take several minutes while PyTorch and CLIP weights download.

## Requirements

- `chromadb`, `sentence-transformers`, `pillow`, `jupyter`, `ipykernel`

Local Chroma data is written to `chroma_image_data/` when you use the default paths in the notebook.
