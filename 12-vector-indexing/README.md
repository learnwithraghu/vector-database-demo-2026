# Vector Indexing Notebooks

This folder contains notebooks to teach vector indexing concepts.

## Simple notebook for students

Start with:

- `00_simple_indexing_demo.ipynb`

This notebook is intentionally short and beginner-friendly. It shows:

- vectors stored in ChromaDB
- search before indexing (linear scan)
- search after a simple index-like grouping
- why indexing reduces distance checks

## Other notebooks

- `01_indexing.ipynb` (advanced indexing concepts)
- `02_product_quantization.ipynb` (advanced quantization concepts)

## Run with Docker (single image)

Run these commands from inside this folder:

```bash
cd 12-vector-indexing
```

### 1) Clean old container/image first

```bash
docker rm -f vector-indexing-demo 2>/dev/null || true
docker rmi vector-indexing-demo 2>/dev/null || true
```

### 2) Build image

```bash
docker build -t vector-indexing-demo .
```

### 3) Run Jupyter

```bash
docker run --rm -p 8888:8888 --name vector-indexing-demo vector-indexing-demo
```

Then open [http://localhost:8888](http://localhost:8888) and run `00_simple_indexing_demo.ipynb`.

## Cleanup

If you used `--rm`, the container is automatically removed when stopped.

Optional manual cleanup:

```bash
docker stop vector-indexing-demo 2>/dev/null || true
docker rm vector-indexing-demo 2>/dev/null || true
docker rmi vector-indexing-demo
```

## Local setup without Docker (optional)

```bash
cd 12-vector-indexing
bash setup.sh
```
