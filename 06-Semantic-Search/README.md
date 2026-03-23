# Fruits Vector Demo (Stages 1 + 2)

This folder now hosts two notebooks that show how vector databases behave with a small fruit dataset. Stage 1 (`01-fruits-vector-setup.ipynb`) builds a vector store with handcrafted embeddings and finds the closest match to a mango-like query. Stage 2 (`02-fruits-similarity-metrics.ipynb`) reuses the same fruit vectors to compare cosine similarity, Euclidean distance, and dot product rankings.

## Docker workflow

The image installs dependencies from `requirements.txt`, copies both notebooks, and launches Jupyter automatically.

### Fresh start

```bash
cd 06-Semantic-Search
# clean up dangling images and volumes before building
docker system prune -af
docker volume prune -f

docker build -t fruits-vector-demo .
docker run -p 8888:8888 fruits-vector-demo
```

Access JupyterLab via the URL printed in the container logs (no authentication required), then open `01-fruits-vector-setup.ipynb` followed by `02-fruits-similarity-metrics.ipynb`.

The notebooks already live in `/workspace`, so they are ready as soon as the server starts.
