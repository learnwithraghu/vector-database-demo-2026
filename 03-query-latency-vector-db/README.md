# Query Latency Demo (Two-Notebook Flow)

This folder is organized for a fast demo flow:

1. **Build embeddings in one click** with `01_build_embeddings_one_cell.ipynb`
2. **Explain latency and throughput** with `02_query_latency_explained.ipynb`

## Required Input File

The embedding notebook reads this exact file:

- `pdf_doc/NFL_2025.pdf`

If it is missing, create the `pdf_doc` folder and place the PDF there before running notebook 1.

## Notebook Roles

- `01_build_embeddings_one_cell.ipynb`
  - One code cell
  - Uses helper functions (with inline explanation) for readability
  - Loads PDF -> chunks text -> creates embeddings -> builds ZVec collection
- `02_query_latency_explained.ipynb`
  - Short, focused code cells
  - Text-box style explanations for concepts
  - Measures and compares p50/p95/p99 latency and throughput

## Why No Docker Compose

`docker compose` is not required here because this demo runs one container only (Jupyter runtime).

## Run With Docker (Recommended)

From `03-query-latency-vector-db`:

```bash
docker system prune -a --volumes -f
docker build -t query-latency-notebook .
docker run --rm -it -p 8888:8888 -v "${PWD}:/workspace" query-latency-notebook
```

Open `http://localhost:8888`, then run notebooks in this order:

1. `01_build_embeddings_one_cell.ipynb`
2. `02_query_latency_explained.ipynb`

Both notebooks are included in the same Docker image built from this folder.

The bind mount (`-v "${PWD}:/workspace"`) keeps notebook edits and generated outputs on your host machine.

## Local Run (Optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Troubleshooting

- Port conflict:
  - `docker run --rm -it -p 8889:8888 -v "${PWD}:/workspace" query-latency-notebook`
- Clean rebuild:
  - `docker build --no-cache -t query-latency-notebook .`
