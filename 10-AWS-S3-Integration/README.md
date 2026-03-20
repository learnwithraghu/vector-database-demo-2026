# AWS S3 Integration: Notebook-First Demo

This module demonstrates AWS S3 Vectors using two runnable notebooks:

- `connect_test_s3_vector.ipynb` for connection and index visibility checks
- `load_and_query_embedding.ipynb` for document embedding, upload, and semantic search

## Files

- `airline_security_policy.txt`: source text document
- `requirements.txt`: Python dependencies for this demo
- `connect_test_s3_vector.ipynb`: connection and validation notebook
- `load_and_query_embedding.ipynb`: embedding and query notebook
- `Dockerfile`: containerized notebook runtime
- `.dockerignore`: keeps image builds clean and fast

## Prerequisites

- Python 3.9+
- AWS credentials with `s3vectors` permissions
- A vector bucket and index already created in AWS

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run With Docker (Recommended)

From `10-AWS-S3-Integration`, build and run:

```bash
docker system prune -a --volumes -f
docker build -t aws-s3-vectors-notebooks .
docker run --rm -it -p 8888:8888 aws-s3-vectors-notebooks
```

Open `http://localhost:8888` in your browser, then run notebook cells one by one.

## Notebook 1: Connect and Test

Open `connect_test_s3_vector.ipynb` and run top to bottom.

What it does:

- creates a `boto3` `s3vectors` client
- lists vector buckets visible to your IAM principal
- checks whether your configured bucket exists
- lists indexes inside that bucket

## Notebook 2: Load and Query Embeddings

Open `load_and_query_embedding.ipynb` and run top to bottom.

What it does:

- reads `airline_security_policy.txt` as paragraph chunks
- embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
- validates embedding dimension against the AWS index dimension
- uploads vectors with metadata to S3 Vectors
- runs a semantic query and prints ranked matches

## Configuration You Must Fill

Both notebooks have one config cell where you set:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `BUCKET_NAME`
- `VECTOR_INDEX_NAME`

The second notebook also sets `MODEL_NAME` and `TOP_K`.

## Important Dimension Rule

The vector index dimension in AWS must match the embedding model output dimension.

- `all-MiniLM-L6-v2` typically outputs 384-dimensional vectors
- if your index uses a different dimension, recreate the index or use a model with matching output size
