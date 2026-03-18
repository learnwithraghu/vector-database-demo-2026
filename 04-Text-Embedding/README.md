# 04-Text-Embedding (KodeKloud Airlines + LanceDB)

This lab shows **how text embeddings work** end-to-end:

- Load a KodeKloud Airlines policy document
- Convert each section into **embeddings** (vectors)
- Store vectors in **LanceDB**
- Ask questions and retrieve the most relevant policy chunks via vector search

## Run with Docker

```bash
cd 04-Text-Embedding
docker build -t kk-text-embedding .
docker run --rm -p 8888:8888 kk-text-embedding
```

Open `http://localhost:8888` and run either:
- `text_embedding_lancedb.ipynb` (instructor)
- `text_embedding_lancedb_student.ipynb` (lab version with 4 fill-ins)

Lab guide: `guide.md`

