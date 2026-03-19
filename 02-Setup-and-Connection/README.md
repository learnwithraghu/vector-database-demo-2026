# 02 - Setup and Connection (Dockerized Notebook)

This demo uses one image and one terminal.

## 1) Cleanup old run (optional)

```bash
docker rm -f vector-setup-demo 2>/dev/null || true
```

## 2) Build image

```bash
cd 02-Setup-and-Connection
docker build -t vector-setup-demo .
```

## 3) Run (foreground, no `-d`)

```bash
docker run --name vector-setup-demo -p 6333:6333 -p 6334:6334 -p 8888:8888 vector-setup-demo
```

This single container starts both:
- Qdrant on `6333/6334`
- JupyterLab on `8888`

Open:
- Notebook: `http://localhost:8888`
- Qdrant Dashboard: `http://localhost:6333/dashboard`

Notebook file:
- `setup_and_connection.ipynb`

## Stop

- In same terminal: `Ctrl + C`
- Then cleanup:

```bash
docker rm -f vector-setup-demo 2>/dev/null || true
```
