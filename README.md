# Zero to Hero: Building an Airline Policy Bot with Qdrant ✈️

Welcome to the **Zero to Hero** course! In this series, we will build an intelligent **Airline Policy Bot** that can answer questions about baggage, pets, and fees. We will go from simple scripts to a "Context-Aware" AI assistant.

This repository uses **Qdrant**, a high-performance open-source vector search engine.

## Course Prerequisites
*   **Docker** (for running Qdrant locally)
*   **Python 3.8+**
*   **Pip** (Python package manager)

## 🗺️ Course Roadmap

We follow a progressive "Problem -> Solution" narrative.

### 1. [Getting Started](./01-Setup-and-Connection)
*   **Goal:** Get Qdrant running and verify your environment.

### 2. [The First Vector](./02-Hello-Vector-World)
*   **Goal:** Learn the basic concepts of vectors with a simple color demo.

### 3. [Understanding Embeddings](./03-Embedding-Basics)
*   **Goal:** Dive deeper into how embeddings represent data.

### 4. [Real World Embeddings](./04-Real-World-Embeddings)
*   **Goal:** Use Sentence Transformers on real diet data to understand semantic similarity.

### 5. [Semantic Search (The Problem)](./05-Semantic-Search)
*   **Goal:** Ingest data and try standard "Semantic Search".
*   **Observation:** You might see mixed policies or confusing results without filters.

### 6. [Precision with Metadata (The Solution)](./06-Metadata-Filtering)
*   **Goal:** Apply Filters to solve the confusion.
*   **Observation:** The search now shows the correct policy in "Context-Aware" mode.

### 7. [Hybrid Search](./07-Hybrid-Search)
*   **Goal:** Combine keyword search with vector search for exact fee lookups.

### 8. [Admin Operations](./08-Admin-Operations)
*   **Goal:** Manage aliases, snapshots, and collection maintenance.

### 9. [Cloud Integration](./09-AWS-S3-Integration)
*   **Goal:** Learn how to integrate with AWS S3 for vector storage.

### 10. [Reset Environment](./99-Reset-Environment)
*   **Goal:** Utilities to clear your environment and start fresh.

---

## 🚀 Future Ideas
1.  **RAG (Retrieval Augmented Generation)**: Connect this to an LLM to generate full sentences.
2.  **Multimodal**: Search for lost luggage images.
