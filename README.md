# Zero to Hero: Building an Airline Policy Bot with Qdrant ✈️

Welcome to the **Zero to Hero** course! In this series, we will build an intelligent **Airline Policy Bot** that can answer questions about baggage, pets, and fees. We will go from simple scripts to a "Context-Aware" AI assistant.

This repository uses **Qdrant**, a high-performance open-source vector search engine.

## Course Prerequisites
*   **Docker** (for running Qdrant locally)
*   **Python 3.8+**
*   **Pip** (Python package manager)

## 🗺️ Course Roadmap

We follow a progressive "Problem -> Solution" narrative.

### 1. [Setup and Connection](./01-setup-and-connection)
*   **Goal:** Get Qdrant running.

### 2. [Hello Vector World](./02-hello-vector-world)
*   **Goal:** Basic concepts.

### 3. [Semantic Search (The Problem)](./03-semantic-search)
*   **Goal:** Ingest data and try "Semantic Search".
*   **Run (Backend):** `python3 03-semantic-search/semantic_search.py`
*   **Run (Frontend):** `streamlit run 03-semantic-search/chatbot_ui_03.py`
*   **Observation:** The results show mixed policies (confusing).


### 4. [Metadata Filtering (The Solution)](./04-metadata-filtering)
*   **Goal:** Apply Filters.
*   **Run:** `python3 04-metadata-filtering/filtering.py`
*   **Observation:** The search now shows the correct policy in "Context-Aware" mode.

### 5. [Hybrid Search (Precision)](./05-hybrid-search)
*   **Goal:** Find exact fees.
*   **Run:** `python3 05-hybrid-search/hybrid_search.py`

### 6. [Admin Operations](./06-admin-operations)
*   **Goal:** Manage aliases and snapshots.


---

## 🚀 Future Ideas
1.  **RAG (Retrieval Augmented Generation)**: Connect this to an LLM to generate full sentences.
2.  **Multimodal**: Search for lost luggage images.
