# Two-stage Fruit Vectors demo (for vector database training)

This companion document now clarifies how the new **Stage 1** and **Stage 2** notebooks in `06-Semantic-Search` use a hand-crafted fruit dataset stored inside **LanceDB**. Keep the cell-by-cell discipline: each cell explains its intent before the code, and both notebooks intentionally avoid relying on real embedding models. The notes below describe the teaching narrative and the Docker-based delivery.

## Stage 1 – Build a fruit vector database (notebook: `06-Semantic-Search/01-fruits-vector-setup.ipynb`)

1. **Cell 1 – Title + overview**
   - Markdown introduces the core idea: we manually define fruit vectors, persist them to LanceDB, and later query the space so students can see how nearest neighbors behave when the target (mango) is missing.

2. **Cell 2 – Imports, helpers, and LanceDB path**
   - Markdown explains that we only need `numpy`, `lancedb`, and the small `Vector` schema helper. The code cell imports the modules and declares the shared `fruits_lancedb` directory plus the `fruit_vectors` table name.

3. **Cell 3 – Schema and persistence helpers**
   - Markdown summarizes the `FruitVector` LanceModel schema, the helper that drops/recreates the table, and the cosine similarity function the notebook uses for later ranking.
   - Code cell defines the schema, the helper, and prepares the table so Stage 1 starts from a clean LanceDB state.

4. **Cell 4 – Insert handcrafted fruit vectors**
   - Markdown makes it clear that no embedding model runs: we craft three-dimensional vectors for each fruit, insert them into LanceDB, and then print the stored entries to show the raw data.
   - Code cell defines the vector dictionary, writes the rows into the LanceDB table, prints the count, and displays the stored records to verify everything is persisted.

5. **Cell 5 – Mango-like query and stage closeout**
   - Markdown walks through querying the stored table with a mango-like vector, ranking everything by cosine similarity, and confirming mango is absent from the table before announcing Stage 1 is done.
   - Code cell reads all rows back from LanceDB, computes the ranking, prints the top matches, asserts mango is missing, and leaves the table intact for Stage 2.

## Stage 2 – Compare similarity metrics (notebook: `06-Semantic-Search/02-fruits-similarity-metrics.ipynb`)

1. **Cell 1 – Title + context**
   - Markdown explains that Stage 2 reopens the LanceDB table created earlier, reissues the mango query, and compares multiple metrics to demonstrate how the winner stays consistent while the order of other fruits shifts.

2. **Cell 2 – Reload the stored fruit vectors**
   - Markdown describes that the notebook reconnects to the shared LanceDB path so it can run stand-alone without rerunning Stage 1.
   - Code cell opens the table, converts the stored vectors into numpy arrays, prepares the same mango query vector, and prints how many fruits were loaded.

3. **Cell 3 – Compare metrics**
   - Markdown sets expectations: we will compute cosine similarity (higher is better), Euclidean distance (lower is better), and dot product (higher is better), then display ordered rankings for each metric so learners can compare.
   - Code cell defines helper functions for each metric, sorts the fruits accordingly, and prints the top three per metric to highlight how the ordering sensitivity changes while the mango neighbour stays consistent.

## Docker packaging reminder

- The folder's Docker image now installs `lancedb` from `requirements.txt`, copies both notebooks, and launches JupyterLab on port 8888 so the Stage 1 store + Stage 2 comparison experience is ready as soon as the container starts.
- Follow the README's prune/build/run commands to delete old images/volumes, rebuild the image, and start JupyterLab for a fresh training session.
