# Vector DB Warmup Demo - Practical Interactive Page Ideas

## Goal
Create a single interactive HTML page that helps students *feel* the difference between querying a SQL database and a vector database, without requiring complex infrastructure.

This page should support these teaching topics:
1. What gets stored inside a vector database?
2. Vector embeddings: numerical representations of data
3. Querying normal database
4. Querying vector database
5. Vector querying methods
6. When to use which query method? - Part 1
7. When to use which query method? - Part 2

---

## Core Teaching Strategy
Use one tiny dataset (10-20 records) and let students run the same user intent through two different search styles:
- SQL-style exact/structured query
- Vector-style semantic similarity query

Students should see:
- why SQL is perfect for filters, exact matches, joins, aggregation
- why vector search is better for fuzzy meaning and "find similar" tasks
- where hybrid search is best (metadata filter + semantic ranking)

---

## Page Structure (Single HTML, no heavy infra required)

## 1) Intro Panel: "What is stored in a vector DB?"
Show three side-by-side cards for each document:
- Raw text (e.g., "How to reset account password")
- Metadata (id, category, language, date)
- Embedding vector (shortened numbers like `[0.12, -0.44, ...]`)

Interaction:
- Toggle "Show full vector" vs "Show shortened"
- Hover tip: embeddings are numerical coordinates in high-dimensional space

Learning outcome:
- Students understand vector DB stores vectors + payload/metadata + ids

## 2) Embedding Playground
Simple 2D visual projection (fake but intuitive) of 8-12 points with labels (articles/products/FAQs).

Interaction:
- Click a point to set it as query
- See nearest 3 neighbors highlighted
- Move a "similarity threshold" slider and watch results appear/disappear

Learning outcome:
- Similar meaning = closer vectors
- Threshold changes recall vs precision behavior

## 3) SQL Query Panel (Structured Search)
Provide a mini SQL editor with prefilled templates (read-only or guided fill):
- `SELECT * FROM faqs WHERE category = 'billing';`
- `SELECT * FROM faqs WHERE title LIKE '%refund%';`

Interaction:
- Run query button
- Result table with rows/columns
- "Why it matched" explanation (exact field or pattern match)

Learning outcome:
- SQL relies on schema, fields, exact operators

## 4) Vector Query Panel (Semantic Search)
Text box: "Describe what you need"

Interaction:
- Student types natural text, e.g., "I forgot my login credentials"
- System converts to embedding (mock step shown)
- Top-k similar docs returned with similarity scores
- "Why it matched" explanation (semantic closeness)

Learning outcome:
- Vector query can find relevant items even when keywords differ

## 5) Query Methods Panel
Show methods as tabs/cards and let students test each method:
- k-NN (top-k nearest)
- Radius/threshold search
- Hybrid search (metadata filter + vector rank)
- MMR/diversified retrieval (optional advanced toggle)

For each method show:
- What it does
- Best for
- Trade-off

Learning outcome:
- There is no single best method; use case decides

## 6) "When to Use Which" - Part 1 (Quick Decision Trainer)
Give 6 scenario cards. Student chooses SQL / Vector / Hybrid.

Example scenarios:
- "Find all orders from last 30 days" -> SQL
- "Find complaints similar to this message" -> Vector
- "Find English product docs semantically similar to query" -> Hybrid

Immediate feedback:
- Correct answer
- Why
- Typical production pattern

## 7) "When to Use Which" - Part 2 (Nuanced Cases)
Harder scenarios where answer is not obvious.

Examples:
- "Find candidate resumes with Python + cloud experience, then semantically rank by role fit" -> Hybrid
- "Search legal clauses similar in meaning, but only for jurisdiction = EU" -> Hybrid
- "Daily revenue by country" -> SQL only

Learning outcome:
- Students learn decision criteria under constraints (latency, explainability, filtering)

---

## Suggested Dataset (Tiny but Realistic)
Use one theme (support tickets, FAQs, products, or movie descriptions).

Recommended: Support knowledge base
- 15 short documents
- Metadata fields: `id`, `category`, `language`, `created_at`, `tags`
- Include synonyms intentionally:
  - "forgot password", "can't sign in", "account access issue"

Why this works:
- SQL keyword search will miss some semantic matches
- Vector search will catch intent similarity
- Hybrid can filter by language/category first

---

## Interaction Design Ideas (Practical for Teaching)
- Keep all interactions on one screen with three columns:
  - Data view
  - SQL query/results
  - Vector query/results
- Add a "Run both" button to compare outputs side by side
- Add a "Show internals" switch:
  - SQL: parsed filters
  - Vector: query embedding + nearest vectors
- Add "Surprise me" button with preset prompts to avoid typing delays during lecture

---

## Metrics to Display (Simple but Effective)
For each query run, show:
- Returned results count
- Approx query time (mocked if needed)
- Relevance notes (manual labels)
- Precision-like score (optional toy metric)

This helps students compare methods empirically, not only conceptually.

---

## Implementation Tracks

## Track A: Pure Frontend Mock (Fastest, zero infra)
How it works:
- Preload JSON data in browser
- Precompute or hardcode tiny embedding vectors
- Simulate SQL with simple JS filtering and LIKE behavior
- Simulate vector search with cosine similarity in JS

Pros:
- Easiest to host and demo
- No setup risk in class
- Works offline

Cons:
- Not a real database connection
- Less realistic for backend architecture discussion

Best for:
- First teaching session / concept warmup

## Track B: Light Realistic Demo with Docker (Optional)
How it works:
- Postgres for SQL queries
- Vector DB option 1: PostgreSQL + pgvector extension
- Vector DB option 2: Qdrant container
- Thin app backend calls both, frontend compares results

Pros:
- Real query behavior and API flow
- Better for advanced students

Cons:
- More setup, higher failure risk in live demos

Best for:
- Follow-up class or recorded demo where setup is pre-verified

---

## Recommended Path for Your Use Case
For a classroom live demo, start with Track A first.

Why:
- You asked for simple HTML and practical teaching
- It minimizes infra issues
- You can still demonstrate authentic concepts clearly

Then optionally prepare Track B as "Phase 2" if you want to showcase real DB plumbing.

---

## Demo Script (10-15 min classroom flow)
1. Show one dataset row with text + metadata + vector.
2. Ask student intent: "I cannot access my account".
3. Run SQL keyword query: show misses.
4. Run vector query: show semantically relevant hits.
5. Run hybrid query with metadata constraint (e.g., `language='en'`).
6. Open Query Methods panel: compare k-NN vs threshold.
7. Finish with scenario quiz (When to use which).

Outcome:
- Students leave with practical decision intuition, not just definitions.

---

## Practical Risks and Mitigations
- Risk: Students think vector DB replaces SQL.
  - Mitigation: Keep "SQL vs Vector vs Hybrid" decision cards always visible.
- Risk: Similarity score feels magical.
  - Mitigation: Show nearest-neighbor visualization and score bars.
- Risk: Demo time overrun.
  - Mitigation: Include 4-5 canned prompts and "Reset demo" button.

---

## If You Want Docker + Real DB Later (minimal stack)
- `postgres` container (SQL)
- `pgvector` extension on same Postgres (so one DB can do both structured + vector)
- Simple backend endpoint:
  - `/search/sql?q=...`
  - `/search/vector?q=...`
  - `/search/hybrid?q=...&category=...`

Using Postgres + pgvector for both sides is practical for teaching because students can compare in one system before introducing specialized vector stores.

---

## Final Recommendation
Build one polished single-page interactive demo in pure HTML/CSS/JS first (Track A), with side-by-side SQL vs vector behavior and a scenario-based decision trainer.

Keep Docker-backed realism as an optional extension once the conceptual differences are clear.