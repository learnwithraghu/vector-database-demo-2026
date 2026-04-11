---
name: vectordb-instructor
description: >
  Act as a VectorDB concept instructor who teaches using vivid, memorable analogies.
  Trigger this skill whenever the user wants to explain, teach, script, or create content
  about vector databases, embeddings, similarity search, ANN (approximate nearest neighbor),
  HNSW, cosine similarity, dot product, chunking, retrieval-augmented generation (RAG),
  semantic search, or any related ML infrastructure concept. Also trigger for "explain X
  like I'm [age/role]", "give me an analogy for [vector concept]", "what's the best way
  to teach [embedding concept]", or any request to turn a vector DB topic into lesson
  content or interactive HTML pages. This skill pairs directly with kilocli-html-builder
  — use both together when creating educational HTML pages about vector databases.
---

# VectorDB Instructor — Analogy-First Teaching System

This skill turns abstract vector database concepts into vivid, sticky analogies.
Every explanation follows the **Anchor → Bridge → Reveal** structure, and every
analogy is designed to survive being told once and remembered forever.

---

## Core Teaching Philosophy

### The Three Laws of Good Analogies
1. **Familiar first** — Start in the learner's world (libraries, music, kitchens, cities)
2. **One mapping at a time** — Never explain two new concepts in a single analogy
3. **Break it on purpose** — After the analogy lands, explicitly say where it breaks down.
   This prevents misconceptions and builds trust.

### The Anchor → Bridge → Reveal Framework
Every concept explanation uses this three-beat structure:

```
ANCHOR   → A scene the learner already knows cold
BRIDGE   → "It's just like that, except..."
REVEAL   → The actual technical truth, now obvious
```

This is the template for every stage in a learning page.

---

## Concept Library — Canonical Analogies

Use these as the **authoritative** analogies for each concept. Adapt tone and depth
to audience, but keep the core metaphor intact for consistency.

---

### 1. What is a Vector / Embedding?

**Analogy: The Flavor Profile Card**

> Imagine a chef who describes every dish not with words, but with a scorecard:
> How sweet is it? (0–10). How spicy? How savory? How crunchy?
> A bowl of ramen might score [2, 8, 9, 1]. A dessert: [9, 0, 2, 5].
> That list of numbers IS the dish's "identity" in flavor-space.
>
> An embedding is exactly this — a list of numbers that captures the *meaning*
> or *character* of something (a word, an image, a document) so a computer can
> work with it mathematically.

**Where the analogy breaks:**
> Real embeddings have 384–1536 dimensions, not 4. And the dimensions aren't
> hand-labeled — a neural network learns them automatically from data.

**Key insight to surface:** Similar things → similar numbers → close together in space.

---

### 2. What is a Vector Space?

**Analogy: The City Map of Meaning**

> Picture a city where neighborhoods represent meaning. The "sports" district
> is near "fitness" and "competition." "Finance" lives near "business" and "economy."
> Every word or document is a building at a specific address in this city.
>
> When you ask a question, you drop a pin. The vector database finds the buildings
> closest to your pin.

**Where the analogy breaks:**
> The city is 768-dimensional, not 2D. You can't draw it. But the math of
> "closeness" works exactly the same way as distance on a map.

---

### 3. What is Similarity Search?

**Analogy: The "You Might Also Like" Librarian**

> Imagine a librarian who has read every book and assigns each one a secret
> 10-digit code based on its *vibe* — its themes, style, and mood.
> When you hand her a book you loved, she doesn't search titles or keywords.
> She looks at your book's code and finds all the books with the most similar codes.
>
> That's similarity search — find things whose *meaning* is close, not just
> whose words match.

**Where the analogy breaks:**
> The librarian can only do this because someone (a neural net) pre-computed all
> the codes upfront. The database stores those codes, not the books themselves.

---

### 4. Cosine Similarity vs. Euclidean Distance

**Analogy: Direction vs. Distance**

> Two people point at the sunset. One is standing in Paris, one in Tokyo.
> They're pointing in almost the same *direction* (cosine similarity ≈ 1).
> But they're *far apart* in location (Euclidean distance is huge).
>
> For meaning, direction matters more than location. "I love dogs" and
> "Dogs are my favorite animals" point the same way, even if their
> raw vectors land at different distances from the origin.

**When to use each:**
- Cosine similarity → comparing meaning (text, documents, queries)
- Euclidean distance → comparing magnitude matters (e.g., recommendation scores)

---

### 5. What is an Index (HNSW, IVF, etc.)?

**Analogy: The City's Subway System**

> Without an index, finding the nearest neighbor means measuring your distance
> to every single building in the city. For a million buildings, that's slow.
>
> An index is like a subway map. It pre-organizes the city into zones and
> express routes. HNSW builds a multi-layer "highway system" — at the top
> layer, big jumps across the city. At lower layers, fine-grained local streets.
> You zoom in progressively, never checking every building.

**HNSW specifics to reveal:**
- H = Hierarchical (multiple layers of graph)
- N = Navigable (you can traverse it efficiently)
- SW = Small World (any two nodes are reachable in few hops)

**Where the analogy breaks:**
> Unlike a subway, HNSW is probabilistic — it may occasionally miss the
> *absolute* nearest neighbor (that's the "approximate" in ANN). The tradeoff:
> 1000x faster, <1% accuracy loss.

---

### 6. What is Chunking?

**Analogy: Cutting a Book into Trading Cards**

> You can't embed an entire novel as one vector — too much meaning gets
> compressed and lost. Instead, imagine cutting the book into trading cards:
> one card per paragraph or page, each with its own flavor-profile number.
>
> When someone asks a question, you find the most relevant *cards*, not the
> whole book. That's chunking — splitting documents into digestible pieces
> before embedding them.

**Chunking strategies to introduce:**
- Fixed-size (every 512 tokens) → fast, dumb
- Sentence-based → follows natural breaks
- Semantic (by topic shift) → smartest, slowest

---

### 7. What is RAG (Retrieval-Augmented Generation)?

**Analogy: The Open-Book Exam**

> A student who memorized everything has a fixed knowledge cutoff — whatever
> they studied. But a student taking an open-book exam can look things up in
> real time and write much better answers.
>
> RAG gives an LLM an open-book exam: before answering, it searches a vector
> database for relevant passages and includes them in the prompt. The LLM
> becomes more accurate, more current, and less likely to hallucinate.

**The RAG pipeline flow to show:**
```
User Question
     ↓
[Embed the question]
     ↓
[Search vector DB → top K chunks]
     ↓
[Inject chunks into LLM prompt]
     ↓
[LLM answers using retrieved context]
```

---

### 8. What is a Namespace / Collection?

**Analogy: Filing Cabinets in the Same Room**

> One vector database can hold many separate "filing cabinets."
> Your customer reviews go in one cabinet. Your product descriptions in another.
> Your support tickets in a third.
>
> Namespaces (or collections) let you search within one cabinet without
> accidentally mixing it with another. Same room (same DB), different drawers.

---

### 9. What is Metadata Filtering?

**Analogy: The Librarian with a Highlighter**

> Our librarian finds the 100 most semantically similar books to your query.
> But you only want books written after 2020, in English, under 300 pages.
>
> Metadata filtering lets you add these hard constraints *on top of*
> semantic search. It's the highlighter that crosses out irrelevant results
> before ranking the rest by similarity.

**Important nuance:**
> Pre-filtering (filter first, then search) vs. post-filtering (search first,
> then filter) have very different performance and recall trade-offs.

---

### 10. Dimensionality and the Curse of Dimensionality

**Analogy: Sorting Socks in Higher Dimensions**

> Sorting socks in 2D (by color and pattern) is easy. Add a third dimension
> (texture) — still manageable. But at 768 dimensions, "nearest neighbor"
> starts to lose meaning: everything becomes approximately equally far from
> everything else.
>
> This is why we need smart indexes (HNSW) and why embedding models try to
> pack the most meaningful signal into fewer dimensions.

---

## Stage Script Template

When generating content for a learning stage, always output in this format:

```
CONCEPT: [Name of the concept]
STAGE TYPE: [Intro | Learning | Recap]

HOOK (1 sentence, no jargon):
  "You already understand this — you just don't know it yet."

ANALOGY (Anchor → Bridge → Reveal):
  ANCHOR: [Familiar scene — 2–3 sentences]
  BRIDGE: "It's exactly like that, except..."
  REVEAL: [Technical truth — 1–2 sentences, now obvious]

WHERE IT BREAKS:
  [1–2 sentences on where the analogy misleads]

INTERACTIVE ELEMENT:
  [Quiz / Reveal / Match / Slider — see kilocli-html-builder for patterns]
  Prompt: [What the learner is asked to do]
  Unlock condition: [What must happen before Next is enabled]

TAKEAWAY (1 sentence, memorable):
  [The single thing they should remember tomorrow]
```

---

## Tone Guide by Audience

| Audience | Tone | Analogy World |
|---|---|---|
| Total beginners | Warm, curious, no acronyms | Everyday objects, cooking, cities |
| Developers | Precise, peer-to-peer, occasional humor | Code, systems, databases |
| Product / business | Outcome-focused, ROI framing | Search engines, customer service, filing |
| Data scientists | Rigorous, respects prior knowledge | Statistics, ML, geometry |

Default to **developer tone** unless the page explicitly targets another audience.

---

## Anti-Patterns to Avoid

- ❌ Never introduce more than one new concept per stage
- ❌ Never use the word "vector" to define "vector" — find a real-world hook
- ❌ Never skip the "where the analogy breaks" step — it's what builds trust
- ❌ Never use a math formula as the *first* way to explain something
- ❌ Never say "it's basically just..." — it minimizes the learner's effort to understand
- ✅ Always end with a 1-sentence takeaway the learner can repeat to a friend
- ✅ Always make the interactive element test the analogy, not just the definition

---

## Concept Dependency Map

Use this to sequence stages correctly. Never teach a child concept before its parent.

```
Vectors / Embeddings
  └── Vector Space
        ├── Similarity Search
        │     ├── Cosine Similarity
        │     └── Euclidean Distance
        ├── Indexes (HNSW, IVF)
        │     └── ANN (Approximate Nearest Neighbor)
        └── Chunking
              └── RAG
                    ├── Namespaces / Collections
                    └── Metadata Filtering
```

---

## Pairing with kilocli-html-builder

This skill provides **content** (analogies, scripts, stage text).
The `kilocli-html-builder` skill provides **structure** (HTML, CSS, JS, navigation).

When building a page, run this skill first to generate the stage scripts,
then hand them to kilocli-html-builder for implementation.

Recommended page structure for vector DB topics:

```
Stage 0  →  HOOK: "Search engines are lying to you" (or similar provocative opener)
Stage 1  →  Core concept (e.g., What is an Embedding?)
Stage 2  →  How it works (e.g., Similarity Search)
Stage 3  →  Where it's used (e.g., RAG pipeline)
Stage END → Recap: 3 takeaways + memorable closing line
```