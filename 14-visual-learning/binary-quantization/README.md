# Binary Quantization — Visual Learning

This lesson is a self-contained, single-file HTML experience that explains **binary quantization** for vector search: compressing each float dimension to one sign bit, comparing exact (cosine) vs binary (Hamming) retrieval, and reasoning about recall and dimension tradeoffs.

## Prerequisites

- Basic familiarity with vectors and similarity search (helpful but not required).
- A modern desktop browser (Chrome, Safari, Firefox, or Edge).

## What is included

| Path | Purpose |
|------|---------|
| `demo/index.html` | Full reference lesson — all interactions work. Open this file directly in a browser. |
| `student_lab/index.html` | Same UI with **four** implementation gaps for practice. Open in a browser after filling in the TODOs. |
| `student_lab/guide.md` | Step-by-step instructions and exact answers for each missing block. |

## How to run

1. Clone or download this repository.
2. **Demo:** open `demo/index.html` (double-click or “Open with” your browser).
3. **Lab:** open `student_lab/index.html` before editing; then implement the `STUDENT TODO` sections in the `<script>` block using `guide.md`.

No build step or server is required.
