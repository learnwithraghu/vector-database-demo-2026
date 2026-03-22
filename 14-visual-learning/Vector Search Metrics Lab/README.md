# Vector Search Metrics Lab

Interactive visual lesson (single HTML file) for **cosine similarity**, **Euclidean distance**, and **dot product**. Students **drag the query**, pick a **metric** (stages 2–3), and watch the **ranked list** change — same document vectors, different “nearest neighbor.”

## Prerequisites

- A modern desktop browser (Chrome, Firefox, Safari, or Edge).
- No build step: open the HTML file directly.

## What’s in this folder

| Path | Purpose |
|------|--------|
| `plan.md` | Original design notes (broader than this build). |
| `demo/index.html` | **Reference** — original single-page metrics lab (unchanged). |
| `student_lab/index.html` | **Student UI** — centered 2D plane, **10 animal** vectors, side panels for metric + ranking, **four MCQ stages** (observe the table; Stages 1–3 keep a fixed query; Stage 4 unlocks **drag**). No code editing. |
| `student_lab/guide.md` | **UI walkthrough** for the student page (stages, MCQs, how to read #1). |

## How to run

1. Open `demo/index.html` for the full experience.
2. For the lab: open `student_lab/index.html` and work through the **stepper**, **MCQs**, and (in Stage 4) **drag the query**; see `student_lab/guide.md`.
3. Use `demo/index.html` for the alternate, non–animal layout.

## Lesson flow

**`demo/index.html`** — Three learning stages + summary; metric bar; draggable query in supported stages.

**`student_lab/index.html`** — Part 1 start page → Parts 2–3: Stages **1–3** (Euclidean / cosine / dot, fixed query) with **MCQ** each → Stage **4** (all metrics, **draggable query**, comparison MCQ) → Summary.

## Quality checklist (authors)

- Page background `#FFFFFF`; accent `#D97706`; generous padding.
- `demo/index.html` and `student_lab/index.html` are each standalone; no broken JS on load.
- Student `guide.md` matches the **student** UI (MCQs, not code blocks).
