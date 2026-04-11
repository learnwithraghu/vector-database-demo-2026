---
name: kilocli-html-builder
description: >
  Build stunning, single-page educational HTML experiences using KiloCLI. Trigger this skill
  whenever the user wants to create a self-contained HTML page for learning, teaching, or
  interactive guided experiences — especially when the page should have no scroll, guided
  navigation with Next buttons, stage-based or step-based flows, analogy-driven content, or
  any kind of single-file interactive lesson. Also trigger for requests like "build me a page
  to teach X", "create an interactive explainer", "make a no-scroll educational page", or
  "guided learning experience". If KiloCLI or kilocode is mentioned alongside HTML output,
  always use this skill.
---

# KiloCLI Single-Page Educational HTML Builder

This skill produces **stunning, production-grade single-file HTML pages** designed for
guided, interactive education. Pages are viewport-locked (no scroll), stage-based, and
use progressive disclosure to guide the learner from start to finish.

---

## Core Design Principles

### 1. No Scroll — Ever
Every stage must fit within `100vh`. Content is never clipped by the viewport edge.
Use `overflow: hidden` on `body` and `html`. If content feels crowded, reduce it — less
is always more on a single screen.

### 2. One Stage at a Time
The page is divided into **stages** (screens). Only one stage is visible at a time.
Navigation is always a **Next →** button (and optionally ← Back). No sidebar, no table of
contents, no jump links.

### 3. Guided, Not Browsed
The learner cannot skip ahead. Each stage unlocks only after the current one is seen
(or interacted with). The experience is linear and intentional.

### 4. Minimal Clutter
Every element on screen must earn its place. Remove decorative text, redundant labels,
and anything that does not directly serve the learner's current task or insight.
**One idea per stage.**

### 5. Stunning Aesthetics
Follow the aesthetic rules below. The page must feel designed — not templated.

---

## Stage Architecture

Every educational page built with this skill follows this structure:

```
Stage 0  →  INTRO / HOOK
            - What we'll learn (1 sentence promise)
            - An evocative visual or metaphor
            - "Let's Begin →" button

Stage 1..N  →  LEARNING STAGES (2–4 stages)
            - One concept per stage
            - Interactive element (quiz, slider, drag, reveal, animation)
            - Progress indicator (dots or step counter, minimal)
            - "Next →" button (disabled until interaction is complete)

Stage LAST  →  END / RECAP
            - What we learned (1–3 bullet takeaways)
            - A memorable closing line
            - Optional: "Explore Again" restart button
```

Typical total stages: **4–6**. Never exceed 7.

---

## HTML File Structure

All output is **one self-contained `.html` file**. No external CSS files, no external JS
files. Fonts may be loaded from Google Fonts (a single `@import`). No frameworks.
No dependencies beyond vanilla JS + CSS.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>[Topic] — Interactive Lesson</title>
  <style>
    /* All CSS here — CSS variables, resets, stage layout, animations */
  </style>
</head>
<body>
  <!-- Stage 0: Intro -->
  <section class="stage" id="stage-0"> ... </section>

  <!-- Stage 1..N: Learning -->
  <section class="stage hidden" id="stage-1"> ... </section>

  <!-- Stage Final: End -->
  <section class="stage hidden" id="stage-end"> ... </section>

  <script>
    // All JS here — stage transitions, interaction logic, progress tracking
  </script>
</body>
</html>
```

---

## Navigation Logic (JS Template)

```javascript
let currentStage = 0;
const stages = document.querySelectorAll('.stage');

function goTo(index) {
  stages[currentStage].classList.add('hidden');
  stages[index].classList.remove('hidden');
  stages[index].classList.add('enter');
  currentStage = index;
}

function next() {
  if (currentStage < stages.length - 1) goTo(currentStage + 1);
}

function back() {
  if (currentStage > 0) goTo(currentStage - 1);
}

// Gate the Next button until interaction is complete
function unlockNext(stageId) {
  const btn = document.querySelector(`#${stageId} .next-btn`);
  if (btn) btn.removeAttribute('disabled');
}
```

---

## CSS Rules — Layout

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  width: 100%; height: 100%;
  overflow: hidden;          /* THE RULE: no scroll ever */
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--text);
}

.stage {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  padding: 2rem clamp(1.5rem, 8vw, 6rem);
  transition: opacity 0.4s ease, transform 0.4s ease;
}

.stage.hidden { opacity: 0; pointer-events: none; transform: translateX(30px); }
.stage.enter  { animation: slideIn 0.4s ease forwards; }

@keyframes slideIn {
  from { opacity: 0; transform: translateX(30px); }
  to   { opacity: 1; transform: translateX(0);    }
}
```

---

## Aesthetic System

Pick ONE aesthetic direction per page and commit to it fully. These are the allowed palettes
and typographic combinations for this skill:

### Dark — Cosmic / Deep
```css
--bg: #0a0a12;
--surface: #12121f;
--accent: #7c6af7;
--accent2: #3de8b0;
--text: #e8e6f0;
--muted: #6b6880;
--font-display: 'Space Grotesk', sans-serif;   /* only time this is acceptable */
--font-body: 'DM Sans', sans-serif;
```

### Light — Editorial / Clean
```css
--bg: #f5f3ee;
--surface: #ffffff;
--accent: #e84545;
--accent2: #1a1a2e;
--text: #1a1a1a;
--muted: #888;
--font-display: 'Playfair Display', serif;
--font-body: 'Source Sans 3', sans-serif;
```

### Dark — Amber / Warm Tech
```css
--bg: #111008;
--surface: #1c1a0f;
--accent: #f5a623;
--accent2: #e05c2a;
--text: #f0eadc;
--muted: #7a7060;
--font-display: 'Syne', sans-serif;
--font-body: 'IBM Plex Sans', sans-serif;
```

### Light — Pastel / Soft
```css
--bg: #eef2ff;
--surface: #ffffff;
--accent: #6366f1;
--accent2: #ec4899;
--text: #1e1b4b;
--muted: #a5b4fc;
--font-display: 'Nunito', sans-serif;
--font-body: 'Nunito', sans-serif;
```

**Rules:**
- Always import the chosen fonts from Google Fonts
- Use `--accent` for CTAs, highlights, and interactive elements
- Use `--muted` for helper text, captions, progress indicators
- Max 2 font families per page

---

## Interactive Stage Patterns

Use one of these patterns per learning stage:

### A. Reveal-on-Click
Content is blurred/hidden behind a card. User clicks to reveal.
```html
<div class="reveal-card" onclick="this.classList.toggle('revealed')">
  <div class="hidden-layer">Click to reveal</div>
  <div class="content-layer">The actual insight lives here.</div>
</div>
```

### B. Choice / Quiz (No wrong/right feedback until chosen)
```html
<div class="choices">
  <button class="choice" onclick="pickChoice(this, true)">Option A</button>
  <button class="choice" onclick="pickChoice(this, false)">Option B</button>
</div>
```
After any choice is made, show feedback and unlock Next.

### C. Analogy Builder (Drag / Match)
Side-by-side columns. Left = real-world term. Right = technical term. 
User draws a connection or clicks to match.

### D. Slider / Dial
A range slider that reveals how a parameter changes the outcome.
Visual changes in real-time. Great for numeric concepts.

### E. Step Reveal
A sequence of items appears one at a time as the user clicks "Show next step."
The Next stage button only appears after all items are revealed.

---

## Progress Indicator

Always show a minimal progress indicator. Dots preferred:

```html
<div class="progress-dots">
  <span class="dot active"></span>
  <span class="dot"></span>
  <span class="dot"></span>
</div>
```

```css
.progress-dots { display: flex; gap: 8px; position: absolute; bottom: 2rem; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); transition: 0.3s; }
.dot.active { background: var(--accent); transform: scale(1.3); }
```

Update dots in JS when calling `goTo()`.

---

## Next Button Style

```css
.next-btn {
  margin-top: 2rem;
  padding: 0.75rem 2rem;
  background: var(--accent);
  color: var(--bg);
  border: none;
  border-radius: 999px;
  font-size: 1rem;
  font-weight: 700;
  cursor: pointer;
  transition: transform 0.2s, opacity 0.2s;
  letter-spacing: 0.03em;
}
.next-btn:hover:not(:disabled) { transform: scale(1.05); }
.next-btn:disabled { opacity: 0.3; cursor: not-allowed; }
```

---

## Content Rules

- **Intro stage**: 1 headline (max 8 words) + 1 sub-line (max 15 words)
- **Learning stage**: 1 concept label + 1–2 sentences of explanation + 1 interactive element
- **End stage**: "You learned:" + 2–3 bullet points (max 10 words each)
- **Never** put more than 80 words of body text on any single stage
- **Always** use an analogy or metaphor for abstract concepts (see VectorDB Instructor skill)

---

## KiloCLI File Output Convention

When saving for use with KiloCLI / kilocode repos:
- Output file: `lessons/[topic-slug].html`
- No build step required — pure HTML
- Works in any modern browser without a server (open file:// directly)
- All assets inlined (SVG preferred over img tags for illustrations)

---

## Checklist Before Finalizing

- [ ] No element overflows the viewport on a 1280×800 screen
- [ ] Every stage has exactly one `next-btn` (or is the final stage)
- [ ] Next button is disabled until interaction is complete (for interactive stages)
- [ ] Progress dots update correctly
- [ ] Fonts are loaded via Google Fonts `@import`
- [ ] Color variables are defined in `:root`
- [ ] No Lorem Ipsum text
- [ ] Analogy/metaphor is used on every learning stage
- [ ] File is fully self-contained (no external JS/CSS files)