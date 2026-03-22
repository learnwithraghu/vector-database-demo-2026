# Vector Search Metrics Lab

## Overview
This lab is an interactive visual lesson for teaching the core similarity metrics used in vector databases:

- Cosine Similarity
- Euclidean Similarity / Distance
- Dot Product Similarity

The lab should help students build intuition by interacting with vectors directly on a 2D plane.  
The goal is not to teach code first, but to let students **see**, **predict**, and **test** how each metric behaves.

The lab should also include UI-based learning activities where students fill in missing concepts, make predictions, and manipulate vectors to create specific outcomes.

---

## Learning Goals

By the end of the lab, students should understand:

- a vector as a point / arrow in 2D space
- that different similarity metrics define "nearest" differently
- that cosine similarity mainly captures direction
- that Euclidean distance mainly captures spatial closeness
- that dot product depends on both direction and magnitude
- why metric choice changes vector database retrieval results
- why normalization matters in vector search

---

## Core Teaching Idea

Create a single interactive page where:

- documents are represented as vectors on a 2D plane
- the query is another vector
- students can drag the query vector
- the system recalculates rankings live
- students can switch between cosine, Euclidean, and dot product
- the interface explains the ranking in simple language

This should feel like a hands-on playground rather than a static diagram.

---

## Main Mental Model

Students should leave with these three intuitions:

- **Cosine Similarity**: "Are these vectors pointing in the same direction?"
- **Euclidean Distance**: "How close are these two endpoints in space?"
- **Dot Product**: "Are these vectors aligned, and how large are they?"

For vector databases:

- each document is embedded as a vector
- the query is embedded as another vector
- retrieval means finding the nearest vectors
- the metric determines what "nearest" means

---

## Lab Name

**Vector Search Metrics Lab**

Optional alternate names:

- Visual Similarity Metrics Lab
- Interactive Vector DB Metrics Lab
- Understanding Vector Search Lab
- Similarity Metrics Playground

Recommended final name: **Vector Search Metrics Lab**

---

## Page Structure

The lab should be a single interactive page with the following sections.

### 1. Top Controls
Include:

- metric selector
  - Cosine Similarity
  - Euclidean Distance / Similarity
  - Dot Product
- toggle: Show formulas
- toggle: Normalize vectors
- toggle: Show top-k results only
- preset selector for teaching scenarios

### 2. Main Interactive Plane
A 2D Cartesian plane that shows:

- origin `(0,0)`
- document vectors
- one query vector
- labels for each vector
- draggable query vector tip
- optionally draggable document vectors for advanced tasks

Vectors should be displayed as arrows from the origin so that angle and magnitude are visually clear.

### 3. Live Explanation Panel
Show:

- selected metric
- current formula
- short plain-language explanation
- why the top result is winning
- changing explanation as the query moves

Examples:

- "This document ranks highest because it points in almost the same direction as the query."
- "This document ranks highest because it is closest to the query point."
- "This document ranks highest because it is aligned with the query and has larger magnitude."

### 4. Ranking Panel
A live sorted result list showing:

- document name
- score or distance
- current rank
- short reason

### 5. Student Activity Panel
This area is for the learning exercises built into the UI.

It should support:

- prediction before reveal
- fill-in-the-blank concept checks
- drag challenges
- formula completion
- explanation matching

---

## Interaction Model

The primary interactions should be:

- drag the query vector tip
- switch similarity metric
- turn normalization on or off
- load preset scenes
- answer prediction questions
- fill missing labels or formula parts
- complete guided challenges

Every major interaction should immediately update the ranking and explanation.

---

## Recommended Teaching Flow

Structure the experience in stages.

### Stage 1: What is a vector?
Introduce:

- vectors as arrows from the origin
- coordinates `(x, y)`
- magnitude
- angle / direction

Student action:

- drag a vector and watch coordinates, angle, and magnitude change

### Stage 2: Compare two vectors
Show:

- one document vector
- one query vector

Display:

- angle difference
- distance
- magnitude of each vector
- score under each metric

Student action:

- move the query and observe how each metric changes

### Stage 3: Compare three similarity methods
Let students switch between:

- cosine
- Euclidean
- dot product

Student goal:

- observe that the same vectors can produce different rankings depending on the metric

### Stage 4: Vector database retrieval
Expand to a small set of documents:

- 6 to 10 vectors
- one query
- live top-k nearest neighbors

Student goal:

- understand that vector DB search depends on the selected metric

### Stage 5: Normalization
Add a normalization toggle.

When enabled:

- vectors are scaled to the same length
- students observe that dot product behaves more like cosine

Student goal:

- understand why normalization matters in embeddings and search

---

## Key Preset Scenes

The lab should include preset configurations that create strong visual differences between metrics.

### Preset 1: Same Direction, Different Length
Example vectors:

- Query = `(1,1)`
- Doc A = `(2,2)`
- Doc B = `(5,5)`

Expected learning:

- cosine treats A and B similarly because direction is the same
- Euclidean prefers the closer one
- dot product rewards larger magnitude

Main message:

- cosine ignores scale
- Euclidean cares about closeness
- dot product includes magnitude

---

### Preset 2: Similar Distance, Different Direction
Arrange vectors so two documents are similarly close to the query, but one is more directionally aligned from the origin.

Expected learning:

- Euclidean can be similar for both
- cosine can strongly prefer the better-aligned vector

Main message:

- direction and distance are not the same thing

---

### Preset 3: Large Magnitude Trap
Example vectors:

- Query = `(1,0)`
- Doc A = `(1.2,0.1)`
- Doc B = `(8,1)`

Expected learning:

- dot product may over-prefer the long vector
- cosine reduces the effect of magnitude
- Euclidean may prefer the nearby vector

Main message:

- dot product can favor large vectors even when they are not nearest in space

---

### Preset 4: Rotating Query
The query moves in a circle around the origin while keeping the same magnitude.

Expected learning:

- cosine changes based on angle match
- Euclidean changes based on endpoint proximity
- dot product changes based on both angle and vector size

Main message:

- the metrics react differently to movement

---

### Preset 5: Normalization Changes Ranking
Use vectors with noticeably different magnitudes.

Expected learning:

- before normalization, dot product may favor larger vectors
- after normalization, dot product ranking becomes much closer to cosine ranking

Main message:

- normalization often changes retrieval behavior in meaningful ways

---

## Visual Rules by Metric

To make the metric differences obvious, the lab should change the visual emphasis depending on the selected mode.

### In Cosine Mode
Highlight:

- angle between vectors
- directional alignment
- vectors with small angular difference

Visual aids:

- angle arc near origin
- directional glow or highlight for aligned vectors

### In Euclidean Mode
Highlight:

- distance from query tip to document tips

Visual aids:

- dashed line from query endpoint to each document endpoint
- shortest line highlighted strongly

### In Dot Product Mode
Highlight:

- direction alignment
- vector magnitude

Visual aids:

- show both angle and vector length clearly
- emphasize longer aligned vectors

---

## Student Learning Activities

The lab should include interactive tasks that are completed directly in the UI.

### Activity 1: Predict the Winner
Before showing the ranking, ask:

- Which document will rank first under cosine?
- Which document will rank first under Euclidean?
- Which document will rank first under dot product?

Students choose an answer and then reveal the result.

Purpose:

- force prediction before explanation

---

### Activity 2: Fill in the Blank
Examples:

- "Cosine similarity mainly depends on ________."
- "Euclidean distance mainly depends on ________."
- "Dot product depends on ________ and ________."

Expected answers:

- angle / direction
- distance
- direction and magnitude

Purpose:

- reinforce the metric definitions in plain language

---

### Activity 3: Drag Challenge
Prompt students with goals such as:

- Move the query so cosine picks A but Euclidean picks B
- Move the query so dot product prefers the longer vector
- Make cosine tie A and B
- Make Euclidean and cosine disagree

Purpose:

- move from recognition to active control

---

### Activity 4: Formula Completion
Show formulas with missing components and let students fill them using UI elements.

Examples:

- cosine similarity = `(A · B) / (||A|| × ||B||)`
- Euclidean distance = `√((x1-x2)^2 + (y1-y2)^2)`
- dot product = `x1x2 + y1y2`

Purpose:

- connect visual intuition with symbolic form

---

### Activity 5: Explain the Ranking
Show a result table with a missing reason field.

Example:

| Doc | Score | Why? |
|-----|-------|------|
| A   | 0.98  | ____ |
| B   | 0.81  | ____ |

Students choose reasons such as:

- small angle
- closest point
- larger magnitude
- poor alignment

Purpose:

- train students to describe results using the correct concept

---

## Suggested Document Story Layer

To connect this lab to vector databases, attach simple document meanings to vectors.

Example document labels:

- A: Cats are playful pets
- B: Dogs need daily walks
- C: Stock market trends
- D: Big cats in the wild

Example query:

- "feline behavior"

Purpose:

- show that vectors can represent semantic meaning
- connect geometric intuition to embedding-based retrieval

The coordinates can still be artificial or simplified for teaching.

---

## Recommended Number of Vectors

Use:

- 1 query vector
- 5 to 8 document vectors for the main demo
- 1 or 2 vectors in the introductory comparison views

This is enough to show ranking while keeping the display readable.

---

## Essential Features

The lab should include the following minimum features:

- draggable query vector
- multiple fixed document vectors
- metric switcher
- live ranking updates
- explanation panel
- preset scenarios
- normalization toggle
- prediction activity
- fill-in-the-blank checks
- drag challenges

---

## Nice-to-Have Features

If time allows, include:

- animated query rotation
- score history over time
- top-k slider
- tooltips for formulas
- hint system for student challenges
- teacher mode with answer reveal
- reset button for each preset

---

## Teacher Demo Script Outline

A simple classroom sequence can be:

1. show two vectors and ask what "similar" should mean
2. demonstrate same direction but different lengths
3. switch between cosine, Euclidean, and dot product
4. add multiple document vectors and show different rankings
5. explain that vector DB retrieval depends on the metric
6. enable normalization and show why rankings change
7. let students complete prediction and drag challenges

---

## Expected Student Takeaways

Students should finish the lab able to say:

- cosine similarity compares direction
- Euclidean distance compares closeness in space
- dot product combines alignment and magnitude
- different metrics can return different nearest neighbors
- normalization can make dot product behave more like cosine
- vector databases depend on similarity choice to retrieve results

---

## Final Recommendation

Build this as a single-page interactive experience called:

**Vector Search Metrics Lab**

The strongest version of the lab is a **visual vector playground** with:

- draggable vectors
- switchable similarity metrics
- preset scenes where metrics disagree
- student prediction and fill-in tasks
- a normalization toggle
- a live ranking panel tied to vector database retrieval

The focus should remain on conceptual understanding through interaction, not on implementation details or code.
