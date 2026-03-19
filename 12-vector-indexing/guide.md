# 12-vector-indexing Student Guide

This guide works with:
- `00_simple_indexing_demo_student.ipynb`

## How to use
1. Open the student notebook.
2. Replace every literal marker `<replace here>` with the correct code (there are 3 placeholders).
3. Answer the MCQs in the notebook flow and refer to the hints if you get stuck.
4. Run cells in order after each placeholder change.

## Task 1 (Code Fill): Fill the embedding “buckets”

**Question:**  
The function `embed(text)` converts text into a 3-number vector using a small dictionary called `vocab`. Fill in `vocab` so the travel/policy/service scores are computed correctly.

**What to do (in words):**
- Find the code cell titled **“Convert text to vectors (teaching embedding)”**.
- Inside that cell, locate the dictionary `vocab = { ... }`.
- Replace each `<replace here>` value with a Python `set` of keywords for:
  - `travel`
  - `policy`
  - `service`

**Hint (copy/paste):**
```python
vocab = {
    "travel": {"flight", "ticket", "airport", "booking", "reservation"},
    "policy": {"policy", "rules", "allowance", "security"},
    "service": {"meal", "onboard", "refund", "cancel", "hotel"},
}
```

## Task 2 (MCQ): Linear scan distance checks

**Question:**  
In this demo, the function `linear_search()` compares the query vector to every stored vector. If there are `N` stored vectors, how many distance computations happen?

**Options:**
- A) `N`
- B) `top_k`
- C) `3` (because the embedding has 3 dimensions)
- D) `N/2`

**What to do (in words):**
- Read the loop inside `linear_search()`.
- Decide which option matches “distance computed once per stored vector”.

**Hint (answer):**
- Correct answer: **A) `N`**

## Task 3 (Code Fill): Fill the toy index bucket assignment

**Question:**  
The toy “index” divides vectors into 3 buckets and only searches the predicted bucket. Fill the bucket assignment line so each vector is placed into bucket `0`, `1`, or `2`.

**What to do (in words):**
- Find the code cell for **“After indexing: toy grouped index”**.
- Inside the loop where the code builds `index_groups`, locate the line with `<replace here>`.
- Replace it with logic that chooses the bucket based on the embedding dimension with the largest value.

**Hint (copy/paste):**
```python
group = int(np.argmax(e))  # 0=travel-ish, 1=policy-ish, 2=service-ish
```

## Task 4 (MCQ): Why indexing reduces distance checks

**Question:**  
Why does the “after indexing” toy method do fewer distance computations than the linear scan?

**Options:**
- A) It compresses vectors before comparing them.
- B) It avoids distance computations by using cosine similarity instead of Euclidean distance.
- C) It prunes the search space by only comparing inside a predicted bucket.
- D) It increases the embedding dimensionality.

**What to do (in words):**
- Look at where the code selects `candidate_ids`.
- Decide what part of the dataset it stops checking.

**Hint (answer):**
- Correct answer: **C) It prunes the search space by only comparing inside a predicted bucket.**

## Task 5 (Code Fill): Compute saved distance checks

**Question:**  
The final comparison prints how many distance checks were saved by indexing. Fill the placeholder so it prints:
`linear_checks - indexed_checks`.

**What to do (in words):**
- Find the final code cell that prints **“Comparison”**.
- Replace the `<replace here>` token in the “Saved checks” line.

**Hint (copy/paste):**
```python
linear_checks - indexed_checks
```

