## Lab Guide — Fruits Vector Demo (Student)

### Overview
This lab combines both stages into one notebook:
- Stage 1: Create and store fruit vectors in LanceDB
- Stage 2: Compare similarity metrics with visualizations

**Instructions**: Fill the `<replace_here>` placeholders, then run cells top-to-bottom.

---

### Cell 1: Imports (No fill-in)
Just run this cell - imports the required libraries.

---

### Cell 2: Create Vector Table
**Task**: Fill in the PyArrow types for the schema.

**Fill-in 1**: `pa.<replace_here>` → string type
- **Solution**: `string`
- Full line: `pa.field("name", pa.string)`

**Fill-in 2**: `pa.<replace_here>` → float type  
- **Solution**: `float32`
- Full line: `pa.field("vector", pa.list_(pa.float32(), 3))`

---

### MCQ 1
**Question**: What do the 3 dimensions in our fruit vectors represent?

A. Size, weight, price  
B. Red, yellow, green color intensity  
C. Sweetness, sourness, bitterness  
D. Calories, fiber, vitamin C

**Correct answer**: B

---

### Cell 3: Define and Store Fruit Vectors
**Task**: Fill in the method to add records to the table.

**Fill-in 3**: `table.<replace_here>`
- **Solution**: `add`
- Full line: `table.add(records)`

---

### Cell 4-5: Verify and Query (No fill-in)
Just run these cells to verify data and define mango query.

---

### Cell 6: Compare Similarity Metrics
**Task**: Fill in the numpy function for dot product.

**Fill-in 4**: `np.<replace_here>`
- **Solution**: `dot`
- Full line: `dot_prod = lambda a, b: np.dot(a, b)`

---

### MCQ 2
**Question**: For Euclidean distance, does a higher or lower value mean more similar?

A. Higher  
B. Lower  
C. It doesn't matter  
D. Only cosine matters

**Correct answer**: B

---

### MCQ 3
**Question**: Which fruit is most similar to mango based on our vectors?

A. Papaya  
B. Pineapple  
C. Apple  
D. Guava

**Correct answer**: A (Papaya has similar color: high red, medium yellow, low green)

---

### Summary

| Cell | Fill-in | Solution |
|------|---------|----------|
| 2 | pa.string type | `string` |
| 2 | pa.float32 type | `float32` |
| 3 | add records | `add` |
| 6 | numpy dot | `dot` |

---

### How to Run
1. Open `03-fruits-similarity-student.ipynb`
2. Fill in the 4 placeholders (short answers!)
3. Run cells in order (Shift+Enter)
4. Check the visualizations!

### What You'll See
- 3D scatter plot of fruits in color space
- 2D projection (Red vs Yellow)
- Comparison of top 3 fruits for each similarity metric
- The top match (papaya) is consistent across all metrics
