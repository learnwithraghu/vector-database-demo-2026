# Student Lab Guide

Complete the four `STUDENT TODO` sections in `student_lab/index.html` (inside the main `<script>` block). The page should load without errors before and after your edits; until TODOs #1–3 are done, encodings and scores will look wrong. Until TODO #4 is done, the dimension slider will not update the bars or verdict.

## Block #1 — Encode floats to bits (`toBin`)

**Location**: `student_lab/index.html`, near the top of the script where `toBin` is defined.

**Task**: Implement sign quantization: for each float in `v`, return `1` if the value is greater than or equal to zero, otherwise `0`. Use `Array.prototype.map`.

**Explanation**: Binary quantization in this lab keeps only the **sign** of each dimension. This function is what turns stored float vectors into bit patterns for fast comparison.

**Hint**:

```javascript
const toBin = (v) => v.map((x) => (x >= 0 ? 1 : 0));
```

## Block #2 — Cosine similarity (`cosine`)

**Location**: `student_lab/index.html`, `cosine` function.

**Task**: Return the cosine similarity of two same-length vectors `a` and `b`: dot product divided by the product of L2 norms. Add a small epsilon (e.g. `1e-9`) to the denominator to avoid division by zero.

**Explanation**: The “exact” search path ranks database vectors by cosine similarity to the query — the ground truth you compare binary search against.

**Hint**:

```javascript
const cosine = (a, b) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
};
```

## Block #3 — Hamming distance (`hamming`)

**Location**: `student_lab/index.html`, `hamming` function.

**Task**: Count how many positions `i` have different bits: use XOR (`^`) on `a[i]` and `b[i]` (0/1 values) and sum. You can use `reduce` over the array length.

**Explanation**: For binary-quantized vectors, a common fast distance is **Hamming distance** (number of mismatched bits). Lower is more similar when comparing the query bits to each stored bit vector.

**Hint**:

```javascript
const hamming = (a, b) => a.reduce((s, _, i) => s + (a[i] ^ b[i]), 0);
```

## Block #4 — Tradeoff slider UI (`updateTradeoff`)

**Location**: `student_lab/index.html`, `updateTradeoff` function (uses the existing `dimSteps` array above it).

**Task**: Read the slider value with `document.getElementById('dimSlider').value`, select `dimSteps[idx]`, then update:

- `dimDisplay` text to `"{dims} dimensions"`
- `recallBar` width and label text from `recallLabel` / midpoint of recall range
- `memBar` at `96.9%` (binary is ~32× smaller than float32)
- `dimVerdict`: set `className` to `` `dim-verdict ${d.verdict}` `` and `innerHTML` to `d.text`

**Explanation**: This ties the **dimension count** narrative to the UI: recall typically rises with more dimensions while memory savings from binary stay large.

**Hint**:

```javascript
function updateTradeoff() {
  const idx = +document.getElementById('dimSlider').value;
  const d = dimSteps[idx];
  const mem = 96.9;
  const midR = Math.round((d.recallLow + d.recallHigh) / 2);

  document.getElementById('dimDisplay').textContent = `${d.dims} dimensions`;
  document.getElementById('recallBar').style.width = midR + '%';
  document.getElementById('recallBar').textContent = d.recallLabel;
  document.getElementById('recallVal').textContent = d.recallLabel;
  document.getElementById('memBar').style.width = mem + '%';
  document.getElementById('memBar').textContent = mem + '%';
  document.getElementById('memVal').textContent = mem + '%';

  const v = document.getElementById('dimVerdict');
  v.className = `dim-verdict ${d.verdict}`;
  v.innerHTML = d.text;
}
```
