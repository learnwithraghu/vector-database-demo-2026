const csvFileInput = document.getElementById("csvFile");
const previewCsvBtn = document.getElementById("previewCsvBtn");
const loadBtn = document.getElementById("loadBtn");
const statusEl = document.getElementById("status");
const csvPreviewEl = document.getElementById("csvPreview");

const sqlEditor = document.getElementById("sqlEditor");
const runSqlBtn = document.getElementById("runSqlBtn");
const sqlMeta = document.getElementById("sqlMeta");
const sqlResults = document.getElementById("sqlResults");

const vectorInput = document.getElementById("vectorInput");
const runVectorBtn = document.getElementById("runVectorBtn");
const vectorMeta = document.getElementById("vectorMeta");
const vectorResults = document.getElementById("vectorResults");

const dbSnapshot = document.getElementById("dbSnapshot");
const vectorSnapshot = document.getElementById("vectorSnapshot");

function setStatus(message, tone = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${tone}`.trim();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderTable(container, rows) {
  if (!rows || rows.length === 0) {
    container.innerHTML = "<p>No rows to display.</p>";
    return;
  }

  const cols = Object.keys(rows[0]);
  const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");

  const body = rows
    .map((r) => {
      const tds = cols.map((c) => `<td>${escapeHtml(r[c] ?? "")}</td>`).join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");

  container.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function parseCsvText(text) {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines.length === 0) {
    return [];
  }

  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1, 7).map((line) => {
    const vals = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = vals[i] || "";
    });
    return row;
  });
}

async function refreshSnapshots() {
  const [sqlRes, vecRes] = await Promise.all([fetch("/api/sql-preview"), fetch("/api/vector-preview")]);

  if (sqlRes.ok) {
    const sqlData = await sqlRes.json();
    renderTable(dbSnapshot, sqlData.rows);
  }

  if (vecRes.ok) {
    const vecData = await vecRes.json();
    renderTable(vectorSnapshot, vecData.rows);
  }
}

previewCsvBtn.addEventListener("click", async () => {
  const file = csvFileInput.files?.[0];
  if (!file) {
    setStatus("Pick a CSV file first.", "warn");
    return;
  }

  const text = await file.text();
  const rows = parseCsvText(text);
  csvPreviewEl.innerHTML = "<h3>CSV Preview (first 6 rows)</h3>";
  const tableHost = document.createElement("div");
  csvPreviewEl.appendChild(tableHost);
  renderTable(tableHost, rows);
  setStatus(`Preview ready for ${file.name}.`, "ok");
});

loadBtn.addEventListener("click", async () => {
  const file = csvFileInput.files?.[0];
  if (!file) {
    setStatus("Pick a CSV file before loading.", "warn");
    return;
  }

  setStatus("Loading CSV into Postgres and LanceDB...");
  const formData = new FormData();
  formData.append("file", file);

  const resp = await fetch("/api/load-csv", {
    method: "POST",
    body: formData
  });

  const data = await resp.json();
  if (!resp.ok) {
    setStatus(`Load failed: ${data.error || "Unknown error"}`, "warn");
    return;
  }

  setStatus(`Load complete. Inserted ${data.loaded} rows into Postgres and LanceDB.`, "ok");
  await refreshSnapshots();
});

runSqlBtn.addEventListener("click", async () => {
  const sql = sqlEditor.value;
  sqlMeta.textContent = "Running SQL query...";

  const resp = await fetch("/api/query/sql", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sql })
  });

  const data = await resp.json();
  if (!resp.ok) {
    sqlMeta.textContent = `Error: ${data.error || "Unknown error"}`;
    sqlResults.innerHTML = "";
    return;
  }

  sqlMeta.textContent = `Rows: ${data.rowCount}. Matched by exact schema fields and SQL operators.`;
  renderTable(sqlResults, data.rows);
});

runVectorBtn.addEventListener("click", async () => {
  const query = vectorInput.value.trim();
  if (!query) {
    vectorMeta.textContent = "Enter a text query first.";
    return;
  }

  vectorMeta.textContent = "Running vector search...";

  const resp = await fetch("/api/query/vector", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query })
  });

  const data = await resp.json();
  if (!resp.ok) {
    vectorMeta.textContent = `Error: ${data.error || "Unknown error"}`;
    vectorResults.innerHTML = "";
    return;
  }

  vectorMeta.textContent = `Top ${data.rowCount} semantic matches from LanceDB. These are ranked by meaning, not exact keyword overlap.`;

  if (!data.rows || data.rows.length === 0) {
    vectorResults.innerHTML = "<p>No semantic matches found.</p>";
    return;
  }

  vectorResults.innerHTML = data.rows
    .map(
      (row) => `
      <article class="vector-card">
        <div><strong>${escapeHtml(row.title)}</strong></div>
        <div>Category: ${escapeHtml(row.category)} | Priority: ${escapeHtml(row.priority)} | Status: ${escapeHtml(row.status)}</div>
        <div class="score">Similarity: ${escapeHtml(row.similarity)}</div>
        <div>Vector: ${escapeHtml(row.vector_preview)}</div>
        <div>${escapeHtml(row.body)}</div>
        <div><strong>Resolution:</strong> ${escapeHtml(row.resolution)}</div>
      </article>
    `
    )
    .join("");
});

refreshSnapshots().catch(() => {
  setStatus("Backend is up. Load a CSV to begin.");
});
