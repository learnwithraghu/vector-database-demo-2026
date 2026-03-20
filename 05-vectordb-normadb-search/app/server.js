const express = require("express");
const multer = require("multer");
const { parse } = require("csv-parse/sync");
const { Pool } = require("pg");
const fs = require("fs");
const path = require("path");

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const PORT = process.env.PORT || 3000;
const DATABASE_URL = process.env.DATABASE_URL || "postgresql://postgres:postgres@db:5432/vector_demo";
const VECTOR_DB_DIR = process.env.VECTOR_DB_DIR || path.join(__dirname, "lancedb-data");
const EMBEDDING_DIMS = 16;
const CONCEPT_DIMS = 6;

const pool = new Pool({
  connectionString: DATABASE_URL
});

let lance = null;
let lanceConnection = null;
let vectorTable = null;

const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "but",
  "by",
  "for",
  "from",
  "got",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "me",
  "my",
  "of",
  "on",
  "or",
  "our",
  "so",
  "that",
  "the",
  "their",
  "this",
  "to",
  "up",
  "was",
  "we",
  "with",
  "you",
  "your"
]);

const SEMANTIC_GROUPS = [
  {
    concept: "concept_auth_access",
    weight: 4,
    terms: [
      "password",
      "sign in",
      "signin",
      "login",
      "log in",
      "credentials",
      "verification code",
      "two-factor",
      "2fa",
      "sms code",
      "account access",
      "locked out",
      "authenticator",
      "account locked"
    ]
  },
  {
    concept: "concept_remote_network",
    weight: 4,
    terms: [
      "vpn",
      "remote",
      "home wifi",
      "home internet works",
      "internet works at home",
      "intranet",
      "internal wiki",
      "internal tools",
      "internal company tools",
      "cannot open internal",
      "cannot reach internal",
      "tunnel",
      "route policy",
      "no internet",
      "wifi"
    ]
  },
  {
    concept: "concept_permissions_access",
    weight: 5,
    terms: [
      "access denied",
      "forbidden",
      "permission denied",
      "shared drive",
      "cannot open files",
      "not allowed",
      "role",
      "permission",
      "group claims"
    ]
  },
  {
    concept: "concept_printing_output",
    weight: 4,
    terms: [
      "printer",
      "print",
      "queue",
      "blank pages",
      "pdf prints",
      "spooler",
      "driver"
    ]
  },
  {
    concept: "concept_email_security",
    weight: 4,
    terms: [
      "email",
      "outlook",
      "mail",
      "phishing",
      "suspicious email",
      "credential",
      "external partner",
      "relay"
    ]
  },
  {
    concept: "concept_device_software",
    weight: 3,
    terms: [
      "install",
      "admin rights",
      "disk space",
      "build agent",
      "export",
      "spreadsheet",
      "overheats",
      "webcam",
      "dock",
      "monitor"
    ]
  }
];

app.use(express.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname, "public")));

function tokenize(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .filter((token) => !STOP_WORDS.has(token));
}

function hashToken(token) {
  let h = 2166136261;
  for (let i = 0; i < token.length; i += 1) {
    h ^= token.charCodeAt(i);
    h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24);
  }
  return Math.abs(h >>> 0);
}

function embeddingFromText(text) {
  const vec = Array(EMBEDDING_DIMS).fill(0);
  const normalized = String(text || "").toLowerCase();
  const tokens = tokenize(normalized);
  const lexicalDims = EMBEDDING_DIMS - CONCEPT_DIMS;

  SEMANTIC_GROUPS.forEach((group, index) => {
    const matchCount = group.terms.filter((term) => normalized.includes(term)).length;
    if (matchCount > 0) {
      vec[index] = group.weight * matchCount;
    }
  });

  if (tokens.length === 0) {
    return vec;
  }

  for (const token of tokens) {
    const h = hashToken(token);
    const idx = CONCEPT_DIMS + (h % lexicalDims);
    const sign = h % 2 === 0 ? 1 : -1;
    const mag = 0.25 * (1 + ((h >> 8) % 3));
    vec[idx] += sign * mag;
  }

  const norm = Math.sqrt(vec.reduce((acc, val) => acc + val * val, 0));
  if (norm === 0) {
    return vec;
  }

  return vec.map((v) => Number((v / norm).toFixed(6)));
}

function parseCsv(buffer) {
  const records = parse(buffer, {
    columns: true,
    skip_empty_lines: true,
    trim: true
  });

  const required = ["id", "title", "body", "category", "priority", "status", "resolution", "language", "created_at", "tags"];
  const missing = required.filter((field) => records.length > 0 && !(field in records[0]));
  if (missing.length > 0) {
    throw new Error(`Missing CSV columns: ${missing.join(", ")}`);
  }

  return records.map((r) => ({
    id: Number(r.id),
    title: String(r.title || ""),
    body: String(r.body || ""),
    category: String(r.category || ""),
    priority: String(r.priority || ""),
    status: String(r.status || ""),
    resolution: String(r.resolution || ""),
    language: String(r.language || ""),
    created_at: String(r.created_at || ""),
    tags: String(r.tags || "")
  }));
}

async function initSqlSchema() {
  await pool.query("DROP TABLE IF EXISTS documents");
  await pool.query(`
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      body TEXT NOT NULL,
      category TEXT NOT NULL,
      priority TEXT NOT NULL,
      status TEXT NOT NULL,
      resolution TEXT NOT NULL,
      language TEXT NOT NULL,
      created_at DATE NOT NULL,
      tags TEXT NOT NULL
    )
  `);
}

async function initVectorDb() {
  fs.mkdirSync(VECTOR_DB_DIR, { recursive: true });
  lance = await import("@lancedb/lancedb");
  lanceConnection = await lance.connect(VECTOR_DB_DIR);
}

function normalizeVector(value) {
  if (!value) {
    return [];
  }
  return Array.isArray(value) ? value.map(Number) : Array.from(value, Number);
}

function toVectorPreview(vec) {
  return `[${vec
    .slice(0, 8)
    .map((num) => Number(num).toFixed(3))
    .join(", ")}${vec.length > 8 ? ", ..." : ""}]`;
}

async function getVectorTable() {
  if (vectorTable && vectorTable.isOpen()) {
    return vectorTable;
  }

  if (!lanceConnection) {
    return null;
  }

  const tableNames = await lanceConnection.tableNames();
  if (!tableNames.includes("documents_vectors")) {
    return null;
  }

  vectorTable = await lanceConnection.openTable("documents_vectors");
  return vectorTable;
}

async function waitForDatabase(retries = 30) {
  for (let i = 1; i <= retries; i += 1) {
    try {
      await pool.query("SELECT 1");
      return;
    } catch (err) {
      if (i === retries) {
        throw err;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}

app.get("/api/health", async (_req, res) => {
  try {
    await pool.query("SELECT 1");
    await getVectorTable();
    res.json({ ok: true });
  } catch (err) {
    res.status(500).json({ ok: false, error: err.message });
  }
});

app.post("/api/load-csv", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "CSV file is required." });
  }

  let records;
  try {
    records = parseCsv(req.file.buffer);
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    await client.query("TRUNCATE TABLE documents");

    const vectorRows = [];

    for (const row of records) {
      const combined = `${row.title} ${row.body} ${row.tags} ${row.resolution} ${row.category}`;
      const embedding = embeddingFromText(combined);
      await client.query(
        `
          INSERT INTO documents (id, title, body, category, priority, status, resolution, language, created_at, tags)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        `,
        [
          row.id,
          row.title,
          row.body,
          row.category,
          row.priority,
          row.status,
          row.resolution,
          row.language,
          row.created_at,
          row.tags
        ]
      );

      vectorRows.push({
        ...row,
        vector: embedding,
        searchable_text: combined
      });
    }

    vectorTable = await lanceConnection.createTable("documents_vectors", vectorRows, {
      mode: "overwrite"
    });

    await client.query("COMMIT");
    return res.json({ loaded: records.length });
  } catch (err) {
    await client.query("ROLLBACK");
    return res.status(500).json({ error: err.message });
  } finally {
    client.release();
  }
});

app.get("/api/sql-preview", async (_req, res) => {
  try {
    const result = await pool.query(
      `
      SELECT id, title, category, language, created_at, tags
      , priority, status, resolution
      FROM documents
      ORDER BY id
      LIMIT 50
      `
    );
    res.json({ rows: result.rows });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/api/vector-preview", async (_req, res) => {
  try {
    const table = await getVectorTable();
    if (!table) {
      return res.json({ rows: [] });
    }

    const rows = await table.query().limit(50).toArray();
    res.json({
      rows: rows.map((row) => ({
        id: row.id,
        title: row.title,
        category: row.category,
        status: row.status,
        vector_preview: toVectorPreview(normalizeVector(row.vector))
      }))
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/api/query/sql", async (req, res) => {
  const sql = String(req.body.sql || "").trim();
  if (!sql) {
    return res.status(400).json({ error: "SQL is required." });
  }

  const normalized = sql.toLowerCase();
  if (!normalized.startsWith("select")) {
    return res.status(400).json({ error: "Only SELECT queries are allowed in this demo." });
  }

  if (!normalized.includes("from documents")) {
    return res.status(400).json({ error: "Query must target documents table." });
  }

  try {
    const result = await pool.query(sql);
    return res.json({ rowCount: result.rowCount, fields: result.fields.map((f) => f.name), rows: result.rows });
  } catch (err) {
    return res.status(400).json({ error: err.message });
  }
});

app.post("/api/query/vector", async (req, res) => {
  const query = String(req.body.query || "").trim();
  const topK = 5;

  if (!query) {
    return res.status(400).json({ error: "Vector query text is required." });
  }

  const queryVector = embeddingFromText(query);

  try {
    const table = await getVectorTable();
    if (!table) {
      return res.status(400).json({ error: "Load a CSV first so LanceDB has data." });
    }

    const rawRows = await table
      .query()
      .nearestTo(queryVector)
      .distanceType("cosine")
      .limit(Math.max(topK * 5, 10))
      .toArray();

    const rows = rawRows
      .map((row) => {
        const distance = Number(row._distance ?? 1);
        const similarity = Number((1 - distance).toFixed(4));
        return {
          id: row.id,
          title: row.title,
          body: row.body,
          category: row.category,
          priority: row.priority,
          status: row.status,
          resolution: row.resolution,
          language: row.language,
          tags: row.tags,
          similarity,
          vector_preview: toVectorPreview(normalizeVector(row.vector))
        };
      })
      .slice(0, topK);

    return res.json({
      query,
      queryVector,
      topK,
      rowCount: rows.length,
      rows
    });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

app.get("*", (_req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

async function start() {
  await waitForDatabase();
  await initSqlSchema();
  await initVectorDb();
  app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

start().catch((err) => {
  console.error("Failed to start:", err);
  process.exit(1);
});
