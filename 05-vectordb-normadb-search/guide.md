# Student lab — slide hand panel

Use this alongside the **http://localhost:3000** demo. Work top to bottom.

---

## Before you start

- Have the repo running: `docker compose up` in this folder, then open the app in your browser.
- You will use **data/support_tickets.csv** (IT support tickets: same data loaded into Postgres **and** LanceDB).

---

## 1 — Upload CSV

**Do this**

1. Click **Choose file** / browse and select **`support_tickets.csv`** (under `data/` in this project).
2. Confirm the filename shows next to the control.

**Why:** The browser now knows which file to read for **Preview** and **Load**. Nothing is sent to the database until you click **Load**.

---

## 2 — Preview CSV

**Do this**

1. Click **Preview CSV**.
2. Skim the **headers** (e.g. id, title, body, category, …).
3. Read **2–3 sample rows** so you know what a “ticket” looks like.

**Why:** Preview parses the CSV **in the browser** (first rows only). It checks you picked the right file and columns **before** a full load.

---

## 3 — Load CSV

**Do this**

1. Click **Load Into Postgres + LanceDB**.
2. Wait until the UI shows success / counts (no errors).
3. Notice: **left** = SQL (Postgres), **right** = vector search (LanceDB).

**Why:** This step **POSTs** the file to the server, inserts rows into Postgres, builds embeddings, and stores vectors in LanceDB.

---

## MCQ

**Q1.** When are rows actually written to Postgres and LanceDB in this app?  
A) As soon as you pick the file in the file chooser  
B) When you click **Load Into Postgres + LanceDB**  
C) When you click **Preview CSV**  

**Q2.** Which question is a **bad** fit for vector search alone (use SQL / analytics instead)?  
A) “Which tickets are most **similar in meaning** to this customer email?”  
B) “**How many** tickets per **category**?” (counts across the whole table)  
C) “Find tickets that **mention** this exact product code in the title.”  

**Answer key:** Q1 **B** · Q2 **B** (Q1: Preview only shows a table in the page; Q2: `GROUP BY` / aggregates are relational SQL territory.)

---

## 4 — Lab A: “Account access” (keyword SQL vs meaning)

**Problem (plain English)**  
Find tickets about users who **cannot get into their account** or **sign-in is broken** — but people describe this in different ways (password, MFA, locked account, SSO loop).

**Run in Postgres (SQL)** — paste and execute:

```sql
SELECT id, title, priority, status, resolution
FROM documents
WHERE title ILIKE '%password%'
   OR body ILIKE '%password%'
ORDER BY id;
```

**Look at the result set** — note how many rows and **which** tickets matched.

**Run on the vector side** — in the natural-language / semantic box, enter:

```text
I am locked out and my sign in code never arrives
```

**Compare**

- Which ticket IDs appear only in SQL, only in vector, or in both?
- **Why** might SQL miss a ticket that is still clearly about login/access?

**Takeaway:** SQL matches **literal** words; vector search matches **intent** when wording differs.

---

## 5 — Lab B: “Remote / internal tools” (VPN wording)

**Problem**  
Find tickets where **home internet works** but **internal** tools (wiki, Jira, intranet) do not — users may not say “VPN.”

**SQL**

```sql
SELECT id, title, category, resolution
FROM documents
WHERE title ILIKE '%vpn%'
   OR body ILIKE '%vpn%'
ORDER BY id;
```

**Look at the result set.**

**Vector query**

```text
My internet works at home but I still cannot open internal company tools
```

**Compare** again: counts, IDs, and wording.

**Takeaway:** Keyword search is tied to **words you guessed**; semantic search can surface **same issue, different vocabulary**.

---

## 6 — Lab C: Aggregation (SQL wins — vector DB is the wrong tool)

**Problem**  
**How many tickets per category?** (e.g. access, network, permissions, …) — this is a **report** question, not a “find similar text” question.

**SQL — run in Postgres**

```sql
SELECT category, COUNT(*) AS total
FROM documents
GROUP BY category
ORDER BY total DESC;
```

**Look at the output** — rows are **groups** with **counts**.

**Vector side**

- There is **no** “GROUP BY category” in vector search. A vector DB finds **similar documents** to a query text; it does **not** replace **aggregation** or **joins** for analytics.
- Optional: type a semantic query like *“permission problems”* on the vector side and see **ranked tickets**, not a bar chart of counts.

**Takeaway**

- **Use SQL** (or a warehouse) for **counts, sums, GROUP BY, joins**, strict filters.
- **Use vector search** for **similarity / intent** when the question is phrased in natural language and keywords are unreliable.

---

## Done

You uploaded, previewed, loaded, and compared **SQL vs vector** on the same data — including one task (**aggregation**) where **relational SQL is the right tool** and a vector DB is **not** a substitute.
