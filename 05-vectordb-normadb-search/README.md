# SQL vs Vector Demo (Postgres + LanceDB)

Interactive lab: the same IT support tickets live in **Postgres** (left) for SQL and **LanceDB** (right) for semantic search. Use this README for setup, dataset notes, and the full teaching reference (queries and talking points).

---

## Docker

### Purge
```bash
docker compose down -v --rmi local --remove-orphans
```

### Build
```bash
docker compose build --no-cache
```

### Run
```bash
docker compose up
```

Then open **http://localhost:3000**, use **[data/support_tickets.csv](data/support_tickets.csv)** in the UI, and follow **[guide.md](guide.md)** for the step-by-step student flow (includes one short check: 1–2 MCQs after load).

---

## What this demo teaches

- **SQL** searches with exact fields, words, and conditions.
- **LanceDB** searches for **similar meaning** (semantic similarity).

For each comparison below: run the SQL query, run the vector query, compare returned tickets, and discuss **why** the results differ.

---

## Setup (quick)

1. Start the app with Docker (`docker compose up`).
2. Open `http://localhost:3000`.
3. Pick `data/support_tickets.csv`.
4. Click **Preview CSV**.
5. Click **Load Into Postgres + LanceDB**.

---

## Example 1: Sign-in trouble means more than “password”

### What we are trying to find

Tickets about users being unable to access their account. People describe this in many ways: forgot password, verification code not arriving, account locked, login loop.

### SQL query

```sql
SELECT id, title, priority, status, resolution
FROM documents
WHERE title ILIKE '%password%'
   OR body ILIKE '%password%'
ORDER BY id;
```

### Vector query

```text
I am locked out and my sign in code never arrives
```

### What this shows

- SQL looks for the literal word `password`, so it can miss tickets that are clearly about account access but use different words (`MFA`, `verification code`, `account locked`, `SSO loop`).
- LanceDB does not need the exact same word; it can return multiple access-related tickets because the meaning is similar.

### Teaching point

This is the cleanest example of semantic search: SQL is precise but brittle; vector search is flexible because it matches intent.

---

## Example 2: Remote access without saying “VPN”

### What we are trying to find

Tickets where users can browse the internet but cannot reach internal company systems from outside the office. Real users often do not say `VPN`; they say things like home internet works but Jira does not open, internal wiki is unreachable, remote file share is slow.

### SQL query

```sql
SELECT id, title, category, resolution
FROM documents
WHERE title ILIKE '%vpn%'
   OR body ILIKE '%vpn%'
ORDER BY id;
```

### Vector query

```text
My internet works at home but I still cannot open internal company tools
```

### What this shows

- SQL only catches tickets that literally mention `vpn`; it may miss tickets that describe the same issue using `intranet`, `internal wiki`, or `home wifi`.
- LanceDB can connect those tickets because the underlying meaning is the same.

### Teaching point

Vector databases help when the user problem is real but the wording is inconsistent.

---

## Example 3: Permission problems without exact error text

### What we are trying to find

Tickets where users can see a folder or site but cannot actually use it. Users may describe this as access denied, forbidden, read only, cannot open the file, not allowed to save changes.

### SQL query

```sql
SELECT id, title, category, resolution
FROM documents
WHERE body ILIKE '%access denied%'
   OR body ILIKE '%forbidden%'
   OR body ILIKE '%read only%'
ORDER BY id;
```

### Vector query

```text
I can see the team folder but I am not allowed to open or edit the files
```

### What this shows

- SQL works only if we list the exact phrases we expect; if wording differs, the query becomes longer and more fragile.
- LanceDB can retrieve permission-related tickets even when wording changes from `forbidden` to `cannot save` or `not allowed`.

### Teaching point

The vector database is not searching for one exact phrase; it is searching for tickets that **mean** the same thing.

---

## Example 4: Where SQL is clearly better — aggregation

Vector databases are not built to replace **GROUP BY** reporting. Use SQL for counts, grouping, and strict filters.

### SQL query

```sql
SELECT category, COUNT(*) AS total
FROM documents
GROUP BY category
ORDER BY total DESC;
```

Use this to explain:

- SQL is better for counting, grouping, sorting, joins, and strict filtering.
- LanceDB is better for finding related meaning across different wording. You would still use SQL (or a warehouse) for “how many tickets per category?”

---

## Short summary (talk track)

- **SQL asks:** which rows satisfy these exact rules?
- **LanceDB asks:** which rows mean something similar to this request?
