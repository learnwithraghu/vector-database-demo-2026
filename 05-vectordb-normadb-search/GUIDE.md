# Local Demo Guide

## Setup

1. Start the app with Docker.
2. Open `http://localhost:3000`.
3. Pick [data/support_tickets.csv](data/support_tickets.csv).
4. Click **Preview CSV**.
5. Click **Load Into Postgres + LanceDB**.

This demo uses the same IT support tickets in two systems:

1. Postgres on the left for SQL.
2. LanceDB on the right for semantic search.

The important teaching idea is this:

- SQL searches for exact fields exact words and exact conditions.
- LanceDB searches for similar meaning.

For each example below:

1. Run the SQL query.
2. Run the vector query.
3. Compare the returned tickets.
4. Ask students why the results are different.

---

## Example 1: Sign In Trouble Means More Than Password

### What we are trying to find

We want tickets about users being unable to access their account.

The problem is that people describe this in many ways:

- forgot password
- verification code not arriving
- account locked
- login loop

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

- SQL is looking for the literal word `password`.
- That means it can miss tickets that are clearly about account access but use different words such as `MFA`, `verification code`, `account locked`, or `SSO loop`.
- LanceDB does not need the exact same word.
- It can return multiple access-related tickets because the meaning is similar.

### Teaching point

This is the cleanest example of semantic search.

SQL is precise but brittle.
Vector search is flexible because it matches intent.

---

## Example 2: Remote Access Problem Without Saying VPN

### What we are trying to find

We want tickets where users can browse the internet but still cannot reach internal company systems from outside the office.

In real life users often do not say `VPN`.
They say things like:

- home internet works but Jira does not open
- internal wiki is unreachable
- remote file share is slow

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

- SQL only catches tickets that literally mention `vpn`.
- It may miss tickets that describe the same remote access issue using words like `intranet`, `internal wiki`, or `home wifi`.
- LanceDB can connect those tickets because the underlying meaning is the same.

### Teaching point

This is how vector databases help when the user problem is real but the wording is inconsistent.

Students should notice that vector search is often closer to how humans ask for help.

---

## Example 3: Permission Problems Without Exact Error Text

### What we are trying to find

We want tickets where users can see a folder or site but cannot actually use it.

Users may describe this as:

- access denied
- forbidden
- read only
- cannot open the file
- not allowed to save changes

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

- SQL works only if we list the exact phrases we expect.
- If the user describes the same issue with different wording then SQL becomes longer and more fragile.
- LanceDB can retrieve permission-related tickets even when the wording changes from `forbidden` to `cannot save` or `not allowed`.

### Teaching point

This is where students usually see the real difference.

The vector database is not searching for one exact phrase.
It is searching for tickets that mean the same thing.

---

## Short Summary To Say Out Loud

- SQL asks: which rows satisfy these exact rules?
- LanceDB asks: which rows mean something similar to this request?

## One Extra SQL Example

If you want to show where SQL is clearly better run this:

```sql
SELECT category, COUNT(*) AS total
FROM documents
GROUP BY category
ORDER BY total DESC;
```

Use it to explain:

- SQL is better for counting grouping sorting joins and strict filtering.
- LanceDB is better for finding related meaning across different wording.