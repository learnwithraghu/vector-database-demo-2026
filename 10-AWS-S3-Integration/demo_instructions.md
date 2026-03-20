## Demo: AWS S3 Vector Buckets (Cloud Scale)

### Goal

Connect to an AWS S3 **Vector Bucket** using a small connection test script, then use a second script to **put** toy embeddings into a vector index and **query** them, with most steps commented so you can uncomment and re-run one stage at a time.

---

### 0. Pre-flight check

- **AWS account** with S3 Vectors available in your chosen Region.
- **IAM**: You can create policies and attach them to an IAM **user** (or role) used for this lab.
- **Python** 3.9+ and a virtual environment (see below).
- **boto3**: Use a recent version that includes the **`s3vectors`** service client (upgrade if `Unknown service: 's3vectors'`).

---

### 1. Create a Vector Bucket and index (console)

Explain that **storage becomes the vector store**:

- **Classic S3**: objects + optional analytics; no native ANN search over embeddings.
- **Vector buckets**: S3 Vectors exposes APIs (`s3vectors:*`) to **index** and **query** vectors in place.

Walkthrough:

1. **S3** → **Create bucket** → choose **Vector bucket** where offered.
2. Name e.g. `airline-policy-vectors-<your-name>`, Region e.g. `us-east-1`.
3. Open the bucket → **Indexes** → **Create vector index**:
   - **Index name**: `airline-policy-index`
   - **Dimensions**: `3` (toy 3-D vectors in the scripts)
   - **Distance metric**: **Cosine**
4. Wait until the index is **Active**.

You only ingest a handful of vectors in this demo; clean up when finished.

---

### 2. IAM: policy to attach to the lab user (CLI + boto3)

Whether students use **AWS CLI** (`aws s3vectors …` where available) or **boto3** `client("s3vectors")`, authorization is the same: IAM must allow the **`s3vectors`** API actions. That is **separate** from classic S3 object APIs (`s3:GetObject`, etc.).

#### Why this policy is needed even if the user has `AmazonS3FullAccess`

- **`AmazonS3FullAccess` only grants `s3:*`**. It does **not** include **`s3vectors:*`**. Vector buckets are served by the **S3 Vectors** control/data plane, which evaluates **`s3vectors:`** actions.
- **Different resource ARN space**: Vector buckets and indexes are not authorized with the same resource patterns as object buckets unless you explicitly add `s3vectors` statements.
- **Principle of least privilege**: For teaching, you can start broad, then replace `Resource` with specific vector-bucket and index ARNs.

#### Example customer-managed policy (replace placeholders)

1. IAM → **Policies** → **Create policy** → JSON.
2. Replace `REGION`, `ACCOUNT_ID`, and bucket/index prefixes with yours.
3. **Attach** the policy to the IAM **user** (or group/role) whose access keys you use in the scripts.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3VectorsDemoReadWrite",
      "Effect": "Allow",
      "Action": [
        "s3vectors:ListVectorBuckets",
        "s3vectors:GetVectorBucket",
        "s3vectors:ListIndexes",
        "s3vectors:GetIndex",
        "s3vectors:PutVectors",
        "s3vectors:QueryVectors",
        "s3vectors:GetVectors"
      ],
      "Resource": [
        "arn:aws:s3vectors:REGION:ACCOUNT_ID:vector-bucket/airline-policy-vectors-*",
        "arn:aws:s3vectors:REGION:ACCOUNT_ID:index/airline-policy-vectors-*/*"
      ]
    }
  ]
}
```

**Why `s3vectors:GetVectors` is listed:** `QueryVectors` with `returnMetadata=true` (used in the demo so students see stored text) also requires **`GetVectors`** per the API permission notes. If you omit metadata from the query, you can drop `GetVectors` from the policy for a stricter lab.

**Broader (workshop-only) variant:** temporarily use `"Resource": "*"` **only** with the narrow `Action` list above, then tighten ARNs before production or shared accounts.

**CLI:** After attach, run a read-only check with the same principal, for example:

```bash
aws s3vectors list-vector-buckets --region us-east-1
```

(Exact CLI subcommands and availability depend on your AWS CLI version; if a command is missing, use boto3 in the provided scripts—the IAM actions are unchanged.)

---

### 3. Stage A — Connection test (`connect_test_s3_vector.py`)

This script uses **only** `boto3.client("s3vectors")` (no classic `s3` client):

- **`list_vector_buckets`** — proves credentials and `s3vectors:ListVectorBuckets`.
- Optional (commented): **`list_indexes`** — proves access to your bucket’s indexes once the bucket name matches.

#### Virtual environment

From `10-AWS-S3-Integration`:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip3 install boto3
```

Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and `BUCKET_NAME` in the script (or switch to environment variables / profiles for production).

```bash
python connect_test_s3_vector.py
```

Use the printed summary to confirm the lab user sees the expected vector bucket before moving on.

---

### 4. Stage B — Put + query (`load_and_query_embedding.py`)

Uses **only** the **`s3vectors`** client:

- **`put_vectors`** — upload toy vectors (`key`, `data.float32`, optional `metadata`).
- **`query_vectors`** — ANN search with `queryVector.float32`, `topK`, optional `returnMetadata` / `returnDistance`.

The file **`airline_security_policy.txt`** is the sample document; paragraphs are turned into short metadata text for the query results.

**Flow:** Uncomment **STAGE 1**, run, then **STAGE 2**, run, and so on through **STAGE 6**, re-running after each reveal so the printed output matches what you explain.

```bash
python load_and_query_embedding.py
```

---

### 5. Cleanup

- Remove or rotate embedded access keys in the scripts after the session.
- `deactivate` the venv; delete test vectors or the index/bucket if policy allows.

---

### Reference: IAM policy summary (copy-paste checklist)

| Action | Typical use in this demo |
|--------|---------------------------|
| `ListVectorBuckets` | Connection test; CLI list |
| `ListIndexes` | Optional: list indexes in your vector bucket |
| `GetVectorBucket` / `GetIndex` | Optional read APIs / console-style checks |
| `PutVectors` | Ingest toy embeddings |
| `QueryVectors` | Similarity search |
| `GetVectors` | Required when returning **metadata** (or filters) on query |
