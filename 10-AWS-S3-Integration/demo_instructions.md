## Demo: AWS S3 Vector Buckets (Cloud Scale)

### 🎯 Goal
Connect to an AWS S3 **Vector Bucket** using one simple connection test script, then use a second script to upload + query a realistic airline policy document, with the embedding/query parts commented so you can reveal them step‑by‑step.

---

### 0. Pre‑flight Check

- **AWS account**: You must have an active AWS account.
- **Permissions**: Ability to create S3 buckets and objects.
- **Python**: Python 3.9+ installed on your laptop.
- **boto3**: Will be installed inside a virtual environment (see below).

---

### 1. Steps to Create an S3 Vector Bucket (Console)

Explain that **storage becomes the database**:
- **Traditionally**: S3 holds files, databases (Qdrant, Pinecone, etc.) hold vectors.
- **With Vector Buckets**: S3 can natively index and search vectors stored inside it.

Walk students through the console:

1. **Open S3 console**
   - Go to AWS Management Console → `S3`.
2. **Create Vector Bucket**
   - Click **Create bucket**.
   - Choose a region (e.g. `us-east-1`) where **Vector Bucket** is available.
   - Bucket name: `airline-policy-vectors-<your-name>`.
   - Under bucket type / features, select **Vector bucket** (or enable vector indexing if shown separately).
   - Create the bucket.
3. **Create Vector Index**
   - Open the new bucket.
   - Go to the **Indexes** (or **Vector indexes**) tab.
   - Click **Create vector index**.
   - **Index name**: `airline-policy-index`.
   - **Dimensions**: `3` (we use tiny 3‑D toy vectors in the demo so students can follow the math).
   - **Distance metric**: `Cosine`.
   - Create the index and wait until its status is **Active**.

Tell students: **we will only write a handful of objects**, and we will clean them up at the end.

---

### 2. Stage 2 – Python: Connection Test Only (`connect_test_s3_vector.py`)

You will use `connect_test_s3_vector.py` as a **small, fully runnable script**:
- Code is **not commented** – you can run it end‑to‑end to prove the connection.
- It only:
  - creates a simple S3 client with demo AWS credentials, and
  - calls `list_objects_v2` on your Vector Bucket to validate connectivity.

#### 2.1 Create and Activate a Python Virtual Environment

From the `10-AWS-S3-Integration` directory:

```bash
# create venv
python3 -m venv .venv

# macOS / Linux: activate
source .venv/bin/activate

# (optional) Windows PowerShell:
# .venv\Scripts\Activate.ps1

# install dependencies inside the venv
pip3 install --upgrade pip
pip3 install boto3
```

Remind students:
- The **prompt will change** (usually shows `(.venv)`).
- All Python packages for this demo stay isolated inside `.venv`.

#### 2.2 Walkthrough of `connect_test_s3_vector.py`

Open `connect_test_s3_vector.py` and highlight:

1. **AWS constants + bucket name**
   - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
   - `BUCKET_NAME` set to `airline-policy-vectors-<your-name>`.
2. **`get_s3_client()`**
   - Uses `boto3.Session` with the constants.
   - Prints a clear message when the client is created.
3. **`validate_vector_bucket_connection(client)`**
   - Calls `list_objects_v2(Bucket=BUCKET_NAME, MaxKeys=5)`.
   - On HTTP 200:
     - prints a success message, and
     - either shows a few object keys (if any) or states that the bucket is empty but reachable.
   - On errors:
     - prints clear ❌ messages (e.g. bucket missing or credentials invalid).

Run the script from the activated venv:

```bash
python connect_test_s3_vector.py
```

Use the printed output to confirm the Vector Bucket + credentials are correct before moving to embeddings.

---

### 3. Stage 3 – Upload a Policy Document to the Vector Bucket and Query It (`load_and_query_embedding.py`)

In Stage 3 you switch to the second script: `load_and_query_embedding.py`.
This script:
- has **active connection code** at the top (so it can run), and
- keeps the embedding / upload / query logic as **commented blocks** that you will reveal step‑by‑step.

We also ship a ready‑made document file: `airline_security_policy.txt` in the same folder.

The flow for this stage:
- Use the **already‑generated** `airline_security_policy.txt` as your demo policy.
- Gradually uncomment code in `load_and_query_embedding.py` to:
  - load and embed the local document,
  - upload text + vectors to S3, and
  - run a cosine‑similarity query to retrieve the most relevant section.

#### 3.1 Walkthrough of `load_and_query_embedding.py`

Open `load_and_query_embedding.py` and show students:

1. **Imports + credentials + S3 client (uncommented, already runnable)**
   - `boto3`, `json`, `math`, `Path`.
   - AWS constants and `BUCKET_NAME` (same as in the connection test).
   - `get_s3_client()` prints that it is creating an S3 client for this demo.
2. **STAGE 1 – `load_local_policy_paragraphs()` (commented)**
   - Reads `airline_security_policy.txt` from disk.
   - Splits the text into paragraphs.
   - Assigns a tiny 3‑D vector to each paragraph:
     - \( \text{dimension}_1 = \) security
     - \( \text{dimension}_2 = \) data / compliance
     - \( \text{dimension}_3 = \) operations
   - Prints how many paragraphs were loaded.
3. **STAGE 2 – `upload_policy_embeddings(client)` (commented)**
   - Uploads the full policy text to:
     - `documents/airline_security_policy.txt`
   - Uploads each paragraph + vector as JSON under:
     - `policy_vectors/section_0.json`, `section_1.json`, etc.
   - Uses clear `[STAGE 2]` print statements for each upload.
4. **STAGE 3 – `cosine_similarity()` + `query_policy_embeddings(client)` (commented)**
   - Defines a simple cosine similarity function.
   - Uses a security‑heavy query vector to represent:
     - “How do we train cabin crew on security?”
   - Lists all `policy_vectors/` objects, downloads each, computes similarity, and prints the top matches with `[STAGE 3]` messages.

At the bottom, the `main()` function:
- creates the S3 client and prints that the demo is ready,
- contains commented example calls you will uncomment in order:
  - `load_local_policy_paragraphs()`
  - `upload_policy_embeddings(client)`
  - `query_policy_embeddings(client)`

Run from the activated venv:

```bash
python load_and_query_embedding.py
```

At first, it will just test connection and tell you to uncomment the STAGE blocks.
Then, as you uncomment each stage, re‑run to show students each step of the pipeline.

Finally, **clean up credentials and code**:
- Comment out or remove your AWS keys from both Python scripts once the workshop is over.
- Deactivate the virtual environment:

```bash
deactivate
```

- Optionally delete the `.venv` directory and/or this demo folder once you are finished teaching.

This keeps **student AWS accounts safe** and ensures no stray resources or credentials remain.

---

### 4. IAM Policy Example for S3 Vectors

S3 Vectors uses a **separate IAM namespace** (`s3vectors:*`) from classic S3 (`s3:*`).
Having `AmazonS3FullAccess` is **not enough** to call `ListVectorBuckets`, `PutVectors`, or `QueryVectors`.

Here is an example **inline / customer‑managed policy** you can attach to the IAM user or role that runs this demo (adjust ARNs to your account, region, and bucket/index names):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3vectors:ListVectorBuckets",
        "s3vectors:GetVectorBucket",
        "s3vectors:ListIndexes",
        "s3vectors:GetIndex",
        "s3vectors:PutVectors",
        "s3vectors:QueryVectors"
      ],
      "Resource": [
        "arn:aws:s3vectors:us-east-1:666234783044:vector-bucket/airline-policy-vectors-*",
        "arn:aws:s3vectors:us-east-1:666234783044:index/airline-policy-vectors-*/*"
      ]
    }
  ]
}
```

You can start with a broader `Resource` if needed while experimenting, then tighten it later for production.
