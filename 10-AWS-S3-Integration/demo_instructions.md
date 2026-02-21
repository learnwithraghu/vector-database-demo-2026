# Demo 7: AWS S3 Vector Buckets (Cloud Scale)

## 🎯 The Goal: Infinite Scale
We have built vector apps on our local machine (Docker). That's great for development.
But what if you have **1 Billion Vectors**?
Managing a cluster of servers is hard. AWS S3 Vectors allows you to store and search vectors directly in S3, leveraging the specific "Vector Bucket" type for serverless scale.

*   **Previous Step**: Admin Operations (Local).
*   **Next Step**: Course Completion.

## 🛠️ Pre-flight Check
**Prerequisites**:
1.  An active AWS Account.
2.  AWS CLI configured locally (optional, for the Python script).
3.  `boto3` installed (`pip install boto3`).

## 📝 Steps for the Instructor

### 1. The Concept: "Storage IS the Database"
*   Traditionally, S3 is just for "files".
*   If you wanted to search them, you had to move them to a Database (like Qdrant/Pinecone).
*   **S3 Vectors** blurs this line. The bucket *understands* vectors.
*   **Benefits**:
    *   **Cost**: S3 storage is cheap.
    *   **Scale**: No nodes to manage.

### 2. Manual Setup (AWS Console)
*Guide the students through the AWS Management Console.*

1.  **Navigate to S3**: Go to the S3 Dashboard.
2.  **Create Bucket**:
    *   Click "Create bucket".
    *   **Important**: Select **"Vector Bucket"** (if available in your region) or standard bucket with "Vector Indexing" enabled.
    *   Name: `airline-policy-vectors-[your-name]`.
3.  **Define Index**:
    *   Go to the "Indexes" tab of your new bucket.
    *   Click "Create Vector Index".
    *   **Algorithm**: HNSW (Hierarchical Navigable Small World).
    *   **Dimensions**: 384 (matching our embedding model).
    *   **Distance Metric**: Cosine.
4.  **Ready**: Once the index status is "Active", we can push data.

### 3. Code Walkthrough (`s3_vector_demo.py`)

This script mimics the behavior of interacting with the S3 Vector API.

#### Phase 1: Authentication
*   We use `boto3` to talk to AWS.
*   Ensure your environment variables (`AWS_ACCESS_KEY_ID`, etc.) are set.

#### Phase 2: Upsert (PutItem)
*   Instead of `client.upsert()`, we might use `s3.put_object()` with special metadata headers or a devoted `s3-vector` client method.
*   *Note*: In this demo, we use a simplified hypothetical API structure for clarity: `put_vector_item`.

#### Phase 3: Search (Query)
*   We send a vector to the bucket endpoint.
*   AWS computes the similarity and returns the S3 keys (filenames) of the closest matches.

### 4. Running the Demo
```bash
# 1. Install AWS SDK
pip install boto3

# 2. Run the script
python s3_vector_demo.py
```

## ⚠️ Note on Cost
S3 Vectors is serverless, meaning you pay for **storage** and **requests**. It is generally cheaper than running EC2 instances 24/7, but always clean up your specific test buckets after the workshop!
