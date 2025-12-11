import boto3
import json
import os
import time

# NOTE: This script requires valid AWS Credentials configured.
# You can set them via:
# export AWS_ACCESS_KEY_ID=...
# export AWS_SECRET_ACCESS_KEY=...
# export AWS_REGION=us-east-1

def get_s3_client():
    """
    Initialize the boto3 client for S3.
    """
    try:
        # We assume a standard session. 
        # For S3 Vectors, ensure your boto3 version is up to date.
        session = boto3.Session()
        client = session.client('s3')
        return client
    except Exception as e:
        print(f"Error connecting to AWS: {e}")
        return None

def upsert_vectors(client, bucket_name, index_name, vectors):
    """
    Upload vectors to the S3 Bucket.
    In a real scenario, this might involve putting a Parquet file 
    or using a specific 'put_record' API depending on the exact integration 
    (e.g., OpenSearch ingestion or direct S3 Vector Put).
    
    Here we simulate a direct PUT key-value pattern for demonstration.
    """
    print(f"🚀 Uploading {len(vectors)} vectors to {bucket_name}...")
    
    for i, vec in enumerate(vectors):
        # We store the vector as a JSON object in S3
        # The 'Vector Index' on the bucket (configured in Console) 
        # watches for these files.
        key = f"vectors/doc_{i}.json"
        body = json.dumps({
            "id": i,
            "values": vec,
            "metadata": {"category": "demo", "timestamp": time.time()}
        })
        
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=body,
            ContentType='application/json'
        )
        print(f"   [+] Uploaded {key}")

def search_vectors(client, bucket_name, query_vector):
    """
    Perform a semantic search.
    Note: As of today, querying often happens via a partner service (Bedrock/OpenSearch)
    that reads this bucket. 
    
    This function simulates a hypothetical direct S3 Select-like query for vectors.
    """
    print("\n🔎 Searching for nearest neighbors...")
    
    # Hypothetical API call structure for S3 Select Vector extension
    # response = client.select_object_content(
    #     Bucket=bucket_name,
    #     Key='index_manifest',
    #     ExpressionType='SQL',
    #     Expression=f"SELECT * FROM S3Object s WHERE VECTOR_DISTANCE(s.vector, {query_vector}) < 0.5"
    # )
    
    # For demo purposes, we will mock the response
    print("   -> Sending query to AWS Compute layer...")
    time.sleep(1) # Simulate network latency
    
    results = [
        {"id": 2, "score": 0.92, "metadata": {"category": "demo"}},
        {"id": 0, "score": 0.85, "metadata": {"category": "demo"}}
    ]
    
    print("✅ Search Results:")
    for res in results:
        print(f"   ID: {res['id']} | Score: {res['score']} | Meta: {res['metadata']}")

def main():
    BUCKET_NAME = "airline-policy-vectors-demo" # Replace with your bucket
    INDEX_NAME = "course-index-01"
    
    # 1. Connect
    client = get_s3_client()
    if not client:
        return

    # 2. Dummy Data (3 dimensions for simplicity in print, but 384 in reality)
    # [Sci-Fi, Comedy, Action]
    vectors = [
        [0.9, 0.1, 0.9], # Action Movie
        [0.1, 0.9, 0.2], # Comedy
        [0.8, 0.2, 0.7], # Another Action
    ]
    
    # 3. Upload
    upsert_vectors(client, BUCKET_NAME, INDEX_NAME, vectors)
    
    # 4. Search
    query = [1.0, 0.0, 1.0] # Pure Action
    search_vectors(client, BUCKET_NAME, query)

if __name__ == "__main__":
    main()
