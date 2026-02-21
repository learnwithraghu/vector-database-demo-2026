from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
import os

def main():
    # 1. Initialize Encoder and Client
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    client = QdrantClient(url="http://localhost:6333")
    
    collection_name = "airline_policies_semantic"

    # 2. Re-create collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    # Vector size 384 for all-MiniLM-L6-v2
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 3. Load Data from Shared File
    # Robust path handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../datasets/airline_policy_dataset.json")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find {file_path}")
        return

    with open(file_path, "r") as f:
        documents = json.load(f)

    # 4. Upload
    print("Encoded & Uploading Airline Policies...")
    points = []
    for idx, doc in enumerate(documents):
        # Enhance: Encode the policy text
        vector = model.encode(doc["text"]).tolist()
        
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload=doc # Store original text and metadata
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Uploaded {len(documents)} policy documents.")

    # 5. Search
    # Query: "Can I bring my cat?"
    # Problem: Without filtering, we get conflicting rules from different airlines.
    query = "Can I bring my cat?"
    print(f"\n🔎 User Query: '{query}'")
    
    query_vector = model.encode(query).tolist()
    hits = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3
    ).points

    print("\n--- Search Results (The 'Before' State) ---")
    for hit in hits:
        print(f"[{hit.payload['metadata']['airline']} - {hit.payload['metadata']['class']}]")
        print(f"Policy: {hit.payload['text']}\n")
        
    print("⚠️  PROBLEM IDENTIFIED:")
    print("The user asked a simple question, but got conflicting answers.")
    print("1. SkyStream says Yes (in cabin).")
    print("2. GlobalAir says No (strictly prohibited).")
    print("-> Without context (Metadata), the AI is confusing! We need to move to Step 04.")
    print("\n👉 ACTION: Go to the Search App '1. Semantic Search' to visualize this confusion!")

if __name__ == "__main__":
    main()
