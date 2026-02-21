from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer
import json
import os

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "airline_policies_semantic" 
    
    # ensure collection exists (populated by previous step)
    if not client.collection_exists(collection_name):
         print(f"❌ Collection {collection_name} not found. Run 04-semantic-search/semantic_search.py first!")
         return
         
    # --- The Problem Recap ---
    print("\n--- 1. Recap: The Problem (Mixed Results) ---")
    query = "Can I bring my cat?"
    print(f"Query: '{query}'")
    
    # --- The Solution ---
    print("\n--- 2. Search with Metadata Filter ---")
    print("Scenario: User is flying 'SkyStream' in 'Economy'.")
    print("Filter: airline='SkyStream' AND class='Economy'")
    
    query_filter = Filter(
        must=[
            FieldCondition(key="metadata.airline", match=MatchValue(value="SkyStream")),
            FieldCondition(key="metadata.class", match=MatchValue(value="Economy"))
        ]
    )
    
    hits = client.query_points(
        collection_name=collection_name,
        query=model.encode(query).tolist(),
        query_filter=query_filter,
        limit=1 
    ).points
    
    for hit in hits:
        print(f"\n✅ Found Perfect Match: [{hit.payload['metadata']['airline']} - {hit.payload['metadata']['class']}]")
        print(f"Policy: {hit.payload['text']}")
        
    print("\n[Success]: We filtered out GlobalAir and Business Class rules. The answer is now unambiguous! 🎯")
    print("\n👉 ACTION: Go to the Search App '2. Context-Aware' to see this filter in action!")

if __name__ == "__main__":
    main()
