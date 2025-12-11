from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def main():
    client = QdrantClient(url="http://localhost:6333")
    
    collection_name = "airline_policies"

    # 1. Re-create the collection (Fresh start)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    # Vector size 4 (arbitrary for this demo)
    # Dimensions: [Comfort, Price, Speed, Service]
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    print(f"Created collection: {collection_name}")

    # 2. Upsert (Insert/Update) Points
    # Dimensions: [Comfort, Price, Speed, Service]
    # Scale 0.0 to 1.0
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            PointStruct(id=1, vector=[0.9, 0.4, 0.8, 0.9], payload={"airline": "SkyStream", "class": "Business"}),
            PointStruct(id=2, vector=[0.5, 0.9, 0.8, 0.5], payload={"airline": "SkyStream", "class": "Economy"}), 
            PointStruct(id=3, vector=[0.2, 0.8, 0.7, 0.3], payload={"airline": "GlobalAir", "class": "Economy"}),
        ],
    )
    print("Upserted 3 airline service profiles.")

    # 3. Search
    # "Find me an option with High Comfort and High Service" 
    # Search Vector: [1.0, 0.0, 0.0, 1.0] 
    search_vector = [1.0, 0.0, 0.0, 1.0]
    
    results = client.query_points(
        collection_name=collection_name,
        query=search_vector,
        limit=2
    ).points

    print("\nSearch Results (Closest to 'High Comfort & Service'):")
    for hit in results:
        print(f"Option: {hit.payload['airline']} ({hit.payload['class']}), Score: {hit.score}")

if __name__ == "__main__":
    main()
