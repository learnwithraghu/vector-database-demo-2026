from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def main():
    print("--- Demo: Hello Vector World (RGB Colors) ---")
    
    # 0. Connect to Qdrant
    # Make sure Docker is running: docker run -p 6333:6333 qdrant/qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    collection_name = "rgb_colors"

    # 1. Re-create the collection (Fresh start)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    
    # Vector size 3 for Red, Green, Blue
    # Dimensions: [Red, Green, Blue] (Normalized 0.0 to 1.0)
    # Example: Pure Red is [1.0, 0.0, 0.0]
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3, distance=Distance.COSINE),
    )
    print(f"Created collection: {collection_name}")

    # 2. Upsert (Insert) Colors
    # We normalized 255 to 1.0 for simplicity, but vectors can be any magnitude.
    # Color Palette:
    # - Red:   [1.0, 0.0, 0.0]
    # - Green: [0.0, 1.0, 0.0]
    # - Blue:  [0.0, 0.0, 1.0]
    # - Yellow:[1.0, 1.0, 0.0] (Red + Green)
    
    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[
            PointStruct(id=1, vector=[1.0, 0.05, 0.05], payload={"color": "Red"}),
            PointStruct(id=2, vector=[0.05, 1.0, 0.05], payload={"color": "Green"}), 
            PointStruct(id=3, vector=[0.05, 0.05, 1.0], payload={"color": "Blue"}),
            PointStruct(id=4, vector=[1.0, 1.0, 0.05],  payload={"color": "Yellow"}),
        ],
    )
    print("Upserted 4 colors: Red, Green, Blue, Yellow.")

    # 3. Search
    # "I have a color that is mostly Red, but a little bit Green. What is it closest to?"
    # Search Vector: Dark Orange-ish [1.0, 0.2, 0.0]
    
    search_vector = [1.0, 0.2, 0.0]
    print(f"\nSearching for vector {search_vector} (Reddish)...")
    
    results = client.query_points(
        collection_name=collection_name,
        query=search_vector,
        limit=2
    ).points

    print("Results:")
    for hit in results:
        print(f" - Found: {hit.payload['color']} (Score: {hit.score:.4f})")

    # Ideally: 
    # 1. Red should be #1 (Closest match)
    # 2. Yellow might be #2 (Since it has a lot of red)
    # 3. Blue/Green should be far away.

if __name__ == "__main__":
    main()
