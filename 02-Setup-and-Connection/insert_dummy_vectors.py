from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def main() -> None:
    # 1. Connect to the same local Qdrant instance as in connect.py
    client = QdrantClient(url="http://localhost:6333")

    print("✅ Connected to Qdrant at http://localhost:6333")

    # 2. Create (or overwrite) a small demo collection for this lab
    collection_name = "demo_vectors"

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )
    print(f"🗂️  Collection ready: {collection_name}")

    # 3. Insert a few dummy vectors with simple payloads
    points = [
        PointStruct(
            id=1,
            vector=[0.1, 0.2, 0.3, 0.4],
            payload={"label": "first", "group": "A"},
        ),
        PointStruct(
            id=2,
            vector=[0.2, 0.1, 0.4, 0.3],
            payload={"label": "second", "group": "A"},
        ),
        PointStruct(
            id=3,
            vector=[0.9, 0.8, 0.7, 0.6],
            payload={"label": "third", "group": "B"},
        ),
    ]

    client.upsert(collection_name=collection_name, points=points)
    print(f"📥 Inserted {len(points)} dummy vectors into '{collection_name}'.")

    # 4. Quick sanity check: scroll a few points back
    records, _ = client.scroll(
        collection_name=collection_name, limit=5, with_payload=True
    )
    print("🔎 Sample records from collection:")
    for r in records:
        print(r)


if __name__ == "__main__":
    main()

