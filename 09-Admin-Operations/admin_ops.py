from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, CreateAliasOperation, CreateAlias

def main():
    client = QdrantClient(url="http://localhost:6333")
    coll_name = "airline_app_v1"
    
    # --- PHASE 1: Seeding Dummy Data ---
    if client.collection_exists(coll_name):
        client.delete_collection(coll_name)
    client.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)
    )
    
    client.upsert(
        collection_name=coll_name,
        points=[
            PointStruct(id=1, vector=[0.1, 0.1, 0.1, 0.1], payload={"msg": "Data 1"}),
            PointStruct(id=2, vector=[0.2, 0.2, 0.2, 0.2], payload={"msg": "Data 2"}),
        ]
    )
    print(f"✅ Collection '{coll_name}' created and seeded.")

    # --- PHASE 2: Inspection (Admin View) ---
    info = client.get_collection(collection_name=coll_name)
    print(f"\n📊 Collection Stats:")
    print(f"   - Status: {info.status}")         # e.g., Green
    print(f"   - Vectors: {info.points_count}") # Should be 2
    print(f"   - Config: {info.config.params}") 

    # --- PHASE 3: Aliases (Zero-Downtime) ---
    alias_name = "production_alias"
    
    print(f"\n🔄 Linking alias '{alias_name}' -> '{coll_name}'...")
    client.update_collection_aliases(
        change_aliases_operations=[
            CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=coll_name,
                    alias_name=alias_name
                )
            )
        ]
    )
    
    # Prove it works: Search via Alias
    results = client.query_points(
        collection_name=alias_name, # <--- We use the ALIAS, not the real name
        query=[0.1, 0.1, 0.1, 0.1],
        limit=1
    ).points
    print(f"   Search via Alias returned: {len(results)} result(s). (Success!)")

    # --- PHASE 4: Snapshots (Backups) ---
    print(f"\n📸 Creating Snapshot...")
    snapshot_info = client.create_snapshot(collection_name=coll_name)
    print(f"   Backup saved: {snapshot_info.name}")
    print(f"   Creation time: {snapshot_info.creation_time}")

if __name__ == "__main__":
    main()
