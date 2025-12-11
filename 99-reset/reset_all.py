from qdrant_client import QdrantClient
import time

def main():
    print("⚠️  WARNING: This will DELETE ALL COLLECTIONS in your local Qdrant instance.")
    print("⚠️  Use this to reset the environment for a fresh run of the course.")
    
    confirm = input("Are you sure? (Type 'yes' to confirm): ")
    if confirm.lower() != "yes":
        print("❌ Aborted.")
        return

    client = QdrantClient(url="http://localhost:6333")
    
    try:
        # Get all collections
        collections = client.get_collections().collections
        
        if not collections:
            print("✅ Qdrant is already empty. No collections found.")
            return

        print(f"\n🔍 Found {len(collections)} collections. Deleting...")
        
        for collection in collections:
            print(f"   🗑️  Deleting '{collection.name}'...")
            client.delete_collection(collection.name)
            
        print("\n✨ All collections deleted.")
        print("✅ Environment Reset! You can now start from Section 01.")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        print("Is Docker running?")

if __name__ == "__main__":
    main()
