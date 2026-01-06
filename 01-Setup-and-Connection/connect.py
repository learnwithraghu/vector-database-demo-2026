from qdrant_client import QdrantClient

def main():
    # 1. Connect to the local Qdrant instance
    client = QdrantClient(url="http://localhost:6333")
    
    print("✅ Connection established successfully!")

    # 2. List all collections (Should be empty initially)
    collections = client.get_collections()
    print(f"📚 Current Collections: {collections}")

if __name__ == "__main__":
    main()
