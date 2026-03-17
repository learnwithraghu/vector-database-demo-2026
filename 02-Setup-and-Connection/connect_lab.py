# **FILL THE LINE**

def main():
    # 1. Connect to the local Qdrant instance
    # **FILL THE WORD**
    client = QdrantClient(url=FILL_URL)

    print("✅ Connection established successfully!")

    # 2. List all collections (Should be empty initially)
    # **FILL THE WORD**
    collections = client.FILL_METHOD()
    print(f"📚 Current Collections: {collections}")


if __name__ == "__main__":
    main()
