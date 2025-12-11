from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import json
import os

def main():
    # Initialize Models (Explicitly!)
    # 1. Dense Model (Meaning)
    print("⏳ Loading Dense Model (sentence-transformers/all-MiniLM-L6-v2)...")
    dense_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Sparse Model (Keywords)
    print("⏳ Loading Sparse Model (prithivida/Splade_PP_en_v1)...")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    client = QdrantClient(url="http://localhost:6333")
    collection_name = "airline_policies_hybrid"

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    # 3. Create Collection with Named Vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        }
    )
    print("✅ Hybrid Collection created!")

    # 4. Load Data
    # Robust path handling
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../datasets/airline_policy_dataset.json")

    with open(file_path, "r") as f:
        documents = json.load(f)

    print("⏳ Encoding Policies (Generating Dense + Sparse Vectors)...")
    
    # Generate embeddings generator
    texts = [doc["text"] for doc in documents]
    
    # FastEmbed returns generators, convert to list for easy handling in this demo
    dense_embeddings = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for idx, (dense, sparse, doc) in enumerate(zip(dense_embeddings, sparse_embeddings, documents)):
        points.append(models.PointStruct(
            id=idx,
            vector={
                "dense": dense.tolist(),
                "sparse": models.SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist()
                )
            },
            payload=doc
        ))

    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Uploaded {len(points)} policies.")

    # 5. Search
    # Challenge: A specific hidden fee or regulation code.
    # Dataset Text: "The fee is $125 each way."
    # Query: "What is the fee?" (Dense might find general pricing) vs "$125" (Sparse finds exact)
    query_text = "$125"
    print(f"\n🔎 Query: '{query_text}'")

    # Encode Query
    query_dense_vec = list(dense_model.embed([query_text]))[0].tolist()
    query_sparse_obj = list(sparse_model.embed([query_text]))[0]
    query_sparse_vec = models.SparseVector(
        indices=query_sparse_obj.indices.tolist(),
        values=query_sparse_obj.values.tolist()
    )

    print("\n--- A. Dense Only (Semantic) ---")
    print(f"Goal: Find '{query_text}' using meaning.")
    results = client.query_points(
        collection_name=collection_name,
        query=query_dense_vec,
        using="dense",
        limit=1
    ).points
    
    for hit in results:
        print(f"   * Found: '{hit.payload['text'][:60]}...' (Score: {hit.score:.3f})")
    print("[Result]: Dense search struggles with exact numbers or short codes.")

    print("\n--- B. Hybrid Search (RRF Fusion) ---")
    print(f"Goal: Combine 'Meaning' with 'Exact Keyword {query_text}'.")
    
    # Advanced: We configure a Prefetch for each vector type, then Fuse them!
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_dense_vec,
                using="dense",
                limit=10
            ),
            models.Prefetch(
                query=query_sparse_vec,
                using="sparse",
                limit=10
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF), # Reciprocal Rank Fusion
        limit=1
    ).points

    for hit in results:
        print(f"   * Found: '{hit.payload['text'][:60]}...'")
    print("[Success]: Sparse vector latched onto the exact number '$125'. 🎯")
    print("\n👉 ACTION: Go to the Search App '3. Hybrid Search' to try this yourself!")

if __name__ == "__main__":
    main()
