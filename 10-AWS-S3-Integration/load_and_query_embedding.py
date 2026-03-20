# ============================================================
# LOAD LOCAL TEXT -> TOY EMBEDDINGS -> S3 VECTORS PUT / QUERY
# boto3 service client: "s3vectors" ONLY (no classic s3 client)
# ============================================================
# Uncomment one stage at a time, re-run, and narrate the prints.
# ============================================================

print("=" * 60)
print("S3 Vectors demo script — uncomment STAGES 1→2→… in order, then re-run.")
print("=" * 60)


# ------------------------------------------------------------
# STAGE 1: IMPORT + AWS CONFIG + INDEX INFO
# ------------------------------------------------------------
#
# import boto3
# from pathlib import Path
#
# AWS_ACCESS_KEY_ID = "REPLACE_ME_WITH_YOUR_ACCESS_KEY_ID"
# AWS_SECRET_ACCESS_KEY = "REPLACE_ME_WITH_YOUR_SECRET_ACCESS_KEY"
# AWS_REGION = "us-east-1"
#
# BUCKET_NAME = "airline-policy-vectors-YOURNAME"
# VECTOR_INDEX_NAME = "airline-policy-index"
#
# print("STAGE 1 OK: Config loaded (bucket=%s, index=%s)." % (BUCKET_NAME, VECTOR_INDEX_NAME))


# ------------------------------------------------------------
# STAGE 2: CREATE S3 VECTORS CLIENT
# ------------------------------------------------------------
#
# session = boto3.Session(
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION,
# )
# client = session.client("s3vectors")
# print("STAGE 2 OK: client('s3vectors') ready (same API surface CLI/SDK use).")


# ------------------------------------------------------------
# STAGE 3: LOAD LOCAL POLICY TEXT
# ------------------------------------------------------------
#
# policy_path = Path(__file__).parent / "airline_security_policy.txt"
# policy_text = policy_path.read_text(encoding="utf-8")
# paragraphs = [p.strip() for p in policy_text.split("\n\n") if p.strip()]
# print("STAGE 3 OK: Loaded %d paragraph(s) from %s." % (len(paragraphs), policy_path.name))
# for i, p in enumerate(paragraphs):
#     preview = (p[:80] + "…") if len(p) > 80 else p
#     print("  [%d] %s" % (i, preview.replace("\n", " ")))


# ------------------------------------------------------------
# STAGE 4: BUILD TOY 3-D VECTORS (demo only; not a real embedding model)
# ------------------------------------------------------------
#
# # Axes are only for intuition: [security, compliance, operations]
# demo_vectors = [
#     [0.9, 0.2, 0.4],
#     [0.3, 0.9, 0.3],
#     [0.3, 0.3, 0.9],
# ]
#
# limit = min(len(paragraphs), len(demo_vectors))
# print(
#     "STAGE 4 OK: Pairing first %d paragraphs with toy vectors (index dimension must stay 3)."
#     % limit
# )


# ------------------------------------------------------------
# STAGE 5: PUT VECTORS (s3vectors:PutVectors)
# ------------------------------------------------------------
#
# vectors_payload = []
# for i in range(limit):
#     vectors_payload.append(
#         {
#             "key": "section-%d" % i,
#             "data": {"float32": demo_vectors[i]},
#             "metadata": {"text": paragraphs[i]},
#         }
#     )
#
# print("STAGE 5: Uploading %d vector(s) via put_vectors …" % len(vectors_payload))
# put_response = client.put_vectors(
#     vectorBucketName=BUCKET_NAME,
#     indexName=VECTOR_INDEX_NAME,
#     vectors=vectors_payload,
# )
# print("STAGE 5 OK: put_vectors finished (empty JSON body on success).")
# print("STAGE 5: Raw response: %r" % (put_response,))


# ------------------------------------------------------------
# STAGE 6: QUERY VECTORS (s3vectors:QueryVectors + GetVectors if metadata returned)
# ------------------------------------------------------------
#
# # Query biased toward “security / crew training” on the same 3-D toy axes.
# query_vector = [0.95, 0.3, 0.4]
#
# print("STAGE 6: Query vector (float32): %s" % query_vector)
# print("STAGE 6: Calling query_vectors(topK=3, returnMetadata=True) …")
#
# query_response = client.query_vectors(
#     vectorBucketName=BUCKET_NAME,
#     indexName=VECTOR_INDEX_NAME,
#     topK=3,
#     queryVector={"float32": query_vector},
#     returnMetadata=True,
#     returnDistance=True,
# )
#
# hits = query_response.get("vectors") or []
# metric = query_response.get("distanceMetric")
# print("STAGE 6 OK: Got %d hit(s); distanceMetric=%r" % (len(hits), metric))
# print("STAGE 6: Ranked results (lower distance = closer for cosine/euclidean per index config):")
# for rank, row in enumerate(hits, start=1):
#     key = row.get("key")
#     dist = row.get("distance")
#     meta = row.get("metadata") or {}
#     text = meta.get("text", "")
#     preview = (text[:120] + "…") if len(text) > 120 else text
#     print("  #%d key=%r distance=%s" % (rank, key, dist))
#     print("      metadata.text: %s" % preview.replace("\n", " "))
