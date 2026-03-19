# ============================================================
# LOAD LOCAL TEXT -> EMBED -> PUT TO S3 VECTORS -> QUERY
# NO FUNCTIONS, STEP-BY-STEP FOR LIVE DEMO
# ============================================================
#
# Uncomment one stage at a time and run.
#

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
# print("STAGE 1 OK: Config loaded.")


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
# print("STAGE 2 OK: S3 Vectors client created.")


# ------------------------------------------------------------
# STAGE 3: LOAD LOCAL POLICY TEXT
# ------------------------------------------------------------
#
# policy_path = Path(__file__).parent / "airline_security_policy.txt"
# policy_text = policy_path.read_text(encoding="utf-8")
# paragraphs = [p.strip() for p in policy_text.split("\n\n") if p.strip()]
# print(f"STAGE 3 OK: Loaded {len(paragraphs)} paragraph(s) from local text file.")


# ------------------------------------------------------------
# STAGE 4: CREATE SIMPLE DEMO EMBEDDINGS
# ------------------------------------------------------------
#
# # 3D toy embeddings for demo only: [security, compliance, operations]
# demo_vectors = [
#     [0.9, 0.2, 0.4],
#     [0.3, 0.9, 0.3],
#     [0.3, 0.3, 0.9],
# ]
#
# limit = min(len(paragraphs), len(demo_vectors))
# print(f"STAGE 4 OK: Preparing {limit} vectors to upload.")


# ------------------------------------------------------------
# STAGE 5: PUT VECTORS TO S3 VECTOR INDEX
# ------------------------------------------------------------
#
# vectors_payload = []
# for i in range(limit):
#     vectors_payload.append(
#         {
#             "id": f"section-{i}",
#             "values": demo_vectors[i],
#             "metadata": {
#                 "bucket": BUCKET_NAME,
#                 "text": paragraphs[i],
#             },
#         }
#     )
#
# # NOTE: API field names can vary by SDK version.
# # If your SDK expects different field names, adjust accordingly.
# put_response = client.put_vectors(
#     vectorBucketName=BUCKET_NAME,
#     indexName=VECTOR_INDEX_NAME,
#     vectors=vectors_payload,
# )
#
# print("STAGE 5: put_vectors() response:")
# print(put_response)


# ------------------------------------------------------------
# STAGE 6: QUERY VECTORS FROM S3 VECTOR INDEX
# ------------------------------------------------------------
#
# # Example query: "security training for crew"
# query_vector = [0.95, 0.3, 0.4]
#
# query_response = client.query_vectors(
#     vectorBucketName=BUCKET_NAME,
#     indexName=VECTOR_INDEX_NAME,
#     queryVector=query_vector,
#     topK=3,
# )
#
# print("STAGE 6: query_vectors() response:")
# print(query_response)

