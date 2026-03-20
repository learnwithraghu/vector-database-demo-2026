# ============================================================
# SIMPLE S3 VECTORS CONNECTION DEMO (boto3 s3vectors only)
# ============================================================
# Uncomment / comment sections to match your live walkthrough.
# ============================================================

# import boto3

# ------------------------------------------------------------
# STEP 1: AWS CONFIG
# ------------------------------------------------------------
# AWS_ACCESS_KEY_ID = ""
# AWS_SECRET_ACCESS_KEY = ""
# AWS_REGION = "us-east-1"
# BUCKET_NAME = "airline-policy-vectors-YOURNAME"
# VECTOR_INDEX_NAME = "airline-policy-index"

# print("STEP 1 OK: Config loaded (region=%s, bucket=%s)." % (AWS_REGION, BUCKET_NAME))


# ------------------------------------------------------------
# STEP 2: CREATE S3 VECTORS CLIENT (not classic s3)
# ------------------------------------------------------------
# session = boto3.Session(
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=AWS_REGION,
# )
# client = session.client("s3vectors")
# print("STEP 2 OK: boto3 client for service 's3vectors' created.")


# ------------------------------------------------------------
# STEP 3: LIST VECTOR BUCKETS
# ------------------------------------------------------------
# print("STEP 3: Calling list_vector_buckets() …")
# response = client.list_vector_buckets()
# bucket_entries = response.get("vectorBuckets") or []
# names = [b.get("vectorBucketName") for b in bucket_entries if b.get("vectorBucketName")]
# print("STEP 3 OK: Found %d vector bucket(s)." % len(names))
# if names:
#     print("STEP 3: Vector bucket names (up to first 10):")
#     for n in names[:10]:
#         print("  - %s" % n)
# else:
#     print("STEP 3: No buckets returned (empty list). Check Region and IAM s3vectors:ListVectorBuckets.")


# ------------------------------------------------------------
# STEP 4: CONFIRM YOUR BUCKET APPEARS IN THE LIST
# ------------------------------------------------------------
# print("STEP 4: Checking whether configured BUCKET_NAME is in list_vector_buckets result …")
# if BUCKET_NAME in names:
#     print("STEP 4 OK: '%s' is visible to this IAM principal." % BUCKET_NAME)
# else:
#     print(
#         "STEP 4 NOTE: '%s' not in this page of results (wrong name/Region/account, "
#         "or pagination). Names we saw: %s"
#         % (BUCKET_NAME, names[:5] or "(none)")
#     )


# ------------------------------------------------------------
# STEP 5 (optional): LIST INDEXES IN YOUR VECTOR BUCKET
# ------------------------------------------------------------
# print("STEP 5: Calling list_indexes(vectorBucketName=%r) …" % BUCKET_NAME)
# idx_resp = client.list_indexes(vectorBucketName=BUCKET_NAME)
# indexes = idx_resp.get("indexes") or []
# index_names = [i.get("indexName") for i in indexes if i.get("indexName")]
# print("STEP 5 OK: Found %d index(es) in this bucket." % len(index_names))
# if index_names:
#     print("STEP 5: Index names:")
#     for n in index_names:
#         print("  - %s" % n)
# if VECTOR_INDEX_NAME in index_names:
#     print("STEP 5 OK: Expected index %r is present." % VECTOR_INDEX_NAME)
# else:
#     print("STEP 5 NOTE: Expected index %r not listed (create it in console or fix name)." % VECTOR_INDEX_NAME)


# ------------------------------------------------------------
# STEP 6 (optional): RAW JSON FOR DEBUGGING
# ------------------------------------------------------------
# import json
# print("STEP 6: Raw list_vector_buckets response (pretty-printed):")
# print(json.dumps(response, default=str, indent=2))
