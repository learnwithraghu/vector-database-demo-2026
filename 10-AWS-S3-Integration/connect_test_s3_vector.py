# ============================================================
# SIMPLE S3 VECTORS CONNECTION DEMO
# ============================================================
# ------------------------------------------------------------
# STEP 1: IMPORT + AWS CONFIG
# ------------------------------------------------------------
#
import boto3

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = "us-east-1"
BUCKET_NAME = "airline-policy-vectors-YOURNAME"

print("STEP 1 OK: Imports and config loaded.")


# ------------------------------------------------------------
# STEP 2: CREATE S3 VECTORS CLIENT
# ------------------------------------------------------------
#
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)
client = session.client("s3vectors")
print("STEP 2 OK: Created boto3.client('s3vectors').")


# ------------------------------------------------------------
# STEP 3: LIST VECTOR BUCKETS
# ------------------------------------------------------------
#
response = client.list_vector_buckets()
print("STEP 3: list_vector_buckets() response:")
print(response)


# ------------------------------------------------------------
# STEP 4: CHECK THAT YOUR BUCKET NAME IS PRESENT
# ------------------------------------------------------------
#
# buckets = response.get("vectorBuckets", response.get("VectorBuckets", []))
# print("STEP 4: Expected bucket:")
# print(BUCKET_NAME)
# print("STEP 4: Buckets returned by S3 Vectors:")
# for b in buckets:
#     print(f"  - {b}")
#
# print("If your bucket appears above, connection is correct.")
