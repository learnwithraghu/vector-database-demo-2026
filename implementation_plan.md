# [S3 Vector Demo Update]

This tasks involves updating the newly created AWS S3 Vector Demo to use real airline policy data instead of dummy vectors.

## User Review Required
> [!NOTE]
> The Python script `s3_vector_demo.py` requires `sentence-transformers` to be installed to run locally, just like Demo 3.

## Proposed Changes
### Syllabus
#### [MODIFY] [syllabus.md](file:///Users/raghunandana.sanur/Desktop/test-repo/vector-database/syllabus.md)
*   Update Section 7 description to mention loading the airline dataset.

### Demo 07 - AWS S3
#### [MODIFY] [s3_vector_demo.py](file:///Users/raghunandana.sanur/Desktop/test-repo/vector-database/07-aws-s3-vectors/s3_vector_demo.py)
*   Import `sentence_transformers`.
*   Load `../datasets/airline_policy_dataset.json`.
*   Generate embeddings using `all-MiniLM-L6-v2`.
*   Upload the real vectors and metadata to the assumed S3 Vector Bucket.

#### [MODIFY] [demo_instructions.md](file:///Users/raghunandana.sanur/Desktop/test-repo/vector-database/07-aws-s3-vectors/demo_instructions.md)
*   Update prerequisites to include `sentence-transformers`.
*   Update the explanation to reference the real dataset.

## Verification Plan
### Automated Tests
*   Validation is primarily manual due to the need for AWS credentials, but we can verify the script imports and logic are correct by running it (it will fail on AWS connection but should pass the data loading part if we mock it or if we just statically analyze it).
*   For this task, I will rely on code correctness based on the existing `semantic_search.py` pattern.
