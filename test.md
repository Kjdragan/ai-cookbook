# Document Processing Pipeline Testing Plan

This document outlines a step-by-step testing procedure to diagnose and fix issues with our document processing pipeline, particularly concerning database population.

## Problem Statement

The query system reveals that our database contains only a single document ("S1-Simple-test-time scaling.pdf") with 10 chunks, despite previous testing suggesting that the pipeline should process all documents in the input directory.

## Testing Procedure

### 1. Reset the Environment

First, we'll delete all database files to ensure a clean slate:

```bash
# Remove LanceDB database directory
rm -rf lancedb_data

# Remove SQLite tracking database
rm -rf processed_files.db
```

### 2. Verify Input Documents

Check that we have multiple documents in the input directory:

```bash
# List documents in input directory
dir input_documents
```

### 3. Run the Pipeline in Verbose Mode

Execute the document processing pipeline with detailed logging:

```bash
uv run python process_documents.py --input_dir input_documents --verbose
```

During execution, verify:
- CUDA initialization and GPU acceleration are working
- All documents are being found and processed
- No errors or exceptions are occurring during processing

### 4. Verify Database Contents

After the pipeline completes, check that the database contains all processed documents:

```bash
# Check database statistics
uv run python query_documents.py --stats
```

The output should show:
- Multiple documents (matching the number in the input directory)
- Appropriate number of chunks per document
- Expected page count for each document

### 5. Diagnostic Queries

Run sample queries to verify content accessibility:

```bash
# Try a query likely to match multiple documents
uv run python query_documents.py --query "machine learning" --limit 5
```

### 6. Check Database Configuration

Verify the database path is consistent across scripts:

- In `process_documents.py`: Check the storage path
- In `query_documents.py`: Confirm the same path is used
- In `datapipeline.py`: Review how document chunks are created and stored

### 7. Review Document Chunking Parameters

Examine document chunking settings:
- Chunk size and overlap
- Document processing parameters
- Text extraction quality

## Expected Outcome

After completing this test procedure:

1. The database should contain all documents from the input directory
2. Each document should have an appropriate number of chunks
3. Query statistics should reflect the correct document count
4. Semantic search should return results from multiple documents

## Next Steps

If issues persist:
1. Examine logs for errors during processing
2. Check for file permission issues
3. Verify document formats are supported
4. Consider data migration if database schemas have changed
