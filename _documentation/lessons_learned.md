# Lessons Learned

## LanceDB Integration

### Table Creation and Schema
- When creating a LanceDB table, do not pass the `embedding` parameter to `create_table`. Instead, configure embeddings when adding data.
- Use PyArrow schema for proper type definitions, especially for non-nullable fields.
- The schema must include a vector field with the correct dimension size matching your embedding model (e.g., 1536 for text-embedding-3-large).

### Embeddings
- Use `get_registry().get("openai").create(name="model-name")` to create embedding functions.
- For batch processing of texts, use `embed_documents` instead of `embed_query`.
- The embedding function automatically handles rate limiting and retries (default 7 retries).
- LanceDB supports three methods for embeddings:
  1. Manual generation outside LanceDB
  2. Built-in embedding functions
  3. Custom embedding functions extending defaults

### Chunking with Docling
- The HybridChunker should be configured with a tokenizer matching your embedding model.
- HybridChunker performs two passes:
  1. Splits oversized chunks based on token limits
  2. Merges undersized successive chunks with same headings (optional via merge_peers)
- Set appropriate max_tokens based on your embedding model's limits (e.g., 8191 for text-embedding-3-large).

## Error Handling and Best Practices
- Always verify LanceDB chunks after storage to ensure data integrity.
- Use structured logging to track the pipeline stages (conversion, chunking, embedding, storage).
- Handle nullable fields appropriately in metadata to prevent schema validation errors.
- Consider using batch processing for embeddings to improve performance.

## GPU Acceleration and OCR
- Docling uses a global AcceleratorOptions configuration that affects all components
- Can be configured with: device=AcceleratorDevice.CUDA and num_threads parameter
- This global setting is used by OCR engines and other models
- EasyOCR is built into Docling and uses GPU acceleration through the global AcceleratorOptions

## Performance Optimization
- Model warmup adds ~11 seconds to initialization but saves ~10 seconds in processing
- Without warmup: ~43 seconds processing time
- With warmup: ~33 seconds processing time
- Consider increasing num_threads when using CUDA (e.g., from 8 to 16)