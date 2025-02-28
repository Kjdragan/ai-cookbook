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

## LanceDB Integration and Pydantic Models

### Schema Design
- LanceDB provides tight integration with Pydantic through `LanceModel` and `Vector` types
- Nested models should use regular `BaseModel` for sub-structures and `LanceModel` for the root model
- Vector dimensions must match the embedding model (e.g., 3072 for text-embedding-3-large)
- Use string format (ISO) for dates to ensure proper serialization to Arrow types

### Data Serialization
- Use `model_dump()` to convert Pydantic models to dictionaries for LanceDB storage
- Ensure all fields have non-null values to prevent Arrow serialization errors
- For empty text, use placeholder strings like "empty" instead of empty strings
- Batch processing should convert each item to dictionary format before storage

### Embedding Functions
- OpenAI embeddings in LanceDB use `generate_embeddings()` method, not `embed_query`
- Input must be a list of strings, even for single items
- Empty strings can cause API errors, use placeholders instead
- API key must be provided either through environment or constructor

### Best Practices
- Initialize database with a valid schema example using real data types
- Convert all datetime objects to ISO format strings for compatibility
- Handle empty or null values explicitly with defaults
- Use proper type hints to catch issues early
- Consider batching for large document processing

### Common Issues
1. Arrow Serialization:
   - Pydantic models must be converted to dictionaries
   - Nested models need special handling
   - Date/time fields need string conversion

2. Embedding Generation:
   - Empty strings cause API errors
   - Method name differences between libraries
   - Proper dimension matching required

3. Schema Evolution:
   - Initial schema must match all future data
   - Vector dimensions must be consistent
   - Consider nullable fields carefully

### Monitoring and Validation
- Add logging for chunk processing stages
- Verify data types before insertion
- Monitor chunk counts and processing results
- Add error handling for API calls and processing steps

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

## Import Management
- When importing custom modules from project directories, various approaches may be needed:
  - Absolute imports: `from docling_playground.utils.tokenizer import OpenAITokenizerWrapper`
  - Relative imports: `from utils.tokenizer import OpenAITokenizerWrapper`
  - The right approach depends on how the module is being executed (direct vs imported)
  - For scripts that need to run both ways, consider dynamic imports that fall back based on errors

## Pipeline Design and Execution
- For document processing pipelines, carefully consider execution flow:
  - Avoid infinite loops unless specifically required for monitoring purposes
  - Provide command-line options for continuous monitoring vs one-time processing
  - Always have a graceful exit path, especially for long-running processes
  - Log detailed information about processing steps and success/failure

## Logging Best Practices
- Use timestamp-based log file naming for separate runs
- Configure both file and console logging simultaneously
- Include log rotation for long-running processes
- Log key metrics about processing (chunk counts, processing times, etc.)
- Use structured logging with consistent formatting across all modules

## Database Management
- Standardize database paths across all modules that access the same data
- When multiple modules need the same database:
  - Define the path in a common configuration location
  - Use absolute paths derived from project root
  - Document the database location in code comments
- Avoid creating multiple database instances in different locations