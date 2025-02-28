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

## LanceDB API Compatibility Issues

- LanceDB has different API versions that require special handling for cross-version compatibility.
- Common API differences include:
  - **Search parameters**: Newer versions support `nprobes` and `refine_factor` parameters, while older versions may not.
  - **Method chaining**: Newer versions use `table.search().limit(n).to_pandas()` pattern, while older versions require `table.search().to_pandas().head(n)`.
  - **Table methods**: The `stats()` method exists in newer versions but not in older ones.
  - **Return values**: Distance scores may be handled differently between versions.
  - **Embedding registry**: Version 0.20.0+ uses a variable store mechanism for API keys, while 0.19.0 passes keys directly.
  - **Embedding methods**: Different versions may have `generate_embeddings()`, `embed_query()`, or other method variations.

- Best practices for handling API differences:
  1. Use try/except blocks to catch `AttributeError` and other exceptions related to missing methods.
  2. Implement fallback mechanisms for different API versions.
  3. Avoid hardcoding parameters that might not be supported across versions.
  4. Test with sample queries before implementing complex search logic.
  5. When possible, abstract API differences behind a consistent interface.
  6. Implement cascading fallbacks from newest to oldest API patterns.
  7. As a last resort, provide manual implementations for core functionality.

- 0.19.0 to 0.20.0 Major API Changes:
  1. Embedding Registry:
     ```python
     # 0.20.0+ with variable store
     registry = get_registry()
     registry.variable_store.set("OPENAI_API_KEY", api_key)
     embedding_model = registry.get("openai").create(name=model_name)
     
     # 0.19.0 direct API key
     embedding_model = get_registry().get("openai").create(
         name=model_name,
         api_key=api_key
     )
     ```
     
     Note: In 0.20.0, passing API keys directly will trigger a warning: "Sensitive key 'api_key' cannot be set to a hardcoded value"
  
  2. Multiple API Patterns:
     Implement multiple fallback patterns for version-agnostic code:
     ```python
     # Try multiple API patterns in sequence
     api_methods = [
         # Newest API with parameters
         lambda: table.search(vector, **params).limit(limit).to_pandas(),
         # Newest API without parameters
         lambda: table.search(vector).limit(limit).to_pandas(),
         # Older API with parameters
         lambda: table.search(vector, **params).to_pandas().head(limit),
         # Older API without parameters
         lambda: table.search(vector).to_pandas().head(limit),
     ]
     
     # Try each method until one works
     for method in api_methods:
         try:
             results = method()
             break
         except Exception:
             continue
     ```

   3. Checking LanceDB Version:
      ```python
      import lancedb
      
      # Get the LanceDB version
      version = lancedb.__version__
      
      # Check if version is compatible
      if version is None:
          print("Unable to determine LanceDB version")
      elif version.startswith("0.20"):
          print("Using LanceDB 0.20.x")
      elif version.startswith("0.19"):
          print("Using LanceDB 0.19.x")
      else:
          print(f"Using unsupported LanceDB version: {version}")
      ```

## LanceDB 0.20.0 Compatibility

- The `variable_store` feature for API key management may not be available in all 0.20.0 builds
- Direct API key passing is restricted for security reasons
- Best practice is to implement multiple fallback mechanisms:
  1. Try using `variable_store` if available
  2. Attempt environment variable references for API keys
  3. Fall back to direct client initialization when needed

## Search Results Handling

- Always validate search result fields before accessing them
- Implement standardized metadata handling across providers
- Provide graceful degradation when fields are missing
- Use consistent scoring metrics (lower is better for distance metrics)

## Error Recovery Patterns

- Implement retry logic for API-related operations
- Log critical state information before and after operations
- Use try/except blocks to handle version-specific API differences
- Provide clear error messages that guide users toward solutions

## Handling NumPy Arrays in LanceDB Metadata

- LanceDB metadata fields may contain NumPy arrays that are not JSON serializable.
- When working with LanceDB data that needs to be serialized:
  - Convert NumPy numeric types to Python primitives (`int()`, `float()`, `bool()`)
  - Convert NumPy arrays to Python lists using `array.tolist()`
  - Implement recursive conversion functions for nested structures
  - Handle special NumPy types like `np.int64`, `np.float32`, etc.

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