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

## Search Module Implementation Lessons

### Import Path Management (February 2025)
- When organizing code with modules and packages, pay careful attention to import paths
- Always check that import references match the actual file locations and class names
- Common errors include:
  - Using incorrect relative import paths (e.g., `.providers.search_provider` vs `.providers.base`)
  - Importing from non-existent modules
  - Class names not matching between import statements and implementation files
- Python's import system can be confusing when mixing direct script execution and package imports
- Best practice is to use absolute imports when possible and ensure consistent naming conventions

### NumPy Data Serialization (February 2025)
- NumPy arrays and types are not JSON serializable by default
- When working with search results containing NumPy data:
  1. Recursively convert all NumPy types to Python primitives before serialization
  2. Handle nested dictionaries and lists that might contain NumPy objects
  3. Create a dedicated helper function for this conversion
  4. Add proper error handling and fallbacks for serialization failures
- Example implementation of a NumPy to Python converter:
  ```python
  def numpy_to_python(obj):
      """Recursively convert numpy types to native Python types."""
      if isinstance(obj, np.ndarray):
          return obj.tolist()
      elif np.issubdtype(type(obj), np.integer):
          return int(obj)
      elif np.issubdtype(type(obj), np.floating):
          return float(obj)
      elif np.issubdtype(type(obj), np.bool_):
          return bool(obj)
      elif isinstance(obj, dict):
          return {k: numpy_to_python(v) for k, v in obj.items()}
      elif isinstance(obj, list) or isinstance(obj, tuple):
          return [numpy_to_python(item) for item in obj]
      else:
          return obj
  ```
- In NumPy 2.0, many type aliases were removed (e.g., `np.float_` is now `np.float64`)
- Use `np.issubdtype()` for more future-proof type checking rather than direct instance checks

### OpenAI Client Compatibility (February 2025)
- Different versions of OpenAI's client and integration patterns require conditional handling
- The direct OpenAI client uses:
  ```python
  response = client.embeddings.create(
      input=query,
      model=model_name
  )
  query_embedding = response.data[0].embedding
  ```
- While the LanceDB registry embedding model uses:
  ```python
  query_embedding = embedding_model.generate_embeddings([query])[0]
  ```
- Mixing these patterns results in the error: 'OpenAI' object has no attribute 'generate_embeddings'
- Solution: Use conditional code paths based on client type
  ```python
  if hasattr(self, 'is_direct_client') and self.is_direct_client:
      # Direct OpenAI client approach
      response = self.embedding_model.embeddings.create(...)
  else:
      # LanceDB registry model approach
      query_embedding = self.embedding_model.generate_embeddings([query])[0]
  ```

### Character Encoding Issues (February 2025)
- Windows console has limitations with Unicode character display
- Using special characters like checkmarks (✓) can cause errors: 'charmap' codec can't encode character
- Best practice is to use ASCII alternatives like [OK] or [CHECK] in console output
- For cross-platform compatibility, consider:
  1. Using only ASCII characters in console output
  2. Implementing platform detection for conditional character sets
  3. Adding try/except blocks around print statements with potential encoding issues
  4. Testing output on multiple platforms and console environments

## LlamaIndex Modular Structure (Version 0.12+)

### Package Organization
- LlamaIndex 0.12+ uses a modular package structure with separate packages for different functionalities:
  1. `llama-index-core`: Core functionality and base classes
  2. `llama-index-embeddings-openai`: OpenAI embedding functions
  3. `llama-index-llms-openai`: OpenAI LLM integration
  4. `llama-index-vector-stores-lancedb`: LanceDB vector store
  5. `llama-index-retrievers-bm25`: BM25 retriever implementation
- Each module needs to be installed separately (e.g., `uv add llama-index-core`)

### Import Patterns
- Core components use the `llama_index.core` namespace:
  ```python
  from llama_index.core import VectorStoreIndex, ServiceContext
  from llama_index.core.schema import Document, QueryBundle
  from llama_index.core.retrievers import BaseRetriever
  ```
- Component-specific modules are imported from their namespaces:
  ```python
  from llama_index.embeddings.openai import OpenAIEmbedding
  from llama_index.llms.openai import OpenAI
  from llama_index.vector_stores.lancedb import LanceDBVectorStore
  ```

### Common Issues
- Importing from the wrong namespace causes `ModuleNotFoundError`
- Previously monolithic imports (`from llama_index.xxx`) no longer work
- Legacy tutorials and examples may use outdated import patterns
- Module functionality may vary slightly between versions
- Class names may remain the same but in different modules

### Best Practices
- Check installed package versions before debugging import errors
- Use try/except blocks to handle potential missing modules
- Always check the latest documentation for correct import patterns
- Use `dir()` or `help()` to inspect module contents when unsure
- Consider adding aliased imports for backward compatibility in utility code

### Migration Approach
- For services using LlamaIndex, update imports incrementally
- Test each component after migration
- Create a compatibility layer if necessary for backward compatibility
- Document new import patterns in project READMEs and examples

## LlamaIndex 0.12+ Migration Lessons

### Import Path Changes

LlamaIndex has moved to a modular structure in versions 0.12 and later, requiring changes to import paths and module usage.

### Key Package Changes

The most significant change is the modular structure, where different functionalities are now in separate packages:

- **llama-index-core**: Core functionality (indexes, schema, retrievers)
- **llama-index-embeddings-***:  Embedding models (OpenAI, HuggingFace, etc.)
- **llama-index-llms-***:  Language models (OpenAI, DeepSeek, etc.)
- **llama-index-vector-stores-***:  Vector database integrations
- **llama-index-readers-***:  Document readers/loaders

### Import Path Changes

Import paths have changed significantly:

1. Base components are now in `llama_index.core`:
   ```python
   # Old
   from llama_index import VectorStoreIndex, ServiceContext
   # New
   from llama_index.core import VectorStoreIndex
   from llama_index.core.settings import Settings  # Replaces ServiceContext
   ```

2. Embeddings:
   ```python
   # Old
   from llama_index.embeddings.openai import OpenAIEmbedding
   # New
   from llama_index.embeddings.openai import OpenAIEmbedding  # Note: requires llama-index-embeddings-openai package
   ```

3. LLMs:
   ```python
   # Old
   from llama_index.llms.openai import OpenAI
   # New
   from llama_index.llms.openai import OpenAI  # Note: requires llama-index-llms-openai package
   ```

4. Postprocessors:
   ```python
   # Old
   from llama_index.postprocessor import BaseNodePostprocessor
   # New
   from llama_index.core.postprocessor.node import BaseNodePostprocessor
   ```

5. Response schemas:
   ```python
   # Old
   from llama_index.response.schema import Response
   # New
   from llama_index.core.response import Response
   ```

### ServiceContext to Settings Migration

ServiceContext is replaced with the Settings class:

```python
# Old
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding(model="text-embedding-ada-002")
)

# New
from llama_index.core.settings import Settings
# Set global settings
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
# Or use local settings
settings = Settings.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding(model="text-embedding-ada-002")
)
```

### Implementation Details

1. **QueryBundle Class Changes**:
   - In LlamaIndex 0.12+, the QueryBundle class no longer has a `metadata` attribute
   - Need to handle metadata externally rather than attaching it to QueryBundle

2. **Response Module Changes**:
   - Response class moved from `llama_index.response.schema` to `llama_index.core.response`
   - Format and structure of Response objects remain generally compatible

3. **QueryRouter Implementation**:
   - The API expectations for QueryRouter changed to support more modular usage
   - Implementations need to explicitly handle the route_query method with correct parameter ordering

4. **CitationResponseSynthesizer Updates**:
   - Required implementation of all abstract methods from BaseSynthesizer
   - Parameter names changed from `base_synthesizer` to `response_synthesizer` in some implementations
   - Need to implement both synchronous and asynchronous methods

5. **ContextualFormatter API**:
   - Parameter expectations changed slightly between versions
   - Methods like format_response need to be called with the correct parameter names

### Testing Considerations

When migrating to LlamaIndex 0.12+, thorough testing is essential:

1. Test all search and retrieval operations
2. Verify custom components work correctly
3. Check for missing imports or parameter mismatches 
4. Test with exact versions of dependencies to ensure consistent behavior

### Common Issues

- `ModuleNotFoundError`: Class may have moved to a different namespace or package
- `ImportError`: Class may have been renamed or restructured
- Missing dependencies: Ensure all modular packages are installed
- Breaking changes in APIs: Some methods and parameters may have changed

### DeepSeek Integration

- DeepSeek LLM integration requires `llama-index-llms-deepseek` package
- Can be initialized with:
  ```python
  from llama_index.llms.deepseek import DeepSeek
  llm = DeepSeek(model="deepseek-chat", api_key="your-api-key")
  ```
- Supports both completion and chat interfaces
- Can be used as a drop-in replacement for other LLM providers