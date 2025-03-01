# LlamaIndex Integration Build Plan

## Overview
This document outlines the strategy and implementation steps for integrating LlamaIndex with our existing LanceDB vector database to enhance our search capabilities. We'll focus on advanced RAG techniques while leveraging our existing document processing pipeline.

## Architecture

### Core Components
1. **Existing Document Pipeline**: Continue using Docling for document processing and chunking
2. **LanceDB**: Maintain as our primary vector database for storage
3. **LlamaIndex**: Add as a query layer on top of LanceDB for advanced retrieval and reasoning

### Integration Approach
- Connect LlamaIndex to existing LanceDB tables via the `LanceDBVectorStore` class
- Implement advanced retrieval techniques without duplicating data
- Enhance search capabilities while maintaining current pipeline efficiencies

## Key RAG Enhancements

### 1. Hybrid Search with Metadata Filtering
- Leverage LanceDB's hybrid search capabilities
- Implement metadata filtering for targeted queries
- Create flexible filtering options based on document attributes

### 2. Query Transformation Pipeline
- Query rewriting using LLM to improve semantic matching
- Query expansion for better recall
- Implement hypothetical document embeddings (HyDE) for complex queries

### 3. Ensemble Retrieval Strategy
- Combine multiple retrieval methods (vector, keyword, hybrid)
- Implement query routing for different query types
- Create specialized retrievers for different document categories

### 4. Structured Response Generation
- Citation-aware response synthesis with source tracking
- Configurable response formats (detailed, concise, bullet points)
- Contextual response generation based on query type

## LLM Flexibility
- Support for multiple LLM providers:
  - OpenAI's GPT-4o
  - Deepseek models (deepseek-chat, deepseek-reasoner)
- Configurable LLM selection for different components
- Fallback mechanisms for API failures

## Implementation Phases

### Phase 1: Core Integration (Week 1)
1. Install necessary dependencies
2. Create LanceDBVectorStore connector to existing database
3. Implement basic query functionality with LlamaIndex
4. Set up flexible LLM configuration

### Phase 2: Advanced Retrieval (Week 2)
1. Implement hybrid search with metadata filtering
2. Add query transformation pipeline
3. Create ensemble retrieval strategy
4. Develop query router for different query types

### Phase 3: Response Generation (Week 3)
1. Implement structured response generation
2. Add citation tracking to responses
3. Create configurable response formats
4. Integrate contextual response generation

### Phase 4: Testing & Optimization (Week 4)
1. Create comprehensive test suite for all components
2. Benchmark performance and quality metrics
3. Optimize latency and resource usage
4. Implement monitoring and telemetry

## Dependencies

### Core Dependencies
```
llama-index-core
llama-index-vector-stores-lancedb
llama-index-llms-openai
lancedb
openai
```

### Optional Dependencies
```
llama-index-embeddings-huggingface
llama-index-callbacks-wandb (for monitoring)
transformers (for local models)
sentence-transformers (for rerankers)
```

## Integration Points

### Provider Interface
We will create a new LlamaIndexSearchProvider that implements our existing SearchProvider interface:

```python
class LlamaIndexSearchProvider(SearchProvider):
    def __init__(self, lancedb_uri, table_name, llm_provider="openai", llm_model="gpt-4o"):
        # Initialize LanceDBVectorStore and other components
        self.setup_vector_store(lancedb_uri, table_name)
        self.setup_llm(llm_provider, llm_model)
        
    def search(self, query, limit=10, filters=None):
        # Implement advanced search logic
        pass
        
    def hybrid_search(self, query, keyword, limit=10, filters=None):
        # Implement hybrid search logic
        pass
```

### API Endpoints
Extend our existing FastAPI endpoints to support new LlamaIndex capabilities:

- `/search/llamaindex` - Advanced semantic search
- `/search/llamaindex/hybrid` - Advanced hybrid search
- `/search/llamaindex/generate` - Retrieval + response generation

## Success Criteria
1. Equal or better search quality compared to direct LanceDB queries
2. Response latency under 1 second for basic queries
3. Successful integration with existing search API
4. Support for all existing metadata filtering options
5. Flexible LLM provider configuration

## Monitoring and Evaluation
- Implement telemetry for search request success rate
- Track query latency and result quality
- Collect user feedback on search relevance
- Monitor LLM token usage for cost management

## Implementation Challenges and Solutions

### Recent Fixes (February 2025)

#### Import Path Correction
- Fixed import in search_client.py from `.providers.search_provider` to `.providers.base`
- Ensured proper class inheritance from the abstract base classes
- Standardized import paths across the codebase to prevent module not found errors

#### NumPy Serialization in Search Results
- Implemented recursive conversion of NumPy types to Python primitives
- Created `numpy_to_python` helper function to handle nested objects
- Fixed "Object of type ndarray is not JSON serializable" error in result formatting
- Added fallback serialization for non-serializable objects
- Updated to use NumPy 2.0 compatible type checking with `np.issubdtype()`

#### OpenAI Client Handling
- Implemented proper handling for both direct OpenAI client and LanceDB registry model
- Added conditional code paths for each approach in hybrid_search and similar_documents methods
- Used `embeddings.create()` API for direct client and `generate_embeddings()` for registry model
- Fixed 'OpenAI' object has no attribute 'generate_embeddings' error in hybrid search

#### Character Encoding
- Replaced Unicode checkmarks with ASCII representations [OK]
- Ensured compatibility with Windows console encoding limitations

These fixes ensure that all search methods (basic vector search, hybrid search, and document similarity) work correctly with the configured LanceDB backend and help prepare for the LlamaIndex integration.

## Future Extensions
1. Implement optional reranking for precision improvement
2. Add knowledge graph integration for concept-based search
3. Develop personalization features based on user context
4. Create evaluation framework for ongoing quality measurement

## Current Status Checklist (February 28, 2025)

### Phase 1: Core Integration
- [x] Fixed search module import paths
- [x] Fixed search result serialization for NumPy arrays
- [x] Implemented robust OpenAI client compatibility
- [x] Resolved encoding issues in console output
- [x] Updated documentation with lessons learned
- [x] Installed LlamaIndex dependencies
- [x] Created LanceDBVectorStore connector to existing database
- [x] Implemented basic query functionality with LlamaIndex
- [x] Set up flexible LLM configuration

### Phase 2: Advanced Retrieval
- [x] Implemented hybrid search with metadata filtering
- [x] Added query transformation pipeline
- [x] Created HyDE (Hypothetical Document Embeddings) implementation
- [x] Developed basic query router for search types
- [x] Implement advanced ensemble retrieval strategy (see `ensemble_retrieval.py`)

### Phase 3: Response Generation
- [x] Implemented structured response generation
- [x] Add citation tracking to responses (see `citation_tracking.py`)
- [x] Created configurable response formats
- [x] Integrate contextual response generation (see `contextual_response.py`)

### Phase 4: Testing & Optimization
- [x] Created basic test suite (`test_llamaindex.py`)
- [x] Built comparison benchmark script (`llamaindex_demo.py`)
- [ ] Optimize latency and resource usage
- [ ] Implement monitoring and telemetry

## Documentation Structure

Documentation for the search module has been consolidated in the `_documentation` directory. For details on the project's documentation organization, please refer to the [Project Documentation Structure](file:///_documentation/search_build.md#project-documentation-structure) section in the main search_build.md document.

Key updates to note:
- This document has been moved from `search_module/llamaindex_buildplan.md` to `_documentation/llamaindex_integration.md`
- Test files have been moved to `_tests/search_module/`
- Demo files have been moved to `examples/search_module/`

## Recommended Next Steps

Now that we have implemented most of the core functionality of our LlamaIndex integration, including a fully functional `LlamaIndexProvider` with query transformation capabilities, our recommended next steps are:

1. **Complete End-to-End Testing**
   - Run the `llamaindex_demo.py` script with various query types
   - Compare results between LanceDB and LlamaIndex providers
   - Document performance differences and quality improvements

2. **Enhance Retrieval Strategies**
   - Implement the full ensemble retrieval strategy that combines results from multiple retrievers
   - Add configurable reranking for improved precision
   - Develop specialized query understanding for different query types

3. **Integration with Search Client**
   - Update the main `search_client.py` to fully leverage LlamaIndex's advanced capabilities
   - Create examples demonstrating how to switch between search providers
   - Document best practices for provider selection

4. **Performance Optimization**
   - Benchmark search performance on large document collections
   - Implement caching strategies for embedding generation
   - Profile and optimize query transformation pipeline

5. **Documentation Updates**
   - Create comprehensive API documentation for the LlamaIndex provider
   - Update user guides with examples of advanced search features
   - Document the query transformation techniques and their impact on search quality

## Implementation Files

For more information on the implementation details, please refer to the following files:

- `ensemble_retrieval.py`: Implementation of the ensemble retrieval strategy
- `citation_tracking.py`: Implementation of citation tracking for responses
- `contextual_response.py`: Implementation of contextual response generation
- `test_llamaindex.py`: Test suite for the LlamaIndex provider
- `llamaindex_demo.py`: Comparison benchmark script for LlamaIndex and LanceDB providers
