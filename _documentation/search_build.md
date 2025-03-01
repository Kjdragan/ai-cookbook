# Search Implementation Plan

## Overview

This document outlines the plan for implementing a modular, pluggable search component for the document processing pipeline. The implementation will support both direct LanceDB searches and advanced retrieval using LlamaIndex, allowing for optimal performance and flexibility.

## 1. Architecture Design

### 1.1 Modular Search Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class SearchResult:
    """Standard search result format across providers."""
    text: str
    metadata: Dict[str, Any]
    score: float
    source_document: str
    page_numbers: List[int]
    
class SearchProvider(ABC):
    """Abstract interface for search providers."""
    
    @abstractmethod
    def search(self, query: str, limit: int = 5, **kwargs) -> List[SearchResult]:
        """Basic search implementation."""
        pass
        
    @abstractmethod
    def hybrid_search(self, query: str, keywords: List[str], limit: int = 5, **kwargs) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword search."""
        pass
    
    @abstractmethod
    def similar_documents(self, document_id: str, limit: int = 5) -> List[SearchResult]:
        """Find documents similar to a given document."""
        pass
```

### 1.2 Provider Implementations

#### LanceDB Provider

```python
class LanceDBSearchProvider(SearchProvider):
    """Search provider using direct LanceDB queries."""
    
    def __init__(self, db_path: str, table_name: str = "chunks"):
        import lancedb
        self.db = lancedb.connect(db_path)
        self.table = self.db.open_table(table_name)
        self.embedding_model = self._initialize_embedding_model()
        
    def search(self, query: str, limit: int = 5, **kwargs) -> List[SearchResult]:
        # Convert query to embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Perform vector search
        results = self.table.search(query_embedding).limit(limit).to_pandas()
        
        # Convert to standard format
        return self._convert_results(results)
        
    def hybrid_search(self, query: str, keywords: List[str], limit: int = 5, **kwargs) -> List[SearchResult]:
        # Vector search with keyword filtering
        query_embedding = self.embedding_model.embed_query(query)
        
        # Combine keywords with OR
        keyword_filter = " OR ".join([f"text LIKE '%{keyword}%'" for keyword in keywords])
        
        results = (self.table.search(query_embedding)
                   .where(keyword_filter)
                   .limit(limit)
                   .to_pandas())
                   
        return self._convert_results(results)
```

#### LlamaIndex Provider

```python
class LlamaIndexSearchProvider(SearchProvider):
    """Search provider using LlamaIndex capabilities."""
    
    def __init__(self, db_path: str, table_name: str = "chunks"):
        from llama_index.vector_stores import LanceDBVectorStore
        from llama_index import VectorStoreIndex, StorageContext
        
        # Initialize LanceDB vector store
        vector_store = LanceDBVectorStore(uri=db_path, table_name=table_name)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )
        
        # Initialize reranker if available
        self.reranker = self._initialize_reranker()
        
    def search(self, query: str, limit: int = 5, **kwargs) -> List[SearchResult]:
        retriever = self.index.as_retriever(similarity_top_k=limit)
        nodes = retriever.retrieve(query)
        
        if self.reranker:
            nodes = self.reranker.rerank(query, nodes)
            
        return self._convert_nodes(nodes)
```

### 1.3 Factory and Configuration

```python
class SearchProviderFactory:
    """Factory for creating search providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> SearchProvider:
        """Create a search provider of the specified type."""
        if provider_type.lower() == "lancedb":
            return LanceDBSearchProvider(**kwargs)
        elif provider_type.lower() == "llamaindex":
            return LlamaIndexSearchProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
```

## Implementation Status

### Completed

1. **Modular Search Interface:**
   - Created abstract `SearchProvider` base class
   - Defined standardized `SearchResult` class
   - Implemented provider discovery mechanism

2. **LanceDB Provider:**
   - Implemented direct vector search
   - Added support for hybrid search
   - Integrated with existing database infrastructure
   - Added metadata filtering

3. **LlamaIndex Provider:**
   - Integrated with LlamaIndex 0.12.21
   - Implemented vector, hybrid, and similar document search methods
   - Added support for multiple LLM providers (OpenAI, Deepseek)
   - Implemented query transformation techniques
   - Updated package structure to work with the modular LlamaIndex packages:
     - `llama-index-core`
     - `llama-index-vector-stores-lancedb`
     - `llama-index-llms-openai`
     - `llama-index-embeddings-openai`
     - `llama-index-llms-deepseek` (optional)
   - Enhanced `EnsembleRetriever` to support:
     - Dictionary and list inputs for retrievers
     - Flexible weight configuration and normalization
     - Improved error handling and logging
   - Fixed import paths for core components:
     - Updated `citation_tracker.py` to use correct postprocessor imports
     - Updated `query_router.py` to use correct LLM imports
     - Updated `contextual_formatter.py` to use correct response imports

4. **Search Client:**
   - Implemented provider switching
   - Added unified search interface
   - Integrated query processing utilities

5. **Search API:**
   - Created FastAPI endpoints for all search types
   - Added standardized response formatting
   - Implemented error handling

## Lessons Learned

1. **LlamaIndex API Changes:**
   - Package structure changed from monolithic `llama_index` to modular packages
   - Core functionality moved to `llama_index.core.*`
   - Component-specific modules moved to separate packages
   - See `_documentation/lessons_learned.md` for detailed migration notes

2. **API Integration:**
   - Implementing a provider-based architecture provided flexibility
   - Standardizing result format early made integration easier
   - Using abstract base classes helped maintain consistent interfaces

3. **Error Handling:**
   - Added comprehensive error handling at multiple levels
   - Implemented graceful fallback mechanisms
   - Added logging throughout the codebase

## Next Steps

1. **Performance Testing:**
   - Benchmark performance of different providers
   - Optimize search parameters for better results
   - Test with larger document collections

2. **Advanced RAG Techniques:**
   - Implement query routing based on query type
   - Add multi-step reasoning capabilities
   - Incorporate self-correction mechanisms

3. **User Interface:**
   - Create a search UI for testing different providers
   - Implement result highlighting
   - Add document preview functionality

4. **Documentation:**
   - Complete API documentation
   - Add usage examples for different search types
   - Document configuration options for each provider

## Technical Reference

### Package Dependencies

```
lancedb>=0.20.0
openai>=1.0.0
numpy>=1.20.0
pandas>=1.3.0
python-dotenv>=0.19.0
pydantic>=2.0.0
fastapi>=0.100.0
uvicorn>=0.20.0
tenacity>=8.0.0
llama-index-core>=0.12.0
llama-index-vector-stores-lancedb>=0.1.0
llama-index-llms-openai>=0.1.0
```

### Key Configuration Options

#### LanceDB Provider

- `db_path`: Path to LanceDB database
- `table_name`: Name of LanceDB table
- `embedding_model`: Name of OpenAI embedding model
- `embedding_dim`: Dimension of embedding vectors (3072 for text-embedding-3-large)

#### LlamaIndex Provider

- `db_path`: Path to LanceDB database
- `table_name`: Name of LanceDB table
- `embedding_dim`: Dimension of embedding vectors
- `llm_provider`: LLM provider (openai or deepseek)
- `llm_model`: Model name (gpt-4o, gpt-3.5-turbo, deepseek-chat, etc.)
- `hybrid_alpha`: Weight between vector and keyword search (0-1)
- `use_query_transform`: Whether to use query transformation techniques

## Project Documentation Structure

To maintain clarity and prevent losing track of progress as we develop the search module, our documentation is structured as follows:

### 1. Implementation Documentation (`_documentation/`)

- **search_build.md**: This primary document that outlines the overall search architecture, components, and implementation status for all search providers
- **llamaindex_integration.md**: Detailed plan and status tracking specifically for the LlamaIndex integration
- **lessons_learned.md**: Technical challenges, solutions, and best practices discovered during implementation

### 2. Provider-Specific Documentation

- **providers/[provider_name]/README.md**: Implementation details, usage examples, and API documentation for each provider
- Each provider directory should maintain its own detailed documentation to minimize cross-document dependencies

### 3. Tests and Examples

- **_tests/search_module/**: Tests for verifying search functionality (moved from module directory)
- **examples/search_module/**: Usage examples and demonstrations (like demo.py) to show how to use different search features

### Documentation Maintenance Rules

1. **General Updates**: When making changes that affect multiple providers, update `search_build.md`
2. **Provider-Specific Updates**: When changing a specific provider, update its README.md file
3. **New Challenges**: Document any technical challenges and solutions in `lessons_learned.md`
4. **Status Tracking**: Keep status checklists updated when completing implementation tasks
5. **Code Examples**: Ensure examples are kept in sync with API changes

This structure ensures documentation remains organized as the project grows, with clear responsibilities for each document type.
