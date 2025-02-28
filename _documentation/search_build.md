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

## 2. Implementation Phases

### Phase 1: Core LanceDB Search

1. Implement `SearchProvider` abstract base class
2. Implement `LanceDBSearchProvider` with:
   - Basic vector search
   - Hybrid search with SQL filtering
   - Document similarity search

3. Create a basic search API endpoint:
```python
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    limit = data.get("limit", 5)
    
    provider = SearchProviderFactory.create_provider(
        "lancedb", 
        db_path="lancedb_data",
        table_name="chunks"
    )
    
    results = provider.search(query, limit)
    return jsonify({"results": [r.__dict__ for r in results]})
```

### Phase 2: LlamaIndex Integration

1. Add LlamaIndex dependencies
2. Implement `LlamaIndexSearchProvider` with:
   - Advanced retrieval strategies
   - Built-in reranking
   - Query transformation

3. Extend the API to support provider selection:
```python
@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    limit = data.get("limit", 5)
    provider_type = data.get("provider", "lancedb")
    
    provider = SearchProviderFactory.create_provider(
        provider_type, 
        db_path="lancedb_data",
        table_name="chunks"
    )
    
    results = provider.search(query, limit)
    return jsonify({"results": [r.__dict__ for r in results]})
```

### Phase 3: Advanced Features

1. Implement query routing logic:
   - Simple queries → LanceDB (faster)
   - Complex queries → LlamaIndex (more powerful)

2. Add reranking capabilities:
   - Cross-encoder reranking
   - Contextual relevance scoring

3. Add query expansion and reformulation:
   - Automatically generate related queries
   - Decompose complex questions

4. Implement evaluation framework:
   - Relevance scoring
   - Performance metrics
   - A/B testing capability

## 3. LanceDB vs LlamaIndex Comparison

### 3.1 LanceDB Strengths

- **Performance**: Direct, optimized vector operations
- **Simplicity**: Minimal dependencies and configuration
- **Integration**: Already part of our pipeline

### 3.2 LlamaIndex Strengths

- **Advanced Retrieval**: Query planning and transformation
- **Reranking**: Built-in reranking models
- **RAG Support**: Seamless LLM integration

### 3.3 Use Case Mapping

| Use Case | Recommended Provider | Rationale |
|----------|----------------------|-----------|
| Simple keyword search | LanceDB | Lower latency, simpler query needs |
| Basic semantic search | LanceDB | Direct vector similarity is sufficient |
| Complex questions | LlamaIndex | Benefits from query planning and decomposition |
| Document similarity | LanceDB | Efficient vector operations |
| Contextual questions | LlamaIndex | Better handling of context and nuance |

## 4. Technical Considerations

### 4.1 Performance Optimization

- Implement caching for frequent queries
- Use batch processing for vector embeddings
- Consider approximate nearest neighbor settings for large datasets

### 4.2 Error Handling

- Implement fallback strategies when primary search fails
- Add logging for search performance and errors
- Handle edge cases like empty queries or no results

### 4.3 Scalability

- Design for increasing document volume
- Consider sharding strategies for LanceDB
- Implement pagination for large result sets

## 5. Integration with Pipeline

The search component will integrate with the existing pipeline through:

1. **Shared Database**: Using the same LanceDB database
2. **Event-based Updates**: Listening for new document processing events
3. **Independent Scaling**: Running as a separate service for independent scaling

## 6. Next Steps

1. Implement core `SearchProvider` interface
2. Develop and test `LanceDBSearchProvider`
3. Create basic search API
4. Add LlamaIndex integration
5. Implement evaluation framework
6. Develop query routing logic
