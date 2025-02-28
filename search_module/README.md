# Document Search Module

A powerful, modular search system for the Docling document processing pipeline. This search module provides advanced retrieval capabilities with vector-based semantic search, keyword filtering, and metadata-based filtering.

## Architecture

The search module follows a modular, provider-based architecture:

- **SearchProvider Interface**: Common API for all search implementations
- **Search Client**: Orchestrates search across providers and handles query processing
- **LanceDB Provider**: Implementation using LanceDB's vector database
- **FastAPI Interface**: REST API for search functionality

## Features

- **Semantic Search**: Vector-based search using embeddings
- **Hybrid Search**: Combine vector search with keyword and metadata filtering
- **Document Similarity**: Find documents similar to a given document
- **Query Processing**: Extract keywords and metadata filters from natural language queries
- **REST API**: HTTP API for integrating search with other applications

## Usage

### Basic Usage

```python
from search_module.search_client import SearchClient
from search_module.providers.lancedb_provider import LanceDBSearchProvider

# Initialize a provider
lancedb_provider = LanceDBSearchProvider()

# Create the search client
search_client = SearchClient(providers=[lancedb_provider])

# Perform a search
results = search_client.search("What is machine learning?", limit=5)

# Display results
for result in results:
    print(f"Score: {result.score}, Text: {result.text[:100]}...")
```

### Hybrid Search

```python
# Perform a hybrid search with automatic keyword extraction
results = search_client.hybrid_search(
    "How does OCR work in PDF documents?",
    extract_auto_keywords=True,
    limit=5
)
```

### Filtered Search

```python
# Use metadata filters in the query
results = search_client.hybrid_search(
    "doc_type:pdf page_numbers:1 machine learning techniques",
    limit=5
)

# Or pass explicit filters
results = search_client.hybrid_search(
    "machine learning techniques",
    filters={"metadata.doc_type": "pdf", "metadata.page_numbers": 1},
    limit=5
)
```

### Document Similarity

```python
# Find documents similar to a given document
results = search_client.similar_documents(
    document_id="report.pdf",
    limit=5
)
```

## API Reference

### Search Client

- `search(query, provider=None, limit=5, **kwargs)`: Perform semantic search
- `hybrid_search(query, provider=None, keywords=None, filters=None, extract_auto_keywords=True, limit=5, **kwargs)`: Perform hybrid search
- `similar_documents(document_id, provider=None, limit=5, **kwargs)`: Find similar documents

### REST API

The module provides a FastAPI-based REST API with the following endpoints:

- `POST /search`: Semantic search
- `POST /hybrid-search`: Hybrid search with keywords and filters
- `POST /similar-documents`: Document similarity search
- `GET /providers`: List available search providers

## Running the API

```bash
# From the project root directory
python -m search_module.api
```

The API will be available at http://localhost:8000

## Running the Demo

A demonstration script is included to showcase the search capabilities:

```bash
# From the project root directory
python -m search_module.demo
```

## Extending the Module

### Adding a New Provider

To add a new search provider (like LlamaIndex), create a new class that implements the `SearchProvider` interface:

```python
from search_module.providers.base import SearchProvider, SearchResult

class MyCustomProvider(SearchProvider):
    def __init__(self, name="custom"):
        super().__init__(name=name)
        # Initialize your provider
        
    def search(self, query, limit=5, **kwargs):
        # Implement semantic search
        
    def hybrid_search(self, query, keywords=None, filters=None, limit=5, **kwargs):
        # Implement hybrid search
        
    def similar_documents(self, document_id, limit=5, **kwargs):
        # Implement document similarity
```

Then register it with the search client:

```python
custom_provider = MyCustomProvider()
search_client.add_provider(custom_provider)
```
