# LlamaIndex Search Provider

This provider integrates LlamaIndex with our existing LanceDB vector database to provide advanced retrieval and reasoning capabilities.

## Features

- **Advanced Semantic Search**: Connects to existing LanceDB tables and provides semantic search capabilities.
- **Query Transformation**: Uses LLMs to improve queries through rewriting and expansion.
- **Hybrid Search**: Combines vector and keyword search with configurable weights.
- **Metadata Filtering**: Supports filtering by document metadata.
- **Multiple LLM Providers**: Works with OpenAI and Deepseek models.

## Usage

```python
from search_module.providers.llamaindex import LlamaIndexProvider

# Initialize the provider
provider = LlamaIndexProvider(
    db_path="lancedb_data",
    table_name="chunks",
    embedding_dim=3072,
    llm_provider="openai",  # or "deepseek"
    llm_model="gpt-4o",     # or "deepseek-chat", "deepseek-reasoner"
    hybrid_alpha=0.5,       # weight between vector and keyword search
    use_query_transform=True
)

# Perform a semantic search
results = provider.vector_search("your query here", limit=5)

# Perform a hybrid search
results = provider.hybrid_search(
    query="your query here", 
    keyword="specific term",
    top_k=5,
    alpha=0.7  # higher values prioritize semantic search
)

# Find similar documents
results = provider.similar_document_search("document_id", top_k=5)
```

## Dependencies

- llama-index-core>=0.12.0
- llama-index-vector-stores-lancedb>=0.1.0
- llama-index-llms-openai>=0.1.0
- lancedb>=0.20.0
- openai>=1.0.0

## Implementation Details

### LLM Configuration

The provider supports multiple LLM providers through a factory pattern:

```python
# Create an LLM instance
from search_module.providers.llamaindex.llm_config import LLMFactory

llm = LLMFactory.create_llm(
    provider="openai",  # or "deepseek"
    model_name="gpt-4o",
    temperature=0.0
)
```

### Query Transformation

Several query transformation techniques are implemented:

1. **Query Rewriting**: LLM-based query reformulation to add context
2. **Query Expansion**: Generating multiple related queries
3. **Hypothetical Document Embedding (HyDE)**: Creating a synthetic document to help with retrieval

```python
# Use query transformation
from search_module.providers.llamaindex.query_transformation import QueryTransformer

transformer = QueryTransformer(llm)
rewritten_query = transformer.rewrite_query("original query")
expanded_queries = transformer.expand_query("original query")
hyde_doc = transformer.generate_hypothetical_document("original query")
```

## LlamaIndex 0.12.21 Integration Notes

This provider has been updated to work with LlamaIndex 0.12.21, which uses a modular package structure:

- `llama-index-core`: Core functionality (indices, retrievers, schema)
- `llama-index-vector-stores-lancedb`: LanceDB vector store integration
- `llama-index-llms-openai`: OpenAI LLM integration

Import paths have been updated to reflect this change:

```python
# Old imports
# from llama_index.vector_stores.lancedb import LanceDBVectorStore
# from llama_index.embeddings import OpenAIEmbedding

# New imports
from llama_index_vector_stores_lancedb import LanceDBVectorStore
from llama_index_core.embeddings import OpenAIEmbedding
