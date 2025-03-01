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
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from search_module.providers.llamaindex import LlamaIndexProvider

# Initialize models
embed_model = OpenAIEmbedding(
    model_name="text-embedding-3-large",
    dimensions=3072
)
llm = OpenAI(model="gpt-4o")

# Create service context
service_context = ServiceContext.from_defaults(
    embed_model=embed_model,
    llm=llm
)

# Initialize the provider
provider = LlamaIndexProvider(
    db_path="lancedb_data",
    table_name="chunks",
    service_context=service_context,
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

## Import Structure (LlamaIndex 0.12+)

LlamaIndex 0.12+ uses a modular package structure that requires specific import patterns:

### Core Components
```python
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.retrievers import BaseRetriever
```

### Component-Specific Modules
```python
# Embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

# LLMs
from llama_index.llms.openai import OpenAI  
from llama_index.llms.deepseek import DeepSeek

# Vector stores
from llama_index.vector_stores.lancedb import LanceDBVectorStore
```

### Advanced Features
```python
# Our custom components
from search_module.providers.llamaindex.retrievers import EnsembleRetriever
from search_module.providers.llamaindex.citation import CitationTracker
from search_module.providers.llamaindex.response import ContextualFormatter
```

## Dependencies

- llama-index-core>=0.12.0
- llama-index-vector-stores-lancedb>=0.1.0
- llama-index-llms-openai>=0.1.0
- llama-index-embeddings-openai>=0.1.0
- llama-index-llms-deepseek>=0.1.0 (optional)
- lancedb>=0.20.0
- openai>=1.0.0

## Implementation Details

### LLM Configuration

The provider supports multiple LLM providers through a factory pattern:

```python
# OpenAI LLM
from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-4o")

# DeepSeek LLM
from llama_index.llms.deepseek import DeepSeek
llm = DeepSeek(model="deepseek-chat", api_key="your_api_key")
```

### Migration Notes (LlamaIndex 0.12+)

Some key changes in the migration to LlamaIndex 0.12+:

1. **Modular Architecture**: 
   - Core functionality is now in `llama-index-core` package
   - Provider-specific functionality is in separate packages (e.g., `llama-index-llms-openai`)
   - Each component needs to be installed separately

2. **Import Path Changes**:
   - Base classes like `BaseNodePostprocessor` moved from `llama_index.core.postprocessor` to `llama_index.core.postprocessor.node`
   - LLM base class moved from `llama_index.llms.base` to `llama_index.core.llms`

3. **Retrievers**:
   - `EnsembleRetriever` has been updated to support both dictionary and list inputs
   - Supports flexible weight configurations and improved error handling

4. **Troubleshooting**:
   - If you encounter import errors, check if the class has been moved to a different module
   - Ensure all required packages are installed (`llama-index-core`, `llama-index-embeddings-openai`, etc.)
   - Use module inspection to find the correct import paths:
     ```python
     import pkgutil
     [m.name for m in pkgutil.iter_modules(llama_index.core.__path__, llama_index.core.__name__ + '.')]
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
