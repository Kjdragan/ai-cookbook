LanceDB
Overview



 lancedb/lancedb
Home
LanceDB
🏃🏼‍♂️ Quick start
📚 Concepts
Vector search
Indexing
IVFPQ
HNSW
Storage
Data management
🔨 Guides
Working with tables
Building a vector index
Vector Search
Full-text search (native)
Full-text search (tantivy-based)
Building a scalar index
Hybrid search
Overview
Hybrid search in LanceDB
Explicitly passing the vector and text query
rerank() arguments
Available Rerankers
Comparing Rerankers
Airbnb financial data example
RAG
Vanilla RAG
Multi-head RAG
Corrective RAG
Agentic RAG
Graph RAG
Self RAG
Adaptive RAG
SFR RAG
Advanced Techniques
HyDE
FLARE
Reranking
Quickstart
Cohere Reranker
Linear Combination Reranker
Reciprocal Rank Fusion Reranker
Cross Encoder Reranker
ColBERT Reranker
Jina Reranker
OpenAI Reranker
AnswerDotAi Rerankers
Voyage AI Rerankers
Building Custom Rerankers
Example
Filtering
Versioning & Reproducibility
sync API
async API
Configuring Storage
Migration Guide
Tuning retrieval performance
Choosing right query type
Reranking
Embedding fine-tuning
🧬 Managing embeddings
Understand Embeddings
Get Started
Embedding functions
Available models
Overview
Text Embedding Functions
Sentence Transformers
Huggingface Embedding Models
Ollama Embeddings
OpenAI Embeddings
Instructor Embeddings
Gemini Embeddings
Cohere Embeddings
Jina Embeddings
AWS Bedrock Text Embedding Functions
IBM watsonx.ai Embeddings
Voyage AI Embeddings
Multimodal Embedding Functions
OpenClip embeddings
Imagebind embeddings
Jina Embeddings
User-defined embedding functions
Example: Multi-lingual semantic search
Example: MultiModal CLIP Embeddings
🔌 Integrations
Tools and data formats
Pandas and PyArrow
Polars
DuckDB
LangChain
LangChain 🔗
LangChain demo
LangChain JS/TS 🔗
LlamaIndex 🦙
LlamaIndex docs
LlamaIndex demo
Pydantic
Voxel51
PromptTools
dlt
phidata
🎯 Examples
Overview
🐍 Python
Overview
Build From Scratch
Multimodal
Rag
Vector Search
Chatbot
Evaluation
AI Agent
Recommender System
Miscellaneous
Serverless QA Bot with S3 and Lambda
Serverless QA Bot with Modal
👾 JavaScript
Overview
Serverless Website Chatbot
YouTube Transcript Search
TransformersJS Embedding Search
🦀 Rust
Overview
📓 Studies
↗Improve retrievers with hybrid search and reranking
💭 FAQs
🔍 Troubleshooting
⚙️ API reference
🐍 Python
👾 JavaScript (vectordb)
👾 JavaScript (lancedb)
🦀 Rust
☁️ LanceDB Cloud
Overview
API reference
🐍 Python
👾 JavaScript
REST API
FAQs
Quick start
Concepts
Vector search
Indexing
IVFPQ
HNSW
Storage
Data management
Guides
Working with tables
Building an ANN index
Vector Search
Full-text search (native)
Full-text search (tantivy-based)
Building a scalar index
Hybrid search
Overview
Hybrid search in LanceDB
Explicitly passing the vector and text query
rerank() arguments
Available Rerankers
Comparing Rerankers
Example - Airbnb financial data search
RAG
Vanilla RAG
Multi-head RAG
Corrective RAG
Agentic RAG
Graph RAG
Self RAG
Adaptive RAG
SFR RAG
Advanced Techniques
HyDE
FLARE
Reranking
Quickstart
Cohere Reranker
Linear Combination Reranker
Reciprocal Rank Fusion Reranker
Cross Encoder Reranker
ColBERT Reranker
Jina Reranker
OpenAI Reranker
AnswerDotAi Rerankers
Building Custom Rerankers
Example - Improve Retrievers using Rerankers & Hybrid search
Filtering
Versioning & Reproducibility
Sync API
Async API
Configuring Storage
Migration Guide
Tuning retrieval performance
Choosing right query type
Reranking
Embedding fine-tuning
Managing Embeddings
Understand Embeddings
Get Started
Embedding functions
Available models
Overview
Text Embedding Functions
Sentence Transformers
Huggingface Embedding Models
Ollama Embeddings
OpenAI Embeddings
Instructor Embeddings
Gemini Embeddings
Cohere Embeddings
Jina Embeddings
AWS Bedrock Text Embedding Functions
IBM watsonx.ai Embeddings
Multimodal Embedding Functions
OpenClip embeddings
Imagebind embeddings
Jina Embeddings
User-defined embedding functions
Example - Multi-lingual semantic search
Example - MultiModal CLIP Embeddings
Integrations
Overview
Pandas and PyArrow
Polars
DuckDB
LangChain 🦜️🔗↗
LangChain.js 🦜️🔗↗
LlamaIndex 🦙↗
Pydantic
Voxel51
PromptTools
dlt
phidata
Examples
Example projects and recipes
🐍 Python
Overview
Build From Scratch
Multimodal
Rag
Vector Search
Chatbot
Evaluation
AI Agent
Recommender System
Miscellaneous
Serverless QA Bot with S3 and Lambda
Serverless QA Bot with Modal
👾 JavaScript
Overview
Serverless Website Chatbot
YouTube Transcript Search
TransformersJS Embedding Search
🦀 Rust
Overview
Studies
Overview
↗Improve retrievers with hybrid search and reranking
API reference
Overview
Python
Javascript (vectordb)
Javascript (lancedb)
Rust
LanceDB Cloud
Overview
API reference
🐍 Python
👾 JavaScript
REST API
FAQs
Hybrid Search
LanceDB supports both semantic and keyword-based search (also termed full-text search, or FTS). In real world applications, it is often useful to combine these two approaches to get the best best results. For example, you may want to search for a document that is semantically similar to a query document, but also contains a specific keyword. This is an example of hybrid search, a search algorithm that combines multiple search techniques.

Hybrid search in LanceDB
You can perform hybrid search in LanceDB by combining the results of semantic and full-text search via a reranking algorithm of your choice. LanceDB provides multiple rerankers out of the box. However, you can always write a custom reranker if your use case need more sophisticated logic .


Sync API
Async API

import os

import openai

import lancedb

from lancedb.embeddings import get_registry


from lancedb.index import FTS


class Documents(LanceModel):
    vector: Vector(embeddings.ndims()) = embeddings.VectorField()
    text: str = embeddings.SourceField()

uri = "data/sample-lancedb"
async_db = await lancedb.connect_async(uri)
data = [
    {"text": "rebel spaceships striking from a hidden base"},
    {"text": "have won their first victory against the evil Galactic Empire"},
    {"text": "during the battle rebel spies managed to steal secret plans"},
    {"text": "to the Empire's ultimate weapon the Death Star"},
]
async_tbl = await async_db.create_table("documents_async", schema=Documents)
# ingest docs with auto-vectorization
await async_tbl.add(data)
# Create a fts index before the hybrid search
await async_tbl.create_index("text", config=FTS())
text_query = "flower moon"
vector_query = embeddings.compute_query_embeddings(text_query)[0]
# hybrid search with default re-ranker
await (
    async_tbl.query()
    .nearest_to(vector_query)
    .nearest_to_text(text_query)
    .to_pandas()
)

Note

You can also pass the vector and text query manually. This is useful if you're not using the embedding API or if you're using a separate embedder service.

Explicitly passing the vector and text query

Sync API
Async API

vector_query = [0.1, 0.2, 0.3, 0.4, 0.5]
text_query = "flower moon"
await (
    async_tbl.query()
    .nearest_to(vector_query)
    .nearest_to_text(text_query)
    .limit(5)
    .to_pandas()
)

By default, LanceDB uses RRFReranker(), which uses reciprocal rank fusion score, to combine and rerank the results of semantic and full-text search. You can customize the hyperparameters as needed or write your own custom reranker. Here's how you can use any of the available rerankers:

rerank() arguments
normalize: str, default "score": The method to normalize the scores. Can be "rank" or "score". If "rank", the scores are converted to ranks and then normalized. If "score", the scores are normalized directly.
reranker: Reranker, default RRF(). The reranker to use. If not specified, the default reranker is used.
Available Rerankers
LanceDB provides a number of rerankers out of the box. You can use any of these rerankers by passing them to the rerank() method. Go to Rerankers to learn more about using the available rerankers and implementing custom rerankers.