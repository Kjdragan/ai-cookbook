Chunking













logoDocling
 DS4SD/docling
Home
Home
Installation
Usage
Supported formats
FAQ
Docling v2
Concepts
Concepts
Architecture
Docling Document
Chunking
Table of contents
Introduction
Base Chunker
Hybrid Chunker
Hierarchical Chunker
Examples
Examples
🔀 Conversion
🔀 Conversion
Simple conversion
Custom conversion
Batch conversion
Multi-format conversion
Figure export
Figure enrichment
Table export
Multimodal export
Annotate picture with local vlm
Annotate picture with remote vlm
Force full page OCR
Automatic OCR language detection with tesseract
RapidOCR with custom OCR models
Accelerator options
Simple translation
Conversion of CSV files
Conversion of custom XML
✂️ Chunking
✂️ Chunking
Hybrid chunking
🤖 RAG with AI dev frameworks
🤖 RAG with AI dev frameworks
RAG with Haystack
RAG with LangChain
RAG with LlamaIndex
🗂️ More examples
🗂️ More examples
RAG with Weaviate
RAG with Granite [↗]
RAG with Azure AI Search
Retrieval with Qdrant
Integrations
Integrations
🤖 Agentic / AI dev frameworks
🤖 Agentic / AI dev frameworks
Bee Agent Framework
Crew AI
Haystack
LangChain
LlamaIndex
txtai
⭐️ Featured
⭐️ Featured
Data Prep Kit
InstructLab
NVIDIA
Prodigy
RHEL AI
spaCy
🗂️ More integrations
🗂️ More integrations
Cloudera
DocETL
Kotaemon
OpenContracts
Vectara
Reference
Reference
Python API
Python API
Document Converter
Pipeline options
Docling Document
CLI
CLI
CLI reference
Chunking
Introduction
A chunker is a Docling abstraction that, given a DoclingDocument, returns a stream of chunks, each of which captures some part of the document as a string accompanied by respective metadata.

To enable both flexibility for downstream applications and out-of-the-box utility, Docling defines a chunker class hierarchy, providing a base type, BaseChunker, as well as specific subclasses.

Docling integration with gen AI frameworks like LlamaIndex is done using the BaseChunker interface, so users can easily plug in any built-in, self-defined, or third-party BaseChunker implementation.

Base Chunker
The BaseChunker base class API defines that any chunker should provide the following:

def chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]: Returning the chunks for the provided document.
def serialize(self, chunk: BaseChunk) -> str: Returning the potentially metadata-enriched serialization of the chunk, typically used to feed an embedding model (or generation model).
Hybrid Chunker
To access HybridChunker

If you are using the docling package, you can import as follows:

from docling.chunking import HybridChunker
If you are only using the docling-core package, you must ensure to install the chunking extra, e.g.

pip install 'docling-core[chunking]'
and then you can import as follows:

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
The HybridChunker implementation uses a hybrid approach, applying tokenization-aware refinements on top of document-based hierarchical chunking.

More precisely:

it starts from the result of the hierarchical chunker and, based on the user-provided tokenizer (typically to be aligned to the embedding model tokenizer), it:
does one pass where it splits chunks only when needed (i.e. oversized w.r.t. tokens), &
another pass where it merges chunks only when possible (i.e. undersized successive chunks with same headings & captions) — users can opt out of this step via param merge_peers (by default True)
👉 Example: see here.

Hierarchical Chunker
The HierarchicalChunker implementation uses the document structure information from the DoclingDocument to create one chunk for each individual detected document element, by default only merging together list items (can be opted out via param merge_list_items). It also takes care of attaching all relevant document metadata, including headers and captions.