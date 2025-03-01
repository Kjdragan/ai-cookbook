# Docling Document Processing Pipeline Context

## Project Overview
The Docling Document Processing Pipeline is a robust system for processing documents with GPU acceleration, advanced chunking, embedding generation, and vector database integration for semantic search capabilities.

## Current Implementation Status
- Successfully implemented a GPU-accelerated document converter using Docling framework
- Added embedding generation with OpenAI's text-embedding-3-large model (3072 dimensions)
- Implemented robust metadata extraction including page numbers, content type detection
- Created a flexible database schema for document chunks in LanceDB
- Added Unicode handling for Windows environments to prevent encoding errors
- Implemented error recovery with fallback strategies
- Created comprehensive logging with encoding error handling

## Core Components

### Document Conversion
- Uses `DocumentConverter` with GPU acceleration via CUDA
- `AcceleratorOptions` with `AcceleratorDevice.CUDA` and `num_threads=8`
- EasyOCR integration for text extraction, also GPU-accelerated
- Table structure analysis enabled

### Text Chunking
- Uses `HybridChunker` with `max_tokens=8191` for OpenAI text-embedding-3-large
- Custom `OpenAITokenizerWrapper` for token counting
- Chunk metadata includes page numbers, headings, and source information
- Content type detection for code, tables, lists, math, and regular text

### Embedding Generation
- Uses OpenAI's text-embedding-3-large model (3072 dimensions)
- Multiple initialization strategies for embedding client:
  1. LanceDB embedding registry with variable_store (newer API)
  2. Environment variable reference approach
  3. Direct OpenAI client as fallback
- Error handling with zero-vector fallbacks

### Metadata Extraction
- Multiple strategies for extracting page numbers:
  1. From document metadata
  2. From text pattern matching (Page X of Y)
  3. From inline page markers
- Content type detection based on text structure analysis

### Database Integration
- LanceDB for vector storage
- SQLite for tracking processed files
- PyArrow schemas for proper type handling
- Pydantic models for data validation

### Unicode Handling
- Custom logging handlers for Windows environments
- Text sanitization to prevent UnicodeEncodeError
- Separate console output file for debugging

## Recent Improvements
1. **Unicode Error Handling**: Added custom logging handlers and text sanitization to prevent UnicodeEncodeError in Windows
2. **Pydantic Integration**: Improved Pydantic model handling with proper conversion to dictionaries for LanceDB
3. **Robust Error Recovery**: Added comprehensive fallback strategies for embedding generation and metadata extraction

## Key Files
- `datapipeline.py`: Core pipeline implementation
- `utils/tokenizer.py`: Custom tokenizer wrapper
- `utils/sitemap.py`: Utility for processing website content

## Environment Setup
- Python 3.12 runtime
- Virtual environment at C:\Users\kevin\repos\docling-playground\.venv
- Package manager: uv
- NVIDIA GPU acceleration with CUDA
- API keys stored in .env file

## Current Challenges
1. ~~Unicode encoding errors in Windows environment~~ (Fixed with custom handlers)
2. Balancing batch sizes for optimal performance
3. Optimizing performance for large document processing
4. Handling diverse document structures

## Next Steps
1. Test the pipeline with more diverse document types
2. Implement more advanced content type detection
3. Add comprehensive unit testing
4. Optimize embedding generation for large documents
5. Consider adding more metadata extraction options

## API and Data Flow Details

### Document Processing Flow:
1. Document is loaded by `DocumentConverter`
2. Converted to text chunks with metadata
3. Each chunk is processed to extract metadata
4. Embeddings are generated for each chunk
5. Chunks are stored in LanceDB
6. Document is marked as processed in SQLite

### Key APIs and Dependencies:
- Docling Framework: Document conversion and chunking
- OpenAI API: Embedding generation
- LanceDB: Vector storage
- EasyOCR: Text extraction (GPU-accelerated)
- PyTorch: Underlying framework for GPU operations

### Processing Modes:
1. **One-time processing**: Process files once and exit
2. **Monitoring mode**: Continuously monitor for new documents (with --monitor flag)

## Code Patterns to Follow
1. Always check for file existence before processing
2. Apply text sanitization before logging or embedding
3. Convert Pydantic models to dictionaries before LanceDB storage
4. Implement multiple fallbacks for embedding generation
5. Use custom logging handlers to handle encoding issues
6. Process documents in batches for better performance
