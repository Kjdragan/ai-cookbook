# Data Pipeline Build Plan

## Overview
This document outlines the plan for extending our data pipeline to handle document processing, chunking, embedding, and storage in LanceDB with support for hybrid search capabilities.

## 1. Core Components

### 1.1 Document Processing Pipeline (Existing)
- Keep current GPU-accelerated setup with Tesseract OCR
- Maintain CUDA acceleration configuration
- Keep warmup functionality for models

### 1.2 File Monitoring System (New)
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
```
- Monitor `_documents_for_processing_input` directory
- Track processed files in SQLite or similar
- Handle all Docling-supported formats
- Implement batch processing queue

### 1.3 Chunking System (New)
```python
from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper
```
- Use Docling's HybridChunker
- Configure for text-embedding-3-large (8191 tokens)
- Preserve document structure and metadata
- Implement batch chunking

### 1.4 LanceDB Integration (New)
```python
import lancedb
from lancedb.pydantic import LanceModel, Vector
```
- Use built-in OpenAI embedding support
- Define Pydantic models for schema
- Implement batch ingestion
- Enable hybrid search

## 2. Data Models

### 2.1 Document Schema
```python
class ChunkMetadata(LanceModel):
    filename: str | None
    page_numbers: List[int] | None
    title: str | None
    doc_type: str
    processed_date: datetime
    source_path: str

class Chunks(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()
    metadata: ChunkMetadata
```

## 3. Processing Flow

### 3.1 File Detection
1. Monitor input directory
2. Group files by type for batch processing
3. Track file status

### 3.2 Document Processing
1. Use existing DataPipeline for conversion
2. Process in batches using Docling's batch capabilities
3. Maintain GPU acceleration benefits

### 3.3 Chunking and Storage
1. Use HybridChunker for smart text splitting
2. Batch process chunks
3. Direct insertion into LanceDB with automatic embedding:
```python
# Get OpenAI embedding function
func = get_registry().get("openai").create(
    name="text-embedding-3-large"
)

# Create and populate table
table = db.create_table("documents", schema=Chunks)
table.add(processed_chunks)  # Automatic embedding
```

### 3.4 Search Capabilities
1. Implement hybrid search combining:
   - Semantic search via embeddings
   - Keyword search for precise matching
2. Use LanceDB's built-in reranking

## 4. Implementation Phases

### Phase 1: Core Pipeline Extension
1. Add file monitoring system
2. Integrate chunking system
3. Set up LanceDB with schema

### Phase 2: Batch Processing
1. Implement batch document processing
2. Add concurrent processing where safe
3. Optimize batch sizes

### Phase 3: Search and Retrieval
1. Implement hybrid search
2. Add reranking capabilities
3. Optimize search performance

### Phase 4: Monitoring and Optimization
1. Add processing metrics
2. Implement error handling
3. Add progress tracking

## 5. Technical Considerations

### 5.1 Batch Processing Strategy
- Group similar document types
- Use pandas DataFrames for LanceDB ingestion
- Configure optimal batch sizes
- Implement cleanup

### 5.2 Error Handling
- Track processing status
- Implement retries
- Log errors with context
- Handle partial failures

### 5.3 Performance Optimization
- Monitor embedding costs
- Track processing times
- Optimize batch sizes
- Cache frequent searches

## 6. Next Steps
1. Implement file monitoring
2. Extend DataPipeline class
3. Set up LanceDB integration
4. Add batch processing
5. Implement search functionality

## 7. Dependencies
- Docling with GPU support
- OpenAI API access
- LanceDB
- Pydantic
- Pandas (for batch operations)
- Watchdog (for file monitoring)
