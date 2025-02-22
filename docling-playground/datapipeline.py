import os
import json
import time
import logging
import pyarrow as pa
from datetime import datetime
from typing import List, Dict, Any, Optional, Type, Union
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.utils.model_downloader import download_models
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.tokenizer import OpenAITokenizerWrapper
from docling.datamodel.pipeline_options import (
    AcceleratorOptions, 
    AcceleratorDevice, 
    PdfPipelineOptions,
    EasyOcrOptions
)

import lancedb
from lancedb.pydantic import LanceModel, Vector, FixedSizeListMixin
from lancedb.embeddings import EmbeddingFunctionRegistry
import numpy as np
from pydantic import Field, ConfigDict

class ChunkMetadata(LanceModel):
    """Metadata for document chunks."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    filename: str
    page_numbers: List[int]
    title: str
    doc_type: str
    processed_date: datetime
    source_path: str
    chunk_id: str  # Unique identifier for each chunk

class ProcessedChunk(LanceModel):
    """Schema for document chunks in LanceDB with automatic embedding support."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    metadata: ChunkMetadata
    text: str = None  # Will be set by embedding function
    vector: Vector = None  # Will be set by embedding function

    @classmethod
    def with_embeddings(cls, func):
        """Create a new schema with the given embedding function."""
        # Create a new class with the embedding function fields
        return type(cls.__name__, (cls,), {
            "__annotations__": {
                "text": str,
                "vector": Vector(func.ndims())
            },
            "text": func.SourceField(),
            "vector": func.VectorField()
        })

    @classmethod
    def get_arrow_schema(cls):
        """Get the Arrow schema for this model."""
        return pa.schema([
            pa.field("metadata", pa.struct([
                pa.field("filename", pa.string()),
                pa.field("page_numbers", pa.list_(pa.int64())),
                pa.field("title", pa.string()),
                pa.field("doc_type", pa.string()),
                pa.field("processed_date", pa.timestamp('ns')),
                pa.field("source_path", pa.string()),
                pa.field("chunk_id", pa.string())
            ])),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 1536))  # Fixed size for text-embedding-3-small
        ])

    def __init__(self, **data):
        super().__init__(**data)

class Chunks(LanceModel):
    """Schema for document chunks in LanceDB."""
    metadata: ChunkMetadata
    text: str
    vector: Optional[List[float]] = None  # Will be set dynamically based on embedding model

class DocumentHandler(FileSystemEventHandler):
    """Handles file system events for document processing."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def on_created(self, event):
        if not event.is_directory:
            self.pipeline.queue_document(event.src_path)

class DataPipeline:
    def __init__(self, artifacts_path=None):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.init_start_time = time.time()
        self.logger.info("Initializing document processing pipeline...")
        
        if artifacts_path is None:
            artifacts_path = os.path.join(str(Path.home()), '.cache', 'docling', 'models')
        
        # Download and prefetch models if needed
        self.logger.info("Checking and downloading required models...")
        self._warmup_models(artifacts_path)
        
        self.logger.info("Setting up CUDA acceleration...")
        # Set up CUDA acceleration
        self.accelerator_options = AcceleratorOptions(
            num_threads=8, 
            device=AcceleratorDevice.CUDA  # Explicitly use CUDA
        )

        # Configure pipeline options with CUDA acceleration
        self.logger.info("Configuring pipeline options...")
        self.pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        self.pipeline_options.accelerator_options = self.accelerator_options
        
        # Configure OCR options to use EasyOCR with acceleration
        self.logger.info("Configuring EasyOCR with acceleration...")
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = EasyOcrOptions()
        
        # Enable table structure analysis
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        # Create converter with CUDA-enabled options
        self.logger.info("Creating document converter...")
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )
        
        # Initialize document monitoring
        self._setup_document_monitoring()
        
        # Initialize database and chunker
        self._setup_database()
        
        # Initialize chunker with OpenAI tokenizer
        self.tokenizer = OpenAITokenizerWrapper()
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=8191,  # text-embedding-3-large's maximum context length
            merge_peers=True  # Merge undersized successive chunks with same headings
        )
        
        init_time = time.time() - self.init_start_time
        self.logger.info(f"Pipeline initialization time: {init_time:.2f} seconds")
        self.logger.info("Pipeline initialization complete. Ready for processing.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _warmup_models(self, artifacts_path):
        """Download and prefetch required models with retry logic."""
        # Create cache directory if it doesn't exist
        os.makedirs(artifacts_path, exist_ok=True)
        
        # Create a cache marker file to track downloaded models
        cache_marker = os.path.join(artifacts_path, ".model_cache_complete")
        
        # Skip download if cache exists and is valid
        if os.path.exists(cache_marker):
            self.logger.info("Using cached models from: %s", artifacts_path)
            return
            
        try:
            self.logger.info("Downloading models...")
            download_models()
            self.logger.info(f"Models downloaded successfully to: {artifacts_path}")
            
            # Create cache marker after successful download
            with open(cache_marker, 'w') as f:
                f.write(str(datetime.now()))
        except Exception as e:
            self.logger.error(f"Error during model download: {str(e)}")
            raise

    def verify_lancedb_chunks(self, expected_count: Optional[int] = None) -> bool:
        """Verify that chunks were properly stored in LanceDB."""
        try:
            # Get all chunks from the table
            all_chunks = self.table.to_pandas()
            chunk_count = len(all_chunks)
            self.logger.info(f"Found {chunk_count} chunks in LanceDB table")
            
            # Print diagnostic information for the first chunk
            if chunk_count > 0:
                self.logger.info("Sample chunk data:")
                sample_chunk = all_chunks.iloc[0]
                self.logger.info("Available columns: %s", list(all_chunks.columns))
                self.logger.info("Sample chunk metadata: %s", sample_chunk.get('metadata', 'No metadata'))
                self.logger.info("Sample chunk text: %s", sample_chunk.get('text', 'No text')[:200] + '...' if sample_chunk.get('text') else 'No text')
                self.logger.info("Sample chunk vector shape: %s", len(sample_chunk.get('vector', [])) if sample_chunk.get('vector') is not None else 'No vector')
            
            if expected_count is not None and chunk_count != expected_count:
                self.logger.warning(f"Expected {expected_count} chunks but found {chunk_count}")
                return False
            
            # Verify required fields are present
            required_fields = ['text', 'vector', 'metadata']
            for field in required_fields:
                if field not in all_chunks.columns:
                    self.logger.error(f"Missing required field: {field}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying chunks: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_document(self, file_path: Path):
        """Process a single document through the pipeline with retry logic."""
        try:
            self.logger.info(f"Starting document processing: {file_path.name}")
            
            # Convert document
            self.logger.info("Converting document...")
            result = self.converter.convert(str(file_path))
            
            # Apply chunking
            self.logger.info("Chunking document...")
            chunks = list(self.chunker.chunk(dl_doc=result.document))
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # Prepare chunks for LanceDB
            self.logger.info("Preparing chunks for storage...")
            processed_chunks = [
                ProcessedChunk.with_embeddings(self.embedding_func)(
                    text=chunk.text or "",  # Ensure non-null
                    metadata=ChunkMetadata(
                        filename=chunk.meta.origin.filename or file_path.name,
                        page_numbers=sorted(set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )) or [0],  # Default to page 0 if no page numbers
                        title=chunk.meta.headings[0] if chunk.meta.headings else "",
                        doc_type=file_path.suffix[1:] or "unknown",  # Remove leading dot
                        processed_date=datetime.now(),
                        source_path=str(file_path),
                        chunk_id=str(i)  # Assign a unique chunk ID
                    ),
                    vector=np.zeros(self.embedding_func.ndims()).tolist()  # Initialize vector, will be replaced by embedding function
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Add chunks to LanceDB - it will handle embeddings automatically
            self.logger.info("Storing chunks in LanceDB...")
            self.table.add(processed_chunks, mode="replace", id_field="metadata.chunk_id")
            
            # Verify storage
            if self.verify_lancedb_chunks(len(processed_chunks)):
                self.logger.info(f"Successfully processed and stored: {file_path.name}")
            else:
                raise Exception("Chunk verification failed")
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path.name}: {str(e)}")
            raise

    def timing_decorator(func):
        """Decorator to log function execution time."""
        def wrapper(self, *args, **kwargs):
            self.logger.info(f"Starting {func.__name__}...")
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            self.logger.info(f"Processing time for {func.__name__}: {end_time - start_time:.2f} seconds")
            return result
        return wrapper

    @timing_decorator
    def process_pdf(self, url):
        """Process a PDF document through the full pipeline."""
        start_time = time.time()
        self.logger.info("Starting process_pdf...")
        
        # Download and convert document
        document = self._download_and_convert_pdf(url)
        
        # Process document through chunking pipeline
        self.logger.info("Processing document through chunking pipeline...")
        chunks = list(self.chunker.chunk(dl_doc=document))
        
        # Log chunk types for diagnosis
        chunk_types = {}
        for chunk in chunks:
            chunk_type = type(chunk.meta.doc_items[0]).__name__ if chunk.meta.doc_items else "Unknown"
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        self.logger.info(f"Created {len(chunks)} chunks with types: {chunk_types}")
        
        # Prepare chunks for LanceDB
        self.logger.info("Preparing chunks for storage...")
        processed_chunks = []
        batch_size = 10  # Process in smaller batches to avoid rate limits
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}")
            
            for chunk in batch:
                # Skip empty chunks
                if not chunk.text:
                    self.logger.debug(f"Skipping empty chunk of type: {type(chunk.meta.doc_items[0]).__name__ if chunk.meta.doc_items else 'Unknown'}")
                    continue
                    
                # Create ProcessedChunk instance with metadata
                processed_chunk = ProcessedChunk.with_embeddings(self.embedding_func)(
                    text=chunk.text,
                    metadata=ChunkMetadata(
                        filename=os.path.basename(url) + ".pdf",
                        page_numbers=sorted(set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )) or [0],
                        title=chunk.meta.headings[0] if chunk.meta.headings else "",
                        doc_type="pdf",
                        processed_date=datetime.now(),
                        source_path=url,
                        chunk_id=str(i)
                    ),
                    vector=np.zeros(self.embedding_func.ndims()).tolist()
                )
                processed_chunks.append(processed_chunk)
        
        # Add chunks to LanceDB - embeddings will be computed automatically
        self.logger.info("Storing chunks in LanceDB...")
        
        # First, clear the existing table data
        self.table.delete(where="1=1")
        
        # Then add all chunks
        self.table.add(
            data=processed_chunks,
            mode="append"  # Use append mode after clearing the table
        )
        
        # Verify storage
        if self.verify_lancedb_chunks(len(processed_chunks)):
            self.logger.info(f"Successfully processed and stored {len(processed_chunks)} chunks")
        else:
            self.logger.warning("Chunk verification failed")

        return 

    @timing_decorator
    def process_html(self, url):
        result = self.converter.convert(url)
        document = result.document
        markdown_output = document.export_to_markdown()

        # Save output to _processed_output directory
        base_name = os.path.basename(url).replace('.html', '')
        markdown_file = os.path.join('_processed_output', f'{base_name}.md')

        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)

        return markdown_file

    @timing_decorator
    def process_sitemap(self, url):
        sitemap_urls = get_sitemap_urls(url)
        conv_results_iter = self.converter.convert_all(sitemap_urls)

        docs = []
        for result in conv_results_iter:
            if result.document:
                document = result.document
                docs.append(document)
        return docs

    def _setup_document_monitoring(self):
        """Set up file system monitoring for new documents."""
        self.input_dir = Path("_documents_for_processing_input")
        self.input_dir.mkdir(exist_ok=True)
        
        self.event_handler = DocumentHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, str(self.input_dir), recursive=False)
        self.observer.start()
        self.logger.info(f"Monitoring directory: {self.input_dir}")

    def _setup_database(self):
        """Initialize LanceDB database and table."""
        try:
            # Create database directory if it doesn't exist
            os.makedirs("lancedb", exist_ok=True)
            
            # Delete existing database if it exists
            if os.path.exists("lancedb/documents.lance"):
                self.logger.info("Removing existing database...")
                import shutil
                shutil.rmtree("lancedb/documents.lance")
            
            # Initialize database
            self.db = lancedb.connect("lancedb")
            
            # Create table with schema
            empty_chunk = ProcessedChunk.with_embeddings(self.embedding_func)(
                text="",
                metadata=ChunkMetadata(
                    filename="",
                    page_numbers=[0],
                    title="",
                    doc_type="",
                    processed_date=datetime.now(),
                    source_path="",
                    chunk_id="0"
                ),
                vector=np.zeros(self.embedding_func.ndims()).tolist()
            )
            
            self.table = self.db.create_table(
                "documents",
                data=[empty_chunk],
                mode="overwrite"
            )
            
            self.logger.info("Initialized documents table")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise

    def queue_document(self, file_path: str):
        """Queue a document for processing."""
        path = Path(file_path)
        self.logger.info(f"Queuing document for processing: {path.name}")
        self.process_document(path)

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'observer'):
            self.logger.info("Shutting down file observer...")
            self.observer.stop()
            self.observer.join()

if __name__ == "__main__":
    start_time = time.time()
    pipeline = DataPipeline()
    pipeline.process_pdf("https://arxiv.org/pdf/2408.09869")
    total_time = time.time() - start_time
    pipeline.logger.info(f"Total execution time: {total_time:.2f} seconds")
