import os
import json
import time
import logging
import pyarrow as pa
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Type, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid
import sqlite3

"""
Docling Document Processing Pipeline with CUDA Acceleration
----------------------------------------------------------
This module implements a document processing pipeline for PDF documents using Docling,
with CUDA-accelerated GPU processing. The pipeline converts documents, extracts text,
tables, and images, then processes them into searchable chunks stored in LanceDB.

Key components:
1. CUDA Acceleration:
   - Uses NVIDIA GPU via PyTorch CUDA integration (PyTorch 2.6.0+cu124)
   - Configures AcceleratorOptions with device=AcceleratorDevice.CUDA
   - Accelerates EasyOCR for text extraction and image processing
   - Optimizes document conversion and OCR tasks

2. Processing Flow:
   - PDF document ingestion
   - GPU-accelerated OCR and text extraction
   - Document chunking with HybridChunker
   - Vector embedding generation via OpenAI's text-embedding-3-large
   - Storage in LanceDB for semantic search

3. Configuration:
   - Set up in pyproject.toml with [tool.uv.sources] for PyTorch CUDA
   - Uses explicit CUDA device selection in accelerator_options
   - Falls back to CPU processing if CUDA is unavailable
"""

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    InputFormat
)
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    AcceleratorDevice,
    PdfPipelineOptions,
    EasyOcrOptions
)
from docling.datamodel.base_models import InputFormat
from docling.utils.model_downloader import download_models
import lancedb
from lancedb import vector
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import numpy as np
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer
import shutil
from dotenv import load_dotenv
import pandas as pd

# Import from the local utils folder
from utils.tokenizer import OpenAITokenizerWrapper
from utils.sitemap import get_sitemap_urls

class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    filename: str
    page_numbers: List[int]
    title: str 
    doc_type: str
    processed_date: str  # Store as ISO format string for compatibility
    source_path: str
    chunk_id: str

class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None

    def to_lance_dict(self) -> Dict[str, Any]:
        """Convert the Pydantic model to a LanceDB-compatible dictionary."""
        data = self.model_dump()  # Using model_dump() instead of dict()
        # Ensure vector is not None for LanceDB
        if data["vector"] is None:
            data["vector"] = [0.0] * 3072  # Default vector for text-embedding-3-large
        
        # Convert page_numbers to int32 to match schema
        if "metadata" in data and "page_numbers" in data["metadata"]:
            data["metadata"]["page_numbers"] = [np.int32(x) for x in data["metadata"]["page_numbers"]]
            
        return data

    @classmethod
    def get_lance_schema(cls) -> pa.Schema:
        """Get the PyArrow schema for LanceDB storage."""
        return pa.schema([
            pa.field("text", pa.string()),
            pa.field("metadata", pa.struct([
                pa.field("chunk_id", pa.string()),
                pa.field("doc_type", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("page_numbers", pa.list_(pa.int64())),  # Changed to int64 to match data
                pa.field("processed_date", pa.string()),
                pa.field("source_path", pa.string()),
                pa.field("title", pa.string())
            ])),
            pa.field("vector", vector(3072))
        ])

class ProcessedChunk(LanceModel):
    """A processed document chunk with metadata and embeddings."""
    text: str
    metadata: ChunkMetadata
    vector: Vector(3072)  # text-embedding-3-large dimension

    @classmethod
    def with_embeddings(cls, func):
        """Create a dynamic class with embedding function fields."""
        return cls  # We already have the vector field defined

class Chunks(LanceModel):
    """Schema for document chunks in LanceDB."""
    metadata: ChunkMetadata
    text: str
    vector: List[float] = Field(..., description="Embedding vector for this chunk")  # Will be set dynamically based on embedding model

class DocumentHandler(FileSystemEventHandler):
    """Handles file system events for document processing."""
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def on_created(self, event):
        if not event.is_directory:
            self.pipeline.queue_document(event.src_path)

class DataPipeline:
    def __init__(self, artifacts_path=None):
        """Initialize the document processing pipeline with CUDA GPU acceleration.
        
        The pipeline is configured to use NVIDIA CUDA for GPU-accelerated document processing,
        which significantly improves performance for OCR, image analysis, and document conversion.
        
        Args:
            artifacts_path (str, optional): Path to cached models. Defaults to None,
                which uses ~/.cache/docling/models.
        """
        # Set up logging
        import os
        from datetime import datetime
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate a timestamp-based log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"pipeline_{timestamp}.log")
        
        # Configure logging to both console and file
        logging.basicConfig(
            level=logging.DEBUG,  # Change to DEBUG level
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also log to console
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to file: {log_file}")
        
        # Load environment variables from .env
        load_dotenv()
        
        self.init_start_time = time.time()
        self.logger.info("Initializing document processing pipeline...")
        
        if artifacts_path is None:
            artifacts_path = os.path.join(str(Path.home()), '.cache', 'docling', 'models')
        
        # Download and prefetch models if needed
        self.logger.info("Checking and downloading required models...")
        self._warmup_models(artifacts_path)
        
        # Check if CUDA is available through PyTorch
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_device = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.logger.info(f"CUDA is available: {cuda_available}")
                self.logger.info(f"CUDA version: {cuda_version}")
                self.logger.info(f"GPU device: {cuda_device}")
                
                self.logger.info("Setting up CUDA acceleration...")
                # Set up CUDA acceleration
                # This configures the pipeline to use NVIDIA GPU via CUDA for faster processing
                # The AcceleratorDevice.CUDA option enables GPU acceleration for document conversion,
                # OCR (optical character recognition), and image processing tasks
                self.accelerator_options = AcceleratorOptions(
                    num_threads=8, 
                    device=AcceleratorDevice.CUDA  # Use CUDA acceleration
                )
            else:
                self.logger.warning("CUDA is not available, falling back to CPU processing")
                self.accelerator_options = AcceleratorOptions(
                    num_threads=8, 
                    device=AcceleratorDevice.CPU  # Fall back to CPU
                )
        except ImportError:
            self.logger.warning("PyTorch not found or CUDA support not available, using CPU")
            self.accelerator_options = AcceleratorOptions(
                num_threads=8, 
                device=AcceleratorDevice.CPU  # Use CPU as fallback
            )

        # Configure pipeline options with appropriate acceleration
        self.logger.info("Configuring pipeline options...")
        self.pipeline_options = PdfPipelineOptions(artifacts_path=artifacts_path)
        self.pipeline_options.accelerator_options = self.accelerator_options
        
        # Configure OCR options with appropriate acceleration
        device_type = "CUDA" if getattr(self.accelerator_options, "device", None) == AcceleratorDevice.CUDA else "CPU"
        self.logger.info(f"Configuring EasyOCR with {device_type} acceleration...")
        self.pipeline_options.do_ocr = True
        self.pipeline_options.ocr_options = EasyOcrOptions()
        
        # Enable table structure analysis
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True

        # Create converter with appropriate acceleration options
        self.logger.info(f"Creating document converter with {device_type} acceleration...")
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )
        
        # Initialize database
        lancedb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lancedb_data")
        self.logger.info(f"Using LanceDB storage at: {lancedb_path}")
        self.db = lancedb.connect(lancedb_path)
        
        # Initialize embedding function
        self.logger.info("Initializing embedding function...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        # Get the embedding registry
        registry = get_registry()
        
        # Try multiple initialization approaches
        try:
            # First attempt: Try using variable_store
            if hasattr(registry, 'variable_store'):
                self.logger.info("Using variable_store for API key")
                registry.variable_store.set("OPENAI_API_KEY", api_key)
                self.embedding_func = registry.get("openai").create(
                    name="text-embedding-3-large"
                )
                self.logger.info("Successfully initialized embedding model with variable_store")
                return
                
            # Second attempt: Check if we can use environment variable reference instead of hardcoded value
            self.logger.info("Trying with environment variable reference")
            try:
                self.embedding_func = registry.get("openai").create(
                    name="text-embedding-3-large",
                    api_key="$env:OPENAI_API_KEY"  # Use environment variable reference
                )
                self.logger.info("Successfully initialized embedding model with env var reference")
                return
            except Exception as e:
                self.logger.warning(f"Environment variable reference approach failed: {str(e)}")
            
            # Last resort: Fall back to direct OpenAI client
            self.logger.info("Falling back to direct OpenAI client")
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Test if the client works by making a sample embedding request
            response = client.embeddings.create(
                input="Test embedding",
                model="text-embedding-3-large"
            )
            
            # If we get here, client is working
            self.embedding_func = client
            self.is_direct_client = True  # Flag to indicate we're using direct client
            self.logger.info("Successfully initialized direct OpenAI client")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise ValueError(f"Could not initialize embedding function: {str(e)}")
        
        # We also need to update other functions to handle direct client if used
        
        self.tokenizer = OpenAITokenizerWrapper()
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=8191,  # text-embedding-3-large's maximum context length
            merge_peers=True  # Merge undersized successive chunks with same headings
        )
        
        # Setup tracking database for processed files
        self._setup_tracking_database()
        
        init_time = time.time() - self.init_start_time
        self.logger.info(f"Pipeline initialization time: {init_time:.2f} seconds")
        self.logger.info("Pipeline initialization complete. Ready for processing.")

    def _setup_tracking_database(self):
        """Set up or connect to the tracking database."""
        self.tracking_db_path = "processed_files.db"
        self.conn = sqlite3.connect(self.tracking_db_path)
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_files (
            file_path TEXT PRIMARY KEY,
            filename TEXT,
            processed_date TEXT,
            chunk_count INTEGER,
            status TEXT
        )
        ''')
        self.conn.commit()
        
    def is_file_processed(self, file_path):
        """Check if a file has already been processed."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM processed_files WHERE file_path = ?", (str(file_path),))
        return cursor.fetchone() is not None

    def mark_file_processed(self, file_path, chunk_count, status="completed"):
        """Mark a file as processed in the tracking database."""
        cursor = self.conn.cursor()
        filename = os.path.basename(file_path)
        processed_date = datetime.now().isoformat()
        cursor.execute(
            "INSERT OR REPLACE INTO processed_files VALUES (?, ?, ?, ?, ?)",
            (str(file_path), filename, processed_date, chunk_count, status)
        )
        self.conn.commit()

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
            if "chunks" not in self.db.table_names():
                self.logger.warning("No 'chunks' table exists in the database")
                return False
                
            table = self.db.open_table("chunks")
            
            # Count total rows in a version-compatible way
            try:
                # Try using len() first (works in most versions)
                chunk_count = len(table)
            except Exception as e:
                self.logger.warning(f"Error using len(table): {str(e)}")
                # Fallback to counting using to_arrow() or to_pandas()
                try:
                    chunk_count = len(table.to_arrow())
                except Exception:
                    try:
                        chunk_count = len(table.to_pandas())
                    except Exception:
                        # Final fallback - try search with a dummy vector that will match everything
                        try:
                            # Create a dummy vector of the same dimension as our embeddings
                            dummy_vec = [0.0] * 3072  # text-embedding-3-large dimension
                            # Use search with a very high limit and count results
                            results = table.search(dummy_vec).limit(10000).to_list()
                            chunk_count = len(results)
                        except Exception as e2:
                            self.logger.error(f"Failed to count chunks: {str(e2)}")
                            return False
            
            self.logger.info(f"Found {chunk_count} chunks in LanceDB table")
            
            # Get sample data for diagnostics (using search with limit)
            try:
                # Try the regular search method first
                dummy_vec = [0.0] * 3072  # text-embedding-3-large dimension
                sample_data = table.search(dummy_vec).limit(5).to_list()
                
                if len(sample_data) > 0:
                    sample_df = pd.DataFrame(sample_data)
                else:
                    # If no results, try another approach
                    sample_df = pd.DataFrame()
            except Exception as e:
                self.logger.warning(f"Error using search().to_list(): {str(e)}")
                # Fallback method
                try:
                    sample_df = table.query().limit(5).to_pandas()
                except Exception as e2:
                    self.logger.warning(f"Error using query().to_pandas(): {str(e2)}")
                    sample_df = pd.DataFrame()  # Empty dataframe as fallback
            
            # Print diagnostic information for the first chunk
            if not sample_df.empty:
                self.logger.info("Sample chunk data:")
                sample_chunk = sample_df.iloc[0]
                self.logger.info(f"Available columns: {list(sample_df.columns)}")
                
                # Safely extract metadata
                if 'metadata' in sample_df.columns:
                    self.logger.info(f"Sample chunk metadata: {sample_chunk.get('metadata', 'No metadata')}")
                
                # Safely extract text
                if 'text' in sample_df.columns:
                    text_sample = sample_chunk.get('text', 'No text')
                    if text_sample:
                        text_preview = text_sample[:200] + '...' if len(text_sample) > 200 else text_sample
                        self.logger.info(f"Sample chunk text: {text_preview}")
                    else:
                        self.logger.info("Sample chunk text: No text")
                
                # Safely extract vector
                if 'vector' in sample_df.columns:
                    vector = sample_chunk.get('vector')
                    if vector is not None:
                        vector_shape = len(vector) if isinstance(vector, (list, np.ndarray)) else 'Not a vector'
                        self.logger.info(f"Sample chunk vector shape: {vector_shape}")
                    else:
                        self.logger.info("Sample chunk vector: None")
            else:
                self.logger.warning("Could not retrieve sample data from chunks table")
            
            if expected_count is not None and chunk_count != expected_count:
                self.logger.warning(f"Expected {expected_count} chunks but found {chunk_count}")
                return False
            
            # Basic validation passed
            return chunk_count > 0
            
        except Exception as e:
            self.logger.error(f"Error verifying chunks: {str(e)}")
            return False

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for a text string."""
        try:
            if hasattr(self, 'is_direct_client') and self.is_direct_client:
                # Using direct OpenAI client
                response = self.embedding_func.embeddings.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                return response.data[0].embedding
            else:
                # Using LanceDB registry model
                return self.embedding_func.generate_embeddings([text])[0]
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise

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
            
            # Store chunks in LanceDB
            self.logger.info("Storing chunks in LanceDB...")
            self.logger.info(f"Processing {len(chunks)} chunks")
            
            # Prepare chunks for LanceDB
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    # Extract page numbers safely
                    page_numbers = []
                    if chunk.meta.doc_items:
                        for item in chunk.meta.doc_items:
                            if hasattr(item.prov, 'page_no'):
                                page_numbers.append(item.prov.page_no)
                    
                    if not page_numbers:
                        page_numbers = [0]  # Default if no page numbers found
                    
                    metadata = {
                        "filename": chunk.meta.origin.filename or file_path.name,
                        "page_numbers": page_numbers,
                        "title": chunk.meta.headings[0] if chunk.meta.headings else "",
                        "doc_type": file_path.suffix[1:] or "unknown",
                        "processed_date": datetime.now().isoformat(),
                        "source_path": str(file_path),
                        "chunk_id": str(i)
                    }
                    processed_chunk = DocumentChunk(
                        text=chunk.text or "empty",
                        metadata=metadata,
                        vector=self._generate_embedding(chunk.text or "empty")
                    )
                    processed_chunks.append(processed_chunk)
                    self.logger.debug(f"Successfully processed chunk {i+1}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    raise

            self.process_chunks(processed_chunks)
            
            # Mark file as processed
            self.mark_file_processed(str(file_path), len(processed_chunks))
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def process_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Process and store document chunks in the vector database."""
        if not chunks:
            self.logger.warning("No chunks to process")
            return
            
        try:
            # Validate chunks before converting to LanceDB format
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    # Ensure chunk has a vector of the right dimension
                    if chunk.vector is None or len(chunk.vector) != 3072:
                        self.logger.warning(f"Chunk {i}: Invalid vector dimension, regenerating embedding")
                        if chunk.text:
                            chunk.vector = self._generate_embedding(chunk.text)
                        else:
                            chunk.vector = [0.0] * 3072  # Default vector if no text
                            
                    # Ensure metadata has required fields
                    if not chunk.metadata:
                        self.logger.warning(f"Chunk {i}: Missing metadata, adding placeholder")
                        chunk.metadata = {
                            "filename": "unknown.pdf",
                            "page_numbers": [0],
                            "title": "Untitled",
                            "doc_type": "unknown",
                            "processed_date": datetime.now().isoformat(),
                            "source_path": "unknown",
                            "chunk_id": str(uuid.uuid4())
                        }
                        
                    valid_chunks.append(chunk)
                except Exception as e:
                    self.logger.error(f"Error validating chunk {i}: {str(e)}")
            
            if not valid_chunks:
                self.logger.error("No valid chunks after validation")
                return
                
            # Convert chunks to LanceDB format
            lance_data = [chunk.to_lance_dict() for chunk in valid_chunks]
            self.logger.info(f"Converted {len(valid_chunks)} valid chunks to LanceDB format")
            
            # Check if table exists
            if "chunks" in self.db.table_names():
                # If table exists, use append mode
                table = self.db.open_table("chunks")
                try:
                    table.add(data=lance_data)
                    self.logger.info(f"Added {len(valid_chunks)} chunks to existing table")
                except Exception as e:
                    self.logger.error(f"Error adding chunks to table: {str(e)}")
                    # Try one more time after a short delay
                    time.sleep(2)
                    table.add(data=lance_data)
                    self.logger.info(f"Successfully added chunks on second attempt")
            else:
                # If table doesn't exist, create it
                try:
                    schema = DocumentChunk.get_lance_schema()
                    self.db.create_table(
                        name="chunks",
                        data=lance_data,
                        schema=schema,
                        mode="create"
                    )
                    self.logger.info(f"Created new table with {len(valid_chunks)} chunks")
                except Exception as e:
                    self.logger.error(f"Error creating table: {str(e)}")
                    raise
            
            # Verify chunk count
            try:
                table = self.db.open_table("chunks")
                stored_count = len(table)
                self.logger.info(f"Successfully stored chunks in the database (total: {stored_count})")
            except Exception as e:
                self.logger.warning(f"Could not verify chunk count: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error processing chunks: {str(e)}")
            raise

    def process_pdf(self, url):
        """Process a PDF document through the full pipeline."""
        start_time = time.time()
        self.logger.info("Starting process_pdf...")
        
        # Download and convert document
        document = self._download_and_convert_pdf(url)
        
        # Process document through chunking pipeline
        self.logger.info("Processing document through chunking pipeline...")
        chunks = list(self.chunker.chunk(dl_doc=document))
        
        # Enhanced chunk type diagnostics
        chunk_types = {}
        chunk_content = {'with_text': 0, 'without_text': 0}
        for chunk in chunks:
            chunk_type = type(chunk.meta.doc_items[0]).__name__ if chunk.meta.doc_items else "Unknown"
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            if chunk.text:
                chunk_content['with_text'] += 1
            else:
                chunk_content['without_text'] += 1
        
        self.logger.info(f"Created {len(chunks)} chunks with types: {chunk_types}")
        self.logger.info(f"Text content breakdown: {chunk_content}")
        
        # Store chunks in LanceDB
        self.logger.info("Storing chunks in LanceDB...")
        self.logger.info(f"Processing {len(chunks)} chunks")
        
        # Prepare chunks for LanceDB
        processed_chunks = []
        batch_size = 8
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1} of {(len(chunks) + batch_size - 1)//batch_size + 1}")
            
            # Process each chunk in the batch
            batch_chunks = []
            for chunk in batch:
                # Log chunk structure for debugging
                self.logger.info(f"Chunk meta structure: {chunk.meta}")
                self.logger.info(f"Doc items structure: {chunk.meta.doc_items}")
                if chunk.meta.doc_items:
                    self.logger.info(f"First doc item structure: {chunk.meta.doc_items[0]}")
                    self.logger.info(f"First doc item prov: {chunk.meta.doc_items[0].prov}")

                # Ensure we have non-null values for all required fields
                metadata = {
                    "filename": chunk.meta.origin.filename or "unknown.pdf",
                    "page_numbers": [prov.page_no for item in chunk.meta.doc_items for prov in item.prov] or [0],
                    "title": chunk.meta.headings[0] if chunk.meta.headings else "Untitled",
                    "doc_type": "pdf",  # Since we're in process_pdf
                    "processed_date": datetime.now().isoformat(),
                    "source_path": chunk.meta.origin.uri or url,
                    "chunk_id": str(i)
                }
                
                # Create processed chunk with embeddings
                processed_chunk = DocumentChunk(
                    text=chunk.text or "empty",
                    metadata=metadata,
                    vector=self._generate_embedding(chunk.text or "empty")
                )
                batch_chunks.append(processed_chunk)
            
            processed_chunks.extend(batch_chunks)
        
        self.process_chunks(processed_chunks)
        
        process_time = time.time() - start_time
        self.logger.info(f"Processing time for process_pdf: {process_time:.2f} seconds")
        self.logger.info(f"Saved markdown to _processed_output\\{os.path.basename(url).replace('.pdf', '')}.md and JSON to _processed_output\\{os.path.basename(url).replace('.pdf', '')}.json")
    
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

    def process_sitemap(self, url):
        sitemap_urls = get_sitemap_urls(url)
        conv_results_iter = self.converter.convert_all(sitemap_urls)

        docs = []
        for result in conv_results_iter:
            if result.document:
                document = result.document
                docs.append(document)
        return docs

    def _download_and_convert_pdf(self, url):
        """Download and convert a PDF document."""
        self.logger.info("Going to convert document batch...")
        result = self.converter.convert(url)
        document = result.document
        
        # Save outputs to _processed_output directory
        os.makedirs('_processed_output', exist_ok=True)
        base_name = os.path.basename(url).replace('.pdf', '')
        
        # Export to markdown and JSON
        markdown_output = document.export_to_markdown()
        json_output = document.export_to_dict()
        
        # Save the outputs
        markdown_file = os.path.join('_processed_output', f'{base_name}.md')
        json_file = os.path.join('_processed_output', f'{base_name}.json')
        
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)
            
        self.logger.info(f"Finished converting document {os.path.basename(url)}")
        return document

    def _setup_document_monitoring(self):
        """Set up file system monitoring for new documents."""
        self.input_dir = Path("_documents_for_processing_input")
        self.input_dir.mkdir(exist_ok=True)
        
        self.event_handler = DocumentHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, str(self.input_dir), recursive=False)
        self.observer.start()
        self.logger.info(f"Monitoring directory: {self.input_dir}")

    def queue_document(self, file_path: str):
        """Queue a document for processing."""
        path = Path(file_path)
        self.logger.info(f"Queuing document for processing: {path.name}")
        if not self.is_file_processed(file_path):
            self.process_document(path)
        else:
            self.logger.info(f"Skipping already processed file: {path.name}")

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'observer'):
            self.logger.info("Shutting down file observer...")
            self.observer.stop()
            self.observer.join()

if __name__ == "__main__":
    start_time = time.time()
    pipeline = DataPipeline()
    
    # Check for command line argument to enable monitoring
    import sys
    enable_monitoring = "--monitor" in sys.argv
    
    # Process all new documents in the input directory
    input_dir = Path(r"C:\Users\kevin\repos\docling-playground\_documents_for_processing_input")
    new_files_processed = 0
    
    # Process existing files first
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"\nProcessing file: {pdf_file.name}")
        try:
            pipeline.process_document(pdf_file)
            new_files_processed += 1
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
            pipeline.mark_file_processed(str(pdf_file), 0, status="error")
    
    # Verify chunks were stored
    pipeline.verify_lancedb_chunks()
    
    end_time = time.time()
    print(f"Processed {new_files_processed} files")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    # Only enter monitoring mode if specifically requested
    if enable_monitoring:
        # Enable file monitoring for real-time processing
        pipeline._setup_document_monitoring()
        print("File monitoring is active. Press Ctrl+C to exit.")
        try:
            # Keep the program running to monitor for new files
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
    else:
        print("Processing complete. Exiting gracefully.")
