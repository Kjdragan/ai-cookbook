import os
import sys
import uuid
import time
import sqlite3
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from types import SimpleNamespace
from typing import Dict, List, Optional, Union, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import lancedb
import pyarrow as pa
import torch
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel, Field

# Import docling components
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
from lancedb import vector
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
import shutil
from dotenv import load_dotenv

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

class ProcessedChunk(BaseModel):
    """Pydantic model for processed document chunks with metadata."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_type: str = "unknown"
    filename: str = "unknown"
    page_numbers: List[int] = Field(default_factory=list)
    primary_page: int = 0
    word_count: int = 0
    character_count: int = 0
    title: str = ""
    content_type: str = "text"
    processed_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    source_path: str = ""
    vector: List[float] = Field(default_factory=list)
    text: str = ""
    
    class Config:
        arbitrary_types_allowed = True

class ProcessedChunkLance(LanceModel):
    """Schema for processed document chunks in LanceDB."""
    metadata: ProcessedChunk
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
        import codecs
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate a timestamp-based log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"pipeline_{timestamp}.log")
        console_log_file = os.path.join(logs_dir, f"console_output_{timestamp}.log")
        
        # Custom logging handler to handle Unicode encoding errors
        class UnicodeCompatibleStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    # Replace problematic characters with their Unicode escape sequences
                    safe_msg = msg.encode('ascii', 'backslashreplace').decode('ascii')
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        class UnicodeCompatibleFileHandler(logging.FileHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Write with UTF-8 encoding and backslashreplace for unsupported chars
                    with codecs.open(self.baseFilename, 'a', encoding='utf-8', errors='backslashreplace') as f:
                        f.write(msg + self.terminator)
                except Exception:
                    self.handleError(record)
        
        # Configure logging to both console and file with Unicode compatibility
        logging.basicConfig(
            level=logging.DEBUG,  # Change to DEBUG level
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                UnicodeCompatibleFileHandler(log_file),
                UnicodeCompatibleStreamHandler()  # Also log to console with Unicode handling
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to file: {log_file}")
        
        # Also capture console output to a file
        console_file_handler = UnicodeCompatibleFileHandler(console_log_file)
        console_logger = logging.getLogger("")  # Root logger
        console_logger.addHandler(console_file_handler)
        self.logger.info(f"Console output captured to: {console_log_file}")
        
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
        self.lancedb_url = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lancedb_data")
        self.logger.info(f"Using LanceDB storage at: {self.lancedb_url}")
        self.db = lancedb.connect(self.lancedb_url)
        
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
            raise

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

    def verify_database(self) -> bool:
        """Verify database contents and display sample data."""
        try:
            # Connect to the database
            db = lancedb.connect(self.lancedb_url)
            
            # Check if chunks table exists
            if "chunks" not in db.table_names():
                self.logger.warning("No chunks table exists in the database.")
                return False
            
            # Open the table
            table = db.open_table("chunks")
            
            # Get the number of rows
            num_chunks = len(table)
            self.logger.info(f"Found {num_chunks} chunks in LanceDB table")
            
            if num_chunks == 0:
                self.logger.warning("Table exists but has no data.")
                return True
            
            # Get a sample row
            sample_df = table.to_pandas().head(1)
            
            # Log available columns
            self.logger.info(f"Available columns: {sample_df.columns.tolist()}")
            
            # Log sample metadata
            if 'metadata' in sample_df.columns:
                self.logger.info(f"Sample chunk metadata: {sample_df['metadata'].iloc[0]}")
                
                # Extract and log page number statistics
                all_df = table.to_pandas()
                page_number_stats = {}
                primary_page_stats = {}
                content_type_stats = {}
                
                # Count occurrences of each page number
                for _, row in all_df.iterrows():
                    if 'page_numbers' in row['metadata']:
                        for page in row['metadata']['page_numbers']:
                            if page not in page_number_stats:
                                page_number_stats[page] = 0
                            page_number_stats[page] += 1
                    
                    if 'primary_page' in row['metadata']:
                        primary_page = row['metadata']['primary_page']
                        if primary_page not in primary_page_stats:
                            primary_page_stats[primary_page] = 0
                        primary_page_stats[primary_page] += 1
                    
                    if 'content_type' in row['metadata']:
                        content_type = row['metadata']['content_type']
                        if content_type not in content_type_stats:
                            content_type_stats[content_type] = 0
                        content_type_stats[content_type] += 1
                
                self.logger.info(f"Page number distribution: {dict(sorted(page_number_stats.items()))}")
                self.logger.info(f"Primary page distribution: {dict(sorted(primary_page_stats.items()))}")
                self.logger.info(f"Content type distribution: {content_type_stats}")
            
            # Log a sample of the text
            if 'text' in sample_df.columns:
                sample_text = sample_df['text'].iloc[0]
                preview_length = min(150, len(sample_text))
                self.logger.info(f"Sample chunk text: {sample_text[:preview_length]}...")
            
            # Check if vector is present and has the right dimension
            if 'vector' in sample_df.columns:
                vector_shape = len(sample_df['vector'].iloc[0])
                self.logger.info(f"Sample chunk vector shape: {vector_shape}")
                if vector_shape != 3072:
                    self.logger.warning(f"Expected vector dimension of 3072, but found {vector_shape}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error verifying chunks: {str(e)}")
            return False

    def add_chunks(self, chunks):
        """Add processed chunks to the LanceDB database."""
        try:
            import pyarrow as pa
            import lancedb
            import json
            
            # Connect to db
            db_path = self.lancedb_url
            db = lancedb.connect(db_path)
            
            num_chunks = len(chunks)
            self.logger.info(f"Adding {num_chunks} chunks to database")
            
            # Process each chunk to add embedding
            vectorized_chunks = []
            
            # Create a table if it doesn't exist
            if "chunks" not in db.table_names():
                self.logger.info("Creating new 'chunks' table")
                schema = self._get_lancedb_schema()
                
                # Create empty DataFrame with the schema
                empty_df = pa.Table.from_pylist([], schema=schema)
                
                # Create table with schema
                db.create_table("chunks", empty_df)
            
            # Use existing table
            table = db.open_table("chunks")
            
            # Track how many chunks we've processed
            self.logger.info(f"Adding {len(chunks)} chunks to existing table")
            
            # In the new schema, chunks are already in the correct format
            # Just need to sanitize and ensure correct types
            sanitized_chunks = []
            for chunk in chunks:
                # Ensure all types are compatible with PyArrow/LanceDB
                sanitized_chunks.append(chunk)
            
            # Add chunks to table
            try:
                # Convert list of dicts to PyArrow Table
                chunks_table = pa.Table.from_pylist(sanitized_chunks)
                table.add(chunks_table)
                self.logger.info(f"Successfully added {len(chunks)} chunks to database")
            except Exception as e:
                self.logger.error(f"Error adding chunks to table: {str(e)}")
                raise
            
            return True
        except Exception as e:
            self.logger.error(f"Error processing chunks: {str(e)}")
            raise

    def process_chunks(self, chunks: List[Dict]) -> None:
        """Process and store document chunks in the vector database."""
        if not chunks:
            self.logger.warning("No chunks to process")
            return
        
        try:
            import pyarrow as pa
            import lancedb
            
            # Connect to LanceDB database
            db = lancedb.connect(self.lancedb_url)
            
            # Create or get the chunks table
            table_name = "chunks"
            
            # Track how many chunks we've processed
            self.logger.info(f"Processing {len(chunks)} chunks")
            
            # Create the table if it doesn't exist
            if table_name not in db.table_names():
                schema = self._get_lancedb_schema()
                empty_df = pa.Table.from_pylist([], schema=schema)
                table = db.create_table(table_name, empty_df)
                self.logger.info(f"Created new table '{table_name}'")
            else:
                table = db.open_table(table_name)
                self.logger.debug(f"Using existing table '{table_name}'")
            
            # Process each document chunk
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    processed_chunk = self.process_chunk(chunk)
                    
                    # Verify processed chunk has valid vector before adding
                    if not processed_chunk.get("vector") or len(processed_chunk.get("vector", [])) != 3072:
                        self.logger.warning(f"Chunk {i} has invalid vector, fixing with zero vector")
                        processed_chunk["vector"] = [0.0] * 3072
                        
                    processed_chunks.append(processed_chunk)
                    if (i + 1) % 10 == 0:
                        self.logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i}: {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
            
            # Add the processed chunks to the database
            if processed_chunks:
                try:
                    # Final validation pass to ensure all vectors are valid before db insertion
                    validated_chunks = []
                    for i, chunk in enumerate(processed_chunks):
                        if not chunk.get("vector") or len(chunk.get("vector", [])) != 3072:
                            self.logger.warning(f"Final validation: Chunk {i} has invalid vector, fixing with zero vector")
                            chunk["vector"] = [0.0] * 3072
                        validated_chunks.append(chunk)
                    
                    # Add chunks to database, handling field order compatibility
                    self.add_chunks(validated_chunks)
                    self.logger.info(f"Successfully added {len(validated_chunks)} chunks to database")
                except Exception as e:
                    self.logger.error(f"Error adding chunks to database: {str(e)}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
            else:
                self.logger.warning("No chunks were successfully processed")
                
            return len(processed_chunks)
        except Exception as e:
            self.logger.error(f"Error in process_chunks: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def _get_lancedb_schema(self):
        """Define the PyArrow schema for LanceDB tables."""
        import pyarrow as pa
        
        # Define schema - IMPORTANT: maintain consistent field order with existing tables
        return pa.schema([
            # Primary fields - order is critical
            ("chunk_id", pa.string()),
            ("doc_type", pa.string()),
            ("filename", pa.string()),
            ("page_numbers", pa.list_(pa.int64())),
            ("primary_page", pa.int64()),
            ("word_count", pa.int64()),
            ("character_count", pa.int64()),
            ("title", pa.string()),
            ("content_type", pa.string()),
            ("processed_date", pa.string()),
            ("source_path", pa.string()),
            # Vector field always last
            ("vector", pa.list_(pa.float32(), 3072)),
            ("text", pa.string())
        ])

    def process_chunk(self, chunk, text_embedder=None):
        """Process a single chunk into the format required for the database."""
        try:
            # Assign a unique chunk ID
            chunk_id = str(uuid.uuid4())
            
            # Try to extract page numbers from chunk metadata
            page_numbers, total_pages = self._extract_page_numbers(chunk) 
            
            # Handle missing page numbers
            if page_numbers is None:
                page_numbers = [0]
            elif isinstance(page_numbers, int):
                page_numbers = [page_numbers]
            
            # Get chunk text and sanitize it to prevent encoding errors
            chunk_text = chunk.text if hasattr(chunk, 'text') else ""
            sanitized_text = self._sanitize_text(chunk_text)
            
            # Log chunk information safely
            self.logger.debug(f"Processing chunk {chunk_id[:8]} with {len(sanitized_text)} characters")
            
            # Detect content type
            content_type = self._detect_content_type(sanitized_text)
            
            # Set filename and title
            filename = "unknown.pdf"
            title = "Untitled Document"
            source_path = ""
            
            if hasattr(chunk, 'meta'):
                if hasattr(chunk.meta, 'origin'):
                    if hasattr(chunk.meta.origin, 'filename') and chunk.meta.origin.filename:
                        filename = chunk.meta.origin.filename
                    if hasattr(chunk.meta.origin, 'uri') and chunk.meta.origin.uri:
                        source_path = chunk.meta.origin.uri
                
                # Extract title from headings
                if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                    title = self._sanitize_text(chunk.meta.headings[0])
            
            # Initialize the vector with zeros - ALWAYS have a default vector
            zero_vector = [0.0] * 3072
            
            # Create the processed chunk with default zero vector
            processed_chunk = ProcessedChunk(
                chunk_id=chunk_id,
                doc_type="pdf",
                filename=filename,
                page_numbers=page_numbers,
                primary_page=page_numbers[0] if page_numbers else 0,
                word_count=len(sanitized_text.split()),
                character_count=len(sanitized_text),
                title=title,
                content_type=content_type,
                processed_date=datetime.now().isoformat(),
                source_path=source_path,
                text=sanitized_text,
                vector=zero_vector  # Initialize with zeros by default
            )
            
            # Generate embedding only if text exists
            if sanitized_text:
                try:
                    # Always use the class's _generate_embedding method
                    # This handles both direct client and registry approaches
                    embedding = self._generate_embedding(sanitized_text)
                    
                    # Ensure embedding is not None and has correct dimensions
                    if embedding and len(embedding) == 3072:
                        processed_chunk.vector = embedding
                    else:
                        self.logger.warning(f"Invalid embedding returned for chunk {chunk_id[:8]}, using zero vector")
                        # Keep the default zero vector
                except Exception as e:
                    self.logger.error(f"Error generating embedding: {str(e)}")
                    # Keep the default zero vector
            
            # Return as dictionary for compatibility with PyArrow and LanceDB
            result = processed_chunk.model_dump()
            
            # Double check that vector is never null before returning
            if not result["vector"] or len(result["vector"]) != 3072:
                self.logger.warning(f"Null or invalid vector detected in final output for chunk {chunk_id[:8]}, fixing with zero vector")
                result["vector"] = zero_vector
                
            return result
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
            # Create a minimal valid chunk as fallback and return as dictionary
            fallback_chunk = ProcessedChunk(
                chunk_id=str(uuid.uuid4()),
                text="Error processing chunk",
                vector=[0.0] * 3072
            )
            return fallback_chunk.model_dump()

    def convert_document(self, file_path: Path):
        """Convert a document to chunks using the document converter."""
        try:
            # Convert the document using Docling converter
            result = self.converter.convert(str(file_path))
            
            # Add debug logging for document structure
            self.logger.debug(f"Document structure overview: {result.document.__class__.__name__}")
            doc_attrs = dir(result.document)
            self.logger.debug(f"Document attributes: {[attr for attr in doc_attrs if not attr.startswith('_')]}")
            
            # Apply chunking
            self.logger.info("Chunking document...")
            chunks = list(self.chunker.chunk(dl_doc=result.document))
            self.logger.info(f"Created {len(chunks)} chunks")
            
            # Log chunk statistics
            chunk_sizes = [len(chunk.text) for chunk in chunks]
            if chunk_sizes:
                avg_size = sum(chunk_sizes) / len(chunk_sizes)
                self.logger.debug(f"Average chunk size: {avg_size:.2f} characters")
                self.logger.debug(f"Min chunk size: {min(chunk_sizes)} characters")
                self.logger.debug(f"Max chunk size: {max(chunk_sizes)} characters")
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error converting document: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            raise

    def process_document(self, file_path: Path):
        """Process a single document through the pipeline with retry logic.
        
        This method handles the full document processing workflow:
        1. Checks if file has already been processed
        2. Converts document to text chunks using GPU-accelerated document conversion
        3. Processes each chunk to extract metadata and generate embeddings
        4. Stores processed chunks in the vector database
        5. Tracks document processing status
        
        Args:
            file_path (Path): Path to the document file
            
        Returns:
            None: The method stores processed chunks in the database
        """
        try:
            file_path_str = str(file_path)
            self.logger.info(f"Starting document processing: {file_path.name}")
            
            # Check if file has already been processed
            if self.is_file_processed(file_path_str):
                self.logger.info(f"Skipping already processed file: {file_path.name}")
                return
                
            # Convert document to text chunks
            self.logger.info(f"Converting document {file_path.name}...")
            document_chunks = self.convert_document(file_path)
            
            if not document_chunks:
                self.logger.warning(f"No chunks were generated from document: {file_path.name}")
                self.mark_file_processed(file_path_str, 0, status="error")
                return
                
            self.logger.info(f"Generated {len(document_chunks)} chunks from document")
            
            # Log chunk statistics
            if len(document_chunks) > 0:
                chunk_sizes = [len(getattr(chunk, 'text', '')) for chunk in document_chunks]
                if chunk_sizes:
                    avg_size = sum(chunk_sizes) / len(chunk_sizes)
                    self.logger.info(f"Chunk statistics: avg={avg_size:.1f}, min={min(chunk_sizes)}, max={max(chunk_sizes)} chars")
            
            # Process chunks in batches and store in database 
            # This reduces memory usage and provides more frequent progress updates
            batch_size = 8  # Process in small batches for better progress tracking
            processed_chunks = []
            total_chunks = len(document_chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch = document_chunks[i:i+batch_size]
                self.logger.info(f"Processing chunk batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}")
                
                batch_processed = []
                for j, chunk in enumerate(batch):
                    chunk_index = i + j
                    self.logger.debug(f"Processing chunk {chunk_index+1}/{total_chunks}")
                    try:
                        processed_chunk = self.process_chunk(chunk, self.embedding_func)
                        batch_processed.append(processed_chunk)
                        self.logger.debug(f"Successfully processed chunk {chunk_index+1}")
                    except Exception as e:
                        self.logger.error(f"Error processing chunk {chunk_index+1}: {str(e)}")
                        import traceback
                        self.logger.debug(traceback.format_exc())
                
                # Add this batch to the accumulated processed chunks
                processed_chunks.extend(batch_processed)
                
                # Log progress
                self.logger.info(f"Progress: {len(processed_chunks)}/{total_chunks} chunks processed")
            
            # Store chunks in the vector database
            if processed_chunks:
                self.logger.info(f"Adding {len(processed_chunks)} processed chunks to vector database...")
                self.add_chunks(processed_chunks)
                
                # Mark the file as processed in SQLite
                num_chunks = len(processed_chunks)
                self.mark_file_processed(file_path_str, num_chunks)
                self.logger.info(f"Successfully processed document with {num_chunks} chunks")
                
                # Return success status and chunks
                return {"status": "success", "chunks": num_chunks}
            else:
                self.logger.warning(f"No chunks were processed for {file_path.name}")
                self.mark_file_processed(file_path_str, 0, status="error")
                return {"status": "error", "chunks": 0}
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
            # Mark file as having an error
            self.mark_file_processed(str(file_path), 0, status="error")
            return {"status": "error", "message": str(e)}

    def process_pdf(self, url):
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

                # Extract page numbers safely
                page_numbers = []
                if chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, 'prov'):
                            # Handle both single prov and list of provs
                            provs = item.prov if isinstance(item.prov, list) else [item.prov]
                            for prov in provs:
                                if hasattr(prov, 'page_no'):
                                    page_numbers.append(prov.page_no)
                    
                    # Remove duplicates and sort
                    if page_numbers:
                        page_numbers = sorted(set(page_numbers))
                
                if not page_numbers:
                    self.logger.debug(f"No page numbers found for chunk {i}, using default [0]")
                    page_numbers = [0]  # Default if no page numbers found
                else:
                    self.logger.debug(f"Found page numbers for chunk {i}: {page_numbers}")

                # Ensure we have non-null values for all required fields
                metadata = {
                    "filename": chunk.meta.origin.filename or "unknown.pdf",
                    "page_numbers": page_numbers,
                    "title": chunk.meta.headings[0] if chunk.meta.headings else "Untitled",
                    "doc_type": "pdf",  
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

    def reset_database(self):
        """Reset LanceDB and SQLite tracking database for fresh tests."""
        self.logger.info("Resetting LanceDB...")
        try:
            # Connect to LanceDB
            db = lancedb.connect(self.lancedb_url)
            
            # Drop the chunks table if it exists
            if "chunks" in db.table_names():
                db.drop_table("chunks")
                self.logger.info("LanceDB 'chunks' table dropped")
        except Exception as e:
            self.logger.error(f"Error resetting LanceDB: {str(e)}")
        
        # Clear SQLite tracking database
        self.logger.info("Resetting SQLite tracking database...")
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.tracking_db_path)
            cursor = conn.cursor()
            
            # Clear the processed_files table
            cursor.execute("DELETE FROM processed_files")
            conn.commit()
            self.logger.info("SQLite 'processed_files' table cleared")
            
            # Close connection
            conn.close()
        except Exception as e:
            self.logger.error(f"Error resetting SQLite database: {str(e)}")
        
        self.logger.info("Database reset complete")

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'observer'):
            self.logger.info("Shutting down file observer...")
            self.observer.stop()
            self.observer.join()

    def _sanitize_text(self, text):
        """Sanitize text by handling Unicode characters safely.
        
        This helps prevent UnicodeEncodeError when dealing with special characters
        in environments with limited character encoding support (e.g., Windows consoles).
        
        Args:
            text (str): Text to sanitize
            
        Returns:
            str: Sanitized text with problematic characters replaced
        """
        if not text:
            return ""
            
        try:
            # Test if text can be encoded to ASCII
            text.encode('ascii')
            return text  # If it works, return original text
        except UnicodeEncodeError:
            # Replace non-ASCII characters with their Unicode escape sequences or closest ASCII equivalents
            text = text.encode('ascii', 'namereplace').decode('ascii')
            return text

    def _generate_embedding(self, text):
        """Generate an embedding for text using the OpenAI API.
        
        This method handles different initialization methods for the OpenAI client.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            list: Embedding vector as a list of floats, always returns a valid vector
        """
        # Default zero vector - use this for any failure cases
        zero_vector = [0.0] * 3072
        
        try:
            if not text or not text.strip():
                # Return a zero vector if text is empty or just whitespace
                self.logger.debug("Empty text received for embedding, returning zero vector")
                return zero_vector
            
            # Sanitize text to avoid encoding issues with special characters
            sanitized_text = self._sanitize_text(text)
            
            # If sanitization resulted in empty text, return zero vector
            if not sanitized_text or not sanitized_text.strip():
                self.logger.debug("Text became empty after sanitization, returning zero vector")
                return zero_vector
            
            # If we're using the direct OpenAI client
            if hasattr(self, 'is_direct_client') and self.is_direct_client:
                try:
                    # Using OpenAI client directly
                    response = self.embedding_func.embeddings.create(
                        input=sanitized_text,
                        model="text-embedding-3-large"
                    )
                    embedding = response.data[0].embedding
                    
                    # Validate the embedding
                    if embedding and len(embedding) == 3072:
                        return embedding
                    else:
                        self.logger.warning("Invalid embedding returned from OpenAI API, using zero vector")
                        return zero_vector
                except Exception as e:
                    self.logger.error(f"Error with direct OpenAI client: {str(e)}")
                    # Return a zero vector as fallback
                    return zero_vector
            else:
                # Using LanceDB embedding registry
                try:
                    embedding = self.embedding_func(sanitized_text)
                    
                    # Validate the embedding
                    if embedding and len(embedding) == 3072:
                        return embedding
                    else:
                        self.logger.warning("Invalid embedding returned from registry, using zero vector")
                        return zero_vector
                except Exception as e:
                    self.logger.error(f"Error with embedding registry: {str(e)}")
                    return zero_vector
                
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            
            # Always return a zero vector as fallback
            return zero_vector

    def _detect_content_type(self, text):
        """Detect the type of content in the text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Content type classification
        """
        try:
            # Skip empty text
            if not text:
                return "empty"
                
            # Count markers for code blocks
            code_markers = text.count('```')
            json_markers = text.count('{') + text.count('}') + text.count('[') + text.count(']')
            
            # Count markers for tables
            table_markers = text.count('|')
            
            # Check for list markers
            list_markers = sum(1 for line in text.split('\n') if line.strip().startswith(('*', '-', '+', '1.', '2.')))
            
            # Check for math markers
            math_markers = text.count('$') + text.count('\\begin{') + text.count('\\end{')
            
            # Classify based on the most prevalent markers
            if code_markers >= 2 or (json_markers > 10 and len(text) < 1000):
                return "code"
            elif table_markers > 10:
                return "table"
            elif list_markers > 5:
                return "list"
            elif math_markers > 5:
                return "math"
            else:
                return "text"
        except Exception:
            # Default to "text" if classification fails
            return "text"
            
    def _extract_page_numbers(self, chunk):
        """Extract page numbers from chunk metadata using multiple strategies.
        
        Args:
            chunk: Document chunk object
            
        Returns:
            tuple: (primary_page_number, total_pages) or (None, None) if not found
        """
        try:
            # Strategy 1: Try to get page numbers from doc_items
            if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items') and chunk.meta.doc_items:
                page_numbers = []
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'prov'):
                        # Handle both single prov and list of provs
                        provs = item.prov if isinstance(item.prov, list) else [item.prov]
                        for prov in provs:
                            if hasattr(prov, 'page_no'):
                                page_numbers.append(prov.page_no)
                
                # Remove duplicates and sort
                if page_numbers:
                    page_numbers = sorted(set(page_numbers))
                    return page_numbers[0], len(page_numbers)
            
            # Strategy 2: Try to extract from text pattern (Page X of Y)
            import re
            page_pattern = re.compile(r"Page\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE)
            match = page_pattern.search(chunk.text)
            if match:
                return int(match.group(1)), int(match.group(2))
            
            # Strategy 3: Check for inline page markers
            inline_pattern = re.compile(r"\[page\s*(\d+)\]", re.IGNORECASE)
            match = inline_pattern.search(chunk.text)
            if match:
                return int(match.group(1)), None
            
            # No page numbers found
            return None, None
            
        except Exception as e:
            self.logger.debug(f"Error extracting page numbers: {str(e)}")
            return None, None

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
