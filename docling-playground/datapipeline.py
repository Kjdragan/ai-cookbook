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
from utils.tokenizer import OpenAITokenizerWrapper
from utils.sitemap import get_sitemap_urls
import lancedb
from lancedb import vector
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import numpy as np
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer
from docling.datamodel.base_models import InputFormat
from docling.utils.model_downloader import download_models
import shutil
from dotenv import load_dotenv

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
        """Initialize the document processing pipeline."""
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,  # Change to DEBUG level
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables from .env
        load_dotenv()
        
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
        
        # Initialize database
        self.db = lancedb.connect("lancedb")
        
        # Initialize embedding function
        self.logger.info("Initializing embedding function...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.embedding_func = get_registry().get("openai").create(
            name="text-embedding-3-large",
            api_key=api_key
        )
        
        # Initialize tokenizer and chunker
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
            all_chunks = self.db.get_table("chunks").to_list()
            chunk_count = len(all_chunks)
            self.logger.info(f"Found {chunk_count} chunks in LanceDB table")
            
            # Print diagnostic information for the first chunk
            if chunk_count > 0:
                self.logger.info("Sample chunk data:")
                sample_chunk = all_chunks[0]
                self.logger.info("Available columns: %s", list(all_chunks[0].keys()))
                self.logger.info("Sample chunk metadata: %s", sample_chunk.get('metadata', 'No metadata'))
                self.logger.info("Sample chunk text: %s", sample_chunk.get('text', 'No text')[:200] + '...' if sample_chunk.get('text') else 'No text')
                self.logger.info("Sample chunk vector shape: %s", len(sample_chunk.get('vector', [])) if sample_chunk.get('vector') is not None else 'No vector')
            
            if expected_count is not None and chunk_count != expected_count:
                self.logger.warning(f"Expected {expected_count} chunks but found {chunk_count}")
                return False
            
            # Verify required fields are present
            required_fields = ['text', 'vector', 'metadata']
            for field in required_fields:
                if field not in all_chunks[0].keys():
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
            
            # Store chunks in LanceDB
            self.logger.info("Storing chunks in LanceDB...")
            self.logger.info(f"Processing {len(chunks)} chunks")
            
            # Prepare chunks for LanceDB
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
                try:
                    metadata = {
                        "filename": chunk.meta.origin.filename or file_path.name,
                        "page_numbers": [item.prov.page_no for item in chunk.meta.doc_items] or [0],
                        "title": chunk.meta.headings[0] if chunk.meta.headings else "",
                        "doc_type": file_path.suffix[1:] or "unknown",
                        "processed_date": datetime.now().isoformat(),
                        "source_path": str(file_path),
                        "chunk_id": str(i)
                    }
                    processed_chunk = DocumentChunk(
                        text=chunk.text or "empty",
                        metadata=metadata,
                        vector=self.embedding_func.generate_embeddings([chunk.text or "empty"])[0]
                    )
                    processed_chunks.append(processed_chunk)
                    self.logger.debug(f"Successfully processed chunk {i+1}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    raise

            self.process_chunks(processed_chunks)
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    def process_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Process and store document chunks in the vector database."""
        try:
            # Convert chunks to LanceDB format
            lance_data = [chunk.to_lance_dict() for chunk in chunks]
            
            # Create or overwrite the chunks table
            self.db.create_table(
                name="chunks",
                data=lance_data,
                schema=DocumentChunk.get_lance_schema(),
                mode="overwrite"
            )
            
            # Verify chunk count
            table = self.db.open_table("chunks")
            stored_count = len(table)
            expected_count = len(chunks)
            
            if stored_count != expected_count:
                self.logger.error(f"Chunk count mismatch! Expected {expected_count}, got {stored_count}")
                raise ValueError("Chunk count mismatch")
                
            self.logger.info(f"Successfully stored {stored_count} chunks in the database")
            
        except Exception as e:
            self.logger.error(f"Error processing chunks: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
                    "chunk_id": str(uuid.uuid4())
                }
                
                # Create processed chunk with embeddings
                processed_chunk = DocumentChunk(
                    text=chunk.text or "empty",
                    metadata=metadata,
                    vector=self.embedding_func.generate_embeddings([chunk.text or "empty"])[0]
                )
                batch_chunks.append(processed_chunk)
            
            processed_chunks.extend(batch_chunks)
        
        self.process_chunks(processed_chunks)
        
        process_time = time.time() - start_time
        self.logger.info(f"Processing time for process_pdf: {process_time:.2f} seconds")
        self.logger.info(f"Saved markdown to _processed_output\\{os.path.basename(url).replace('.pdf', '')}.md and JSON to _processed_output\\{os.path.basename(url).replace('.pdf', '')}.json")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
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
