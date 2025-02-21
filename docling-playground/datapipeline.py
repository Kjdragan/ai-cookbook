import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls
from docling.datamodel.pipeline_options import (
    AcceleratorOptions, 
    AcceleratorDevice, 
    PdfPipelineOptions,
    EasyOcrOptions
)
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from docling.utils.model_downloader import download_models
from typing import List, Dict, Any
import lancedb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from docling.chunking import HybridChunker
from utils.tokenizer import OpenAITokenizerWrapper
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from datetime import datetime

class ChunkMetadata(LanceModel):
    """Metadata for document chunks.
    Fields must be in alphabetical order (Pydantic requirement).
    """
    doc_type: str
    filename: str | None
    page_numbers: List[int] | None
    processed_date: datetime
    source_path: str
    title: str | None

class Chunks(LanceModel):
    """Schema for document chunks in LanceDB."""
    metadata: ChunkMetadata
    text: str
    vector: Vector(1536)  # For text-embedding-3-large

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
        
        # Initialize LanceDB
        self._setup_database()
        
        # Initialize chunker
        self.tokenizer = OpenAITokenizerWrapper()
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=8191,  # text-embedding-3-large's maximum context length
            merge_peers=True
        )
        
        init_time = time.time() - self.init_start_time
        self.logger.info(f"Pipeline initialization time: {init_time:.2f} seconds")
        self.logger.info("Pipeline initialization complete. Ready for processing.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _warmup_models(self, artifacts_path):
        """Download and prefetch required models with retry logic."""
        try:
            # Ensure the artifacts directory exists
            os.makedirs(artifacts_path, exist_ok=True)
            
            # Download models if they're not already present
            download_models()
            self.logger.info(f"Models downloaded successfully to: {artifacts_path}")
        except Exception as e:
            self.logger.error(f"Error during model warmup: {str(e)}")
            raise

    def verify_lancedb_chunks(self, expected_count: Optional[int] = None) -> bool:
        """Verify that chunks were properly stored in LanceDB.
        
        Args:
            expected_count: Optional number of chunks expected
            
        Returns:
            bool: True if verification passed
        """
        try:
            # Check if table exists and has data
            count = len(self.table)
            self.logger.info(f"Found {count} chunks in LanceDB table")
            
            if expected_count is not None and count != expected_count:
                self.logger.warning(f"Expected {expected_count} chunks but found {count}")
                return False
            
            # Verify a random chunk has all required fields
            if count > 0:
                sample = self.table.head(1)
                chunk = sample[0]
                required_fields = ["text", "metadata", "vector"]
                
                for field in required_fields:
                    if field not in chunk:
                        self.logger.error(f"Missing required field: {field}")
                        return False
                        
                self.logger.info("Chunk verification passed")
                return True
            
            return count > 0
            
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
                {
                    "text": chunk.text,
                    "metadata": {
                        "filename": chunk.meta.origin.filename,
                        "page_numbers": sorted(set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )) or None,
                        "title": chunk.meta.headings[0] if chunk.meta.headings else None,
                        "doc_type": file_path.suffix[1:],  # Remove leading dot
                        "processed_date": datetime.now(),
                        "source_path": str(file_path)
                    }
                }
                for chunk in chunks
            ]
            
            # Add chunks to LanceDB (automatic embedding)
            self.logger.info("Storing chunks in LanceDB...")
            self.table.add(processed_chunks)
            
            # Verify storage
            if self.verify_lancedb_chunks(len(processed_chunks)):
                self.logger.info(f"Successfully processed and stored: {file_path.name}")
            else:
                raise Exception("Chunk verification failed")
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path.name}: {str(e)}")
            raise

    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            self.logger.info(f"Starting {func.__name__}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            self.logger.info(f"Processing time for {func.__name__}: {end_time - start_time:.2f} seconds")
            return result
        return wrapper

    @timing_decorator
    def process_pdf(self, url):
        result = self.converter.convert(url)
        document = result.document
        markdown_output = document.export_to_markdown()
        json_output = document.export_to_dict()

        # Save outputs to _processed_output directory
        base_name = os.path.basename(url).replace('.pdf', '')
        markdown_file = os.path.join('_processed_output', f'{base_name}.md')
        json_file = os.path.join('_processed_output', f'{base_name}.json')

        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_output)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)

        return markdown_file, json_file

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
        self.db = lancedb.connect("data/lancedb")
        self.embedding_func = get_registry().get("openai").create(
            name="text-embedding-3-large"
        )
        
        # Create table if it doesn't exist
        try:
            self.table = self.db.open_table("documents")
            self.logger.info("Connected to existing documents table")
        except Exception:
            self.table = self.db.create_table(
                "documents",
                schema=Chunks,
                mode="overwrite"
            )
            self.logger.info("Created new documents table")

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
    markdown_file, json_file = pipeline.process_pdf("https://arxiv.org/pdf/2408.09869")
    total_time = time.time() - start_time
    pipeline.logger.info(f"Saved markdown to {markdown_file} and JSON to {json_file}")
    pipeline.logger.info(f"Total execution time: {total_time:.2f} seconds")
