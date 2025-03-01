"""
Process a specific target document using the DataPipeline.
"""
import os
import sys
import time
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Add the docling-playground directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the DataPipeline
from docling_playground.datapipeline import DataPipeline

def process_specific_document(document_path):
    """Process a specific document using the DataPipeline."""
    start_time = time.time()
    logger.info(f"Starting to process document: {document_path}")
    
    try:
        # Create the pipeline
        pipeline = DataPipeline()
        
        # Process the document
        document_path = Path(document_path)
        if not document_path.exists():
            logger.error(f"Document does not exist: {document_path}")
            return False
        
        # Process the document
        pipeline.process_document(document_path)
        
        # Verify chunks were stored
        pipeline.verify_lancedb_chunks()
        
        end_time = time.time()
        logger.info(f"Document processed successfully in {end_time - start_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False

if __name__ == "__main__":
    # Path to the target document
    target_document = "_documents_for_processing_input/S1-Simple-test-time scaling.pdf"
    
    # Process the document
    success = process_specific_document(target_document)
    
    if success:
        print("\nDocument processed successfully!")
        print("Now we can use it for semantic search demonstrations.")
    else:
        print("\nFailed to process the document.")
