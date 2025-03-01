"""
Process all documents in the _documents_for_processing_input directory
"""
import os
import sys
import time
import glob
from pathlib import Path

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add the docling-playground directory to sys.path
sys.path.append(os.path.join(current_dir, "docling-playground"))
from datapipeline import DataPipeline

def main():
    """Process all documents in the input directory using the DataPipeline."""
    # Path to input directory
    input_dir = Path("_documents_for_processing_input")
    
    if not input_dir.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    print(f"Processing documents from directory: {input_dir}")
    start_time = time.time()
    
    # Initialize the pipeline
    pipeline = DataPipeline()
    
    # Track processing statistics
    files_processed = 0
    files_with_errors = 0
    
    # Process all PDF files in the input directory
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"\nProcessing file: {pdf_file.name}")
        try:
            # Process the document
            pipeline.process_document(pdf_file)
            files_processed += 1
            print(f"Successfully processed {pdf_file.name}")
        except Exception as e:
            print(f"Error processing document {pdf_file.name}: {str(e)}")
            files_with_errors += 1
            # Mark the file as having an error
            pipeline.mark_file_processed(str(pdf_file), 0, status="error")
    
    # Verify the database contents
    print("\nVerifying database contents...")
    pipeline.verify_lancedb_chunks()
    
    # Get the path to the database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb_data")
    
    end_time = time.time()
    print(f"\nProcessing summary:")
    print(f"- Total files processed: {files_processed}")
    print(f"- Files with errors: {files_with_errors}")
    print(f"- Processing completed in {end_time - start_time:.2f} seconds")
    print(f"- Database location: {db_path}")

if __name__ == "__main__":
    main()
