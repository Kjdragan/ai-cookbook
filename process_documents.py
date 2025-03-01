import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add the docling-playground directory to sys.path
sys.path.append(os.path.join(current_dir, "docling-playground"))

# Create a class to capture stdout and stderr
class OutputCapture:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure content is written immediately
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def setup_logging():
    """Set up logging with file output"""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/pipeline_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up console output capture
    console_log_filename = f"logs/console_output_{timestamp}.log"
    sys.stdout = OutputCapture(console_log_filename)
    sys.stderr = OutputCapture(console_log_filename)
    
    return log_filename, console_log_filename

def main():
    """Process all documents in the input directory using the DataPipeline."""
    # Set up logging first
    log_filename, console_log_filename = setup_logging()
    
    try:
        # Import after logging setup to capture import-time logs
        from datapipeline import DataPipeline
        
        # Path to input directory
        input_dir = Path("_documents_for_processing_input")
        
        print(f"Processing documents from directory: {input_dir}")
        
        # Check if input directory exists
        if not input_dir.exists():
            print(f"Error: Input directory '{input_dir}' does not exist.")
            return
        
        # Create pipeline
        print("Creating pipeline...")
        pipeline = DataPipeline()
        
        # Reset database for fresh test
        print("Resetting database for fresh test...")
        pipeline.reset_database()
        
        start_time = time.time()
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
                import traceback
                traceback.print_exc()
                files_with_errors += 1
                # Mark the file as having an error
                pipeline.mark_file_processed(str(pdf_file), 0, status="error")
        
        # Verify the database contents
        print("\nVerifying database contents...")
        pipeline.verify_database()
        
        # Get the path to the database
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb_data")
        
        end_time = time.time()
        print(f"\nProcessing summary:")
        print(f"- Total files processed: {files_processed}")
        print(f"- Files with errors: {files_with_errors}")
        print(f"- Processing completed in {end_time - start_time:.2f} seconds")
        print(f"- Database location: {db_path}")
        print(f"\nLogs saved to: {os.path.abspath(log_filename)}")
        print(f"Console output saved to: {os.path.abspath(console_log_filename)}")
    
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nLogs saved to: {os.path.abspath(log_filename)}")
        print(f"Console output saved to: {os.path.abspath(console_log_filename)}")

if __name__ == "__main__":
    main()
