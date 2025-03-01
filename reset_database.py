"""
Reset the database and regenerate it from scratch.
This script will delete the existing database and rebuild it with a consistent schema.
"""
import os
import sys
import time
import logging
import shutil
from datetime import datetime
from pathlib import Path

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
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
    log_filename = f"logs/reset_db_{timestamp}.log"
    
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
    console_log_filename = f"logs/reset_db_console_{timestamp}.log"
    sys.stdout = OutputCapture(console_log_filename)
    sys.stderr = OutputCapture(console_log_filename)
    
    return log_filename, console_log_filename

def reset_database():
    """Reset the database completely by removing all files and directories."""
    from datapipeline import DataPipeline
    
    # Get database paths
    pipeline = DataPipeline()
    lancedb_path = pipeline.lancedb_url
    
    # SQLite database path - hardcoded since it's not available as an attribute
    sqlite_path = "processed_files.db"
    
    # Report what we're going to delete
    print(f"Will delete LanceDB directory: {os.path.abspath(lancedb_path)}")
    print(f"Will delete SQLite database: {os.path.abspath(sqlite_path)}")
    
    # Delete LanceDB files
    if os.path.exists(lancedb_path):
        try:
            shutil.rmtree(lancedb_path)
            print(f"Successfully deleted LanceDB directory: {lancedb_path}")
        except Exception as e:
            print(f"Error deleting LanceDB directory: {str(e)}")
    else:
        print(f"LanceDB directory doesn't exist: {lancedb_path}")
    
    # Delete SQLite database
    if os.path.exists(sqlite_path):
        try:
            os.remove(sqlite_path)
            print(f"Successfully deleted SQLite database: {sqlite_path}")
        except Exception as e:
            print(f"Error deleting SQLite database: {str(e)}")
    else:
        print(f"SQLite database doesn't exist: {sqlite_path}")
    
    # Verify deletion
    lancedb_exists = os.path.exists(lancedb_path)
    sqlite_exists = os.path.exists(sqlite_path)
    print(f"After deletion: LanceDB exists: {lancedb_exists}, SQLite exists: {sqlite_exists}")
    
    # Initialize a fresh database
    print("Initializing fresh database...")
    pipeline.reset_database()
    print("Database reset complete.")

def main():
    # Set up logging
    log_filename, console_log_filename = setup_logging()
    
    try:
        print("Starting database reset...")
        reset_database()
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
