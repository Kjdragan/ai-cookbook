# Build Amendment Plan

## Issues Identified

1. **Pipeline Only Processes One Document**: 
   - The data pipeline processes files from the input directory but doesn't track which files have already been processed
   - The file monitoring setup exists but is not being utilized properly at runtime

2. **Database Storage Issues**:
   - The error message indicates documents aren't being properly stored or processed into the database
   - When verifying chunks, there's an error: "'LanceTable' object has no attribute 'to_list'"

3. **LanceDB API Inconsistency**:
   - The code uses `.to_list()` on LanceTable objects, but LanceDB API actually uses `.to_pandas()` or `.to_arrow()` or `.to_pylist()`

## Root Cause Analysis

1. **Document Processing Tracking**:
   - The pipeline doesn't maintain a record of previously processed files
   - The file system monitoring is set up (lines 467-476) but not started in the main execution
   - The main execution just processes all PDFs in the directory regardless of whether they've been processed before

2. **LanceDB Method Error**:
   - In the `verify_lancedb_chunks` method (line 220), the code uses `.to_list()` which doesn't exist in the LanceDB API
   - Based on LanceDB documentation, the correct method should be `.to_pandas()` or `.to_arrow()` or `.to_pylist()`

3. **Database Storage**:
   - When adding data to LanceDB, the current implementation uses `mode="overwrite"` (line 321) which replaces all data each time
   - This means only the most recently processed document's chunks remain in the database

## Amendment Plan

### 1. Implement File Processing Tracking

1. Create a SQLite database to track processed files:
```python
# Add to imports
import sqlite3
from datetime import datetime

# Add to DataPipeline.__init__
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
```

2. Add methods to check and update file status:
```python
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
```

### 2. Fix LanceDB API Usage

1. Update the `verify_lancedb_chunks` method to use the correct LanceDB API:
```python
def verify_lancedb_chunks(self, expected_count: Optional[int] = None) -> bool:
    """Verify that chunks were properly stored in LanceDB."""
    try:
        # Get all chunks from the table - use to_pandas() instead of to_list()
        table = self.db.open_table("chunks")
        all_chunks = table.to_pandas() 
        chunk_count = len(all_chunks)
        self.logger.info(f"Found {chunk_count} chunks in LanceDB table")
        
        # Print diagnostic information for the first chunk
        if chunk_count > 0:
            self.logger.info("Sample chunk data:")
            sample_chunk = all_chunks.iloc[0]
            self.logger.info(f"Available columns: {list(all_chunks.columns)}")
            self.logger.info(f"Sample chunk metadata: {sample_chunk.get('metadata', 'No metadata')}")
            self.logger.info(f"Sample chunk text: {sample_chunk.get('text', 'No text')[:200] + '...' if sample_chunk.get('text') else 'No text'}")
            self.logger.info(f"Sample chunk vector shape: {len(sample_chunk.get('vector', [])) if sample_chunk.get('vector') is not None else 'No vector'}")
        
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
```

### 3. Fix Database Storage Issue

1. Update the `process_chunks` method to use `mode="append"` instead of `mode="overwrite"`:
```python
def process_chunks(self, chunks: List[DocumentChunk]) -> None:
    """Process and store document chunks in the vector database."""
    try:
        # Convert chunks to LanceDB format
        lance_data = [chunk.to_lance_dict() for chunk in chunks]
        
        # Check if table exists
        if "chunks" in self.db.table_names():
            # If table exists, use append mode
            table = self.db.open_table("chunks")
            table.add(data=lance_data)
        else:
            # If table doesn't exist, create it
            self.db.create_table(
                name="chunks",
                data=lance_data,
                schema=DocumentChunk.get_lance_schema(),
                mode="create"
            )
        
        # Verify chunk count
        table = self.db.open_table("chunks")
        stored_count = len(table)
        new_count = len(chunks)
        
        self.logger.info(f"Successfully stored {new_count} new chunks in the database (total: {stored_count})")
        
    except Exception as e:
        self.logger.error(f"Error processing chunks: {str(e)}")
        raise
```

### 4. Update Main Execution Logic

1. Modify the main execution to only process unprocessed files:
```python
if __name__ == "__main__":
    start_time = time.time()
    pipeline = DataPipeline()
    
    # Process all new documents in the input directory
    input_dir = Path(r"C:\Users\kevin\repos\docling-playground\_documents_for_processing_input")
    new_files_processed = 0
    
    for pdf_file in input_dir.glob("*.pdf"):
        if not pipeline.is_file_processed(pdf_file):
            print(f"\nProcessing new file: {pdf_file.name}")
            try:
                pipeline.process_document(pdf_file)
                new_files_processed += 1
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")
                pipeline.mark_file_processed(pdf_file, 0, status="error")
        else:
            print(f"Skipping already processed file: {pdf_file.name}")
    
    # Verify chunks were stored
    pipeline.verify_lancedb_chunks()
    
    end_time = time.time()
    print(f"Processed {new_files_processed} new files")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
```

### 5. Enable File Monitoring

1. Add file monitoring activation to main execution to enable real-time processing:
```python
if __name__ == "__main__":
    start_time = time.time()
    pipeline = DataPipeline()
    
    # Enable file monitoring
    pipeline._setup_document_monitoring()
    
    # Process existing unprocessed documents
    input_dir = Path(r"C:\Users\kevin\repos\docling-playground\_documents_for_processing_input")
    new_files_processed = 0
    
    # Process existing files first
    for pdf_file in input_dir.glob("*.pdf"):
        if not pipeline.is_file_processed(pdf_file):
            print(f"\nProcessing new file: {pdf_file.name}")
            try:
                pipeline.process_document(pdf_file)
                new_files_processed += 1
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")
                pipeline.mark_file_processed(pdf_file, 0, status="error")
        else:
            print(f"Skipping already processed file: {pdf_file.name}")
    
    # Verify chunks were stored
    pipeline.verify_lancedb_chunks()
    
    end_time = time.time()
    print(f"Processed {new_files_processed} new files")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    print("File monitoring is active. Press Ctrl+C to exit.")
    try:
        # Keep the program running to monitor for new files
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
```

## Implementation Steps

1. **Fix LanceDB API Usage**:
   - Update the `verify_lancedb_chunks` method to use `.to_pandas()` instead of `.to_list()`

2. **Fix Database Storage**:
   - Modify the `process_chunks` method to use append mode instead of overwrite mode

3. **Implement File Tracking**:
   - Add SQLite database for tracking processed files
   - Add methods to check and update file processing status

4. **Update Main Execution Logic**:
   - Modify to only process new/unprocessed files
   - Enable file monitoring for real-time processing

## Testing Plan

1. **Verify LanceDB API Fix**:
   - Run pipeline with a single document
   - Check if verification completes without errors

2. **Verify Document Storage**:
   - Process multiple documents sequentially
   - Verify all document chunks are in the database

3. **Verify File Tracking**:
   - Process some documents
   - Restart the pipeline
   - Verify only new documents are processed

4. **Verify File Monitoring**:
   - Start the pipeline
   - Add a new document to the input directory
   - Verify it's automatically processed
