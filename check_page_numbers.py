"""
Script to analyze page numbers in the database, with comprehensive logging.
"""
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import traceback

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
    log_filename = f"logs/page_analysis_{timestamp}.log"
    
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
    console_log_filename = f"logs/page_analysis_console_{timestamp}.log"
    sys.stdout = OutputCapture(console_log_filename)
    sys.stderr = OutputCapture(console_log_filename)
    
    return log_filename, console_log_filename

def check_page_numbers():
    """Analyze page numbers in the LanceDB database"""
    from datapipeline import DataPipeline
    import lancedb
    import numpy as np
    
    print("Connecting to LanceDB database...")
    try:
        pipeline = DataPipeline()
        db_path = os.path.abspath("lancedb_data")
        print(f"Database location: {db_path}")
        
        # Connect to LanceDB
        db = lancedb.connect(db_path)
        
        # Check if chunks table exists
        if "chunks" not in db.table_names():
            print("Error: 'chunks' table not found in database.")
            return
        
        # Load chunks table
        chunks_table = db.open_table("chunks")
        chunks_df = chunks_table.to_pandas()
        
        # Print basic statistics
        total_chunks = len(chunks_df)
        print(f"\n=== DATABASE STATISTICS ===")
        print(f"Total chunks: {total_chunks}")
        
        # Document statistics
        docs = chunks_df['filename'].unique()
        print(f"Total documents: {len(docs)}")
        print(f"Documents: {', '.join(docs)}")
        
        # Print detailed page statistics
        print(f"\n=== PAGE NUMBER STATISTICS ===")
        
        # Count chunks with page numbers
        has_page_numbers = chunks_df['page_numbers'].apply(lambda x: x is not None and len(x) > 0)
        chunks_with_pages = has_page_numbers.sum()
        print(f"Chunks with page numbers: {chunks_with_pages} ({chunks_with_pages/total_chunks*100:.2f}%)")
        
        # Count chunks with primary page
        has_primary_page = chunks_df['primary_page'].apply(lambda x: x is not None and x >= 0)
        chunks_with_primary = has_primary_page.sum()
        print(f"Chunks with primary page: {chunks_with_primary} ({chunks_with_primary/total_chunks*100:.2f}%)")
        
        # Distribution of page numbers
        print("\n=== PAGE NUMBER DISTRIBUTION ===")
        page_counts = chunks_df['page_numbers'].apply(lambda x: len(x) if x is not None else 0)
        count_dist = page_counts.value_counts().sort_index()
        for count, freq in count_dist.items():
            print(f"Chunks with {count} page(s): {freq} ({freq/total_chunks*100:.2f}%)")
            
        # Analyze primary page vs page_numbers relationship
        print("\n=== PRIMARY PAGE ANALYSIS ===")
        for _, chunk in chunks_df.iterrows():
            if chunk['primary_page'] is not None and chunk['page_numbers'] is not None:
                if chunk['primary_page'] >= 0 and len(chunk['page_numbers']) > 0:
                    if chunk['primary_page'] not in chunk['page_numbers']:
                        print(f"WARNING: Chunk {chunk['chunk_index']} from {chunk['filename']} has primary_page {chunk['primary_page']} not in page_numbers {chunk['page_numbers']}")
        
        # Print detailed chunk metadata for sample chunks
        print("\n=== SAMPLE CHUNK DETAILS ===")
        sample_size = min(10, len(chunks_df))
        sample_chunks = chunks_df.sample(sample_size) if sample_size > 0 else chunks_df
        
        for _, chunk in sample_chunks.iterrows():
            print(f"\nChunk ID: {chunk['chunk_index']}")
            print(f"Filename: {chunk['filename']}")
            print(f"Page Numbers: {chunk['page_numbers']}")
            print(f"Primary Page: {chunk['primary_page']}")
            print(f"Word Count: {chunk['word_count']}")
            print(f"Character Count: {chunk['character_count']}")
            print(f"Content Type: {chunk.get('content_type', 'N/A')}")
            
            # Print first 100 characters of text
            text_preview = chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']
            print(f"Text Preview: {text_preview}")
            
        # Document level page analysis
        print("\n=== DOCUMENT PAGE ANALYSIS ===")
        for doc in docs:
            doc_chunks = chunks_df[chunks_df['filename'] == doc]
            doc_pages = set()
            for pages in doc_chunks['page_numbers']:
                if pages is not None:
                    doc_pages.update(pages)
            
            print(f"\nDocument: {doc}")
            print(f"Total chunks: {len(doc_chunks)}")
            print(f"Unique pages: {sorted(doc_pages)}")
            print(f"Page count: {len(doc_pages)}")
            
            # Check for page gaps
            if len(doc_pages) > 0:
                max_page = max(doc_pages)
                expected_pages = set(range(min(doc_pages), max_page + 1))
                missing_pages = expected_pages - doc_pages
                if missing_pages:
                    print(f"Missing pages: {sorted(missing_pages)}")
                else:
                    print("No missing pages detected")
                    
        print("\n=== METADATA STATISTICS ===")
        for col in ['word_count', 'character_count', 'primary_page']:
            if col in chunks_df.columns:
                values = chunks_df[col].dropna()
                if len(values) > 0:
                    print(f"{col} - Min: {values.min()}, Max: {values.max()}, Mean: {values.mean():.2f}, Median: {values.median()}")
        
        print("\n=== CONTENT TYPE ANALYSIS ===")
        if 'content_type' in chunks_df.columns:
            content_type_counts = chunks_df['content_type'].value_counts()
            for content_type, count in content_type_counts.items():
                print(f"{content_type}: {count} chunks ({count/total_chunks*100:.2f}%)")
                
        print("\nAnalysis completed successfully.")
        
    except Exception as e:
        print(f"Error analyzing database: {str(e)}")
        traceback.print_exc()

def main():
    # Set up logging
    log_filename, console_log_filename = setup_logging()
    
    try:
        print("Starting page number analysis...")
        check_page_numbers()
        print(f"\nLogs saved to: {os.path.abspath(log_filename)}")
        print(f"Console output saved to: {os.path.abspath(console_log_filename)}")
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        traceback.print_exc()
        print(f"\nLogs saved to: {os.path.abspath(log_filename)}")
        print(f"Console output saved to: {os.path.abspath(console_log_filename)}")

if __name__ == "__main__":
    main()
