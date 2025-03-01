import lancedb
import pandas as pd
import os
from pathlib import Path

def check_lancedb(db_path):
    """Check a LanceDB database for document contents."""
    print(f"\n{'='*50}")
    print(f"Checking database: {db_path}")
    print(f"{'='*50}")
    
    try:
        # Connect to LanceDB
        db = lancedb.connect(db_path)
        print('Tables:', db.table_names())
        
        for table_name in db.table_names():
            print(f"\nExamining table: {table_name}")
            table = db.open_table(table_name)
            
            try:
                # Try getting data with to_pandas
                df = table.to_pandas()
                print(f"Row count: {len(df)}")
                print(f"Columns: {list(df.columns)}")
                
                # Look for document info
                if len(df) > 0:
                    # Check for S1-Simple-test-time scaling.pdf
                    document_name = 'S1-Simple-test-time scaling.pdf'
                    print(f"\nSearching for: {document_name}")
                    
                    # Try different fields that might contain document references
                    found = False
                    
                    # Check metadata field if it exists
                    if 'metadata' in df.columns:
                        print("Checking metadata field...")
                        found_count = 0
                        
                        for idx, row in df.iterrows():
                            metadata = row['metadata']
                            if isinstance(metadata, dict):
                                # Check various fields that might contain the document name
                                for field in ['source_path', 'file_path', 'filename', 'source']:
                                    if field in metadata and document_name in str(metadata[field]):
                                        found_count += 1
                                        if found_count == 1:  # Show details of first match only
                                            print(f"Found document in metadata.{field}!")
                                            print(f"Sample text: {row.get('text', '')[:100]}...")
                                            print(f"Metadata: {metadata}")
                                            found = True
                        
                        if found_count > 0:
                            print(f"Total matches found: {found_count}")
                    
                    # Check direct fields if they exist
                    for field in ['source_path', 'file_path', 'filename', 'source']:
                        if field in df.columns:
                            matching = df[df[field].astype(str).str.contains(document_name, case=False, na=False)]
                            if len(matching) > 0:
                                print(f"Found document in {field} field!")
                                print(f"Total matches: {len(matching)}")
                                print(f"Sample text: {matching.iloc[0].get('text', '')[:100]}...")
                                found = True
                    
                    # Search in text field as last resort
                    if 'text' in df.columns:
                        text_matches = df[df['text'].astype(str).str.contains(document_name, case=False, na=False)]
                        if len(text_matches) > 0:
                            print(f"Found document name mentioned in text field!")
                            print(f"Total matches: {len(text_matches)}")
                            print(f"Sample text: {text_matches.iloc[0].get('text', '')[:100]}...")
                            found = True
                    
                    if not found:
                        print(f"Document '{document_name}' not found in this table")
                    
                    # Show unique document sources if available
                    print("\nUnique document sources:")
                    if 'metadata' in df.columns and isinstance(df['metadata'].iloc[0], dict):
                        for field in ['source', 'file_path', 'filename', 'source_path', 'title']:
                            sources = set()
                            for metadata in df['metadata']:
                                if metadata and field in metadata:
                                    source = metadata.get(field)
                                    if source:
                                        sources.add(str(source))
                            
                            if sources:
                                print(f"metadata.{field}: {sorted(sources)[:5]}...")
                                if len(sources) > 5:
                                    print(f"  ...and {len(sources) - 5} more")
                    
                    # Try direct fields too
                    for field in ['source', 'file_path', 'filename', 'source_path', 'title']:
                        if field in df.columns:
                            sources = set(df[field].astype(str).unique())
                            if sources:
                                print(f"{field}: {sorted(sources)[:5]}...")
                                if len(sources) > 5:
                                    print(f"  ...and {len(sources) - 5} more")
                
            except Exception as e:
                print(f"Error examining table: {str(e)}")
        
    except Exception as e:
        print(f"Error connecting to LanceDB: {str(e)}")

def process_document(file_path):
    """Check if we can process the target document."""
    from docling_playground.datapipeline import DataPipeline
    
    try:
        print(f"\n{'='*50}")
        print(f"Attempting to process document: {file_path}")
        print(f"{'='*50}")
        
        # Create pipeline instance
        pipeline = DataPipeline()
        
        # Process the document
        pipeline.process_document(Path(file_path))
        
        print("Document processed successfully!")
        return True
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return False

def main():
    # Check all LanceDB databases
    db_paths = [
        "lancedb_data",
        "data/lancedb",
        "docling-playground/lancedb"
    ]
    
    for db_path in db_paths:
        check_lancedb(db_path)
    
    # Check if we need to process the target document
    document_path = r"_documents_for_processing_input\S1-Simple-test-time scaling.pdf"
    if os.path.exists(document_path):
        should_process = input("\nDocument exists but not found in databases. Process it now? (y/n): ")
        if should_process.lower() == 'y':
            process_document(document_path)
            
            # Check the main database again
            check_lancedb("lancedb_data")
    else:
        print(f"\nDocument not found: {document_path}")

if __name__ == "__main__":
    main()
