import lancedb
import pandas as pd
import sys

def check_lancedb():
    try:
        # Connect to LanceDB
        db = lancedb.connect('lancedb_data')
        print('Available tables:', db.table_names())
        
        # Check if chunks table exists
        if 'chunks' in db.table_names():
            chunks = db.open_table('chunks')
            
            # Get sample data and count
            try:
                # Get data and count rows directly
                df = chunks.to_pandas()
                row_count = len(df)
                print(f'Number of chunks: {row_count}')
                
                if row_count > 0:
                    print('Schema columns:', list(df.columns))
                    print('Sample metadata fields:', [col for col in df.columns if col != 'vector'])
                    
                    # Print sample of available documents
                    print("\n=== AVAILABLE DOCUMENTS ===")
                    if 'metadata' in df.columns:
                        # Try to extract document info from metadata
                        try:
                            # Get unique document sources
                            if isinstance(df['metadata'].iloc[0], dict):
                                # Check for various possible fields
                                doc_fields = ['source', 'file_path', 'file_name', 'document_name', 'title']
                                
                                for field in doc_fields:
                                    if field in df['metadata'].iloc[0]:
                                        unique_docs = set()
                                        for meta in df['metadata']:
                                            if field in meta and meta[field]:
                                                unique_docs.add(meta[field])
                                        
                                        print(f"Found {len(unique_docs)} unique documents by metadata.{field}:")
                                        for doc in sorted(unique_docs):
                                            print(f"  - {doc}")
                                        break
                        except Exception as e:
                            print(f"Error extracting document names from metadata: {e}")
                    
                    # Check if our specific document exists
                    document_name = 'S1-Simple-test-time scaling.pdf'
                    search_fields = ['file_path', 'source', 'source_path', 'metadata.source', 'metadata.file_path']
                    
                    found = False
                    for field in search_fields:
                        if '.' in field:
                            # Handle nested fields
                            parts = field.split('.')
                            if parts[0] in df.columns:
                                try:
                                    if isinstance(df[parts[0]].iloc[0], dict):
                                        nested_field = parts[1]
                                        matching_docs = df[df[parts[0]].apply(
                                            lambda x: nested_field in x and document_name in str(x.get(nested_field, ''))
                                        )]
                                        
                                        if len(matching_docs) > 0:
                                            found = True
                                            print(f"\nFound {len(matching_docs)} chunks for document '{document_name}' in field '{field}'")
                                            
                                            # Display sample chunk
                                            display_sample_chunk(matching_docs.iloc[0])
                                            break
                                except Exception as e:
                                    print(f"Error checking nested field {field}: {e}")
                        elif field in df.columns:
                            try:
                                matching_docs = df[df[field].astype(str).str.contains(document_name, case=False, na=False)]
                                
                                if len(matching_docs) > 0:
                                    found = True
                                    print(f"\nFound {len(matching_docs)} chunks for document '{document_name}' in field '{field}'")
                                    
                                    # Display sample chunk
                                    display_sample_chunk(matching_docs.iloc[0])
                                    break
                            except Exception as e:
                                print(f"Error checking field {field}: {e}")
                    
                    if not found:
                        print(f"\nDocument '{document_name}' not found in database")
            except Exception as e:
                print(f"Error accessing table data: {e}")
        else:
            print("No 'chunks' table found in database")
    except Exception as e:
        print(f"Error connecting to LanceDB: {e}")

def display_sample_chunk(sample_chunk):
    """Display a sample chunk with formatted output"""
    print("\nSample chunk from the document:")
    
    # Display text content
    if 'text' in sample_chunk:
        text_excerpt = sample_chunk['text'][:200] + "..." if len(sample_chunk['text']) > 200 else sample_chunk['text']
        print(f"Text (excerpt): {text_excerpt}")
    
    # Display metadata
    for col in sample_chunk.index:
        if col != 'text' and col != 'vector':
            print(f"{col}: {sample_chunk[col]}")

if __name__ == "__main__":
    check_lancedb()
