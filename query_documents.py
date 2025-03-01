"""
query_documents.py - Search the LanceDB database with semantic queries

This script demonstrates how to query the document database using semantic search
to retrieve relevant context from processed documents.
"""
import os
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add the docling-playground directory to sys.path
sys.path.append(os.path.join(current_dir, "docling-playground"))

import pandas as pd
import lancedb
from openai import OpenAI
from dotenv import load_dotenv

# Fix for encoding issues with Windows console
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_embedding(text, model="text-embedding-3-large"):
    """Get embeddings for a text using OpenAI API."""
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def semantic_search(query, limit=5, db_path="lancedb_data"):
    """Perform semantic search on the LanceDB database."""
    print(f"\nSearching for: '{query}'")
    
    # Get embedding for query
    start_time = time.time()
    query_embedding = get_embedding(query)
    embedding_time = time.time() - start_time
    print(f"Generated embedding in {embedding_time:.2f} seconds")
    
    # Connect to database
    db = lancedb.connect(db_path)
    
    # Check if chunks table exists
    if "chunks" not in db.table_names():
        print("Error: 'chunks' table not found in the database")
        return pd.DataFrame()
        
    # Perform search
    table = db.open_table("chunks")
    start_time = time.time()
    
    try:
        # Try newer API first
        results = table.search(query_embedding).limit(limit).to_pandas()
    except AttributeError:
        # Fall back to older API
        results = table.search(query_embedding).to_pandas().head(limit)
    
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.2f} seconds")
    
    # Format and display results
    for i, row in results.iterrows():
        print(f"\n--- Result {i+1} (Score: {1.0 - float(row['_distance']):.4f}) ---")
        print(f"Document: {row['metadata']['filename']}")
        print(f"Page(s): {row['metadata']['page_numbers']}")
        if row['metadata'].get('title'):
            print(f"Section: {row['metadata']['title']}")
        print("\nExcerpt:")
        
        # Handle text safely for console output
        text_excerpt = row['text'][:300]
        # Replace problematic characters
        text_excerpt = ''.join(c if ord(c) < 128 else '?' for c in text_excerpt)
        print(f"{text_excerpt}...")
        print("-" * 80)
    
    return results

def hybrid_search(text_query, keyword=None, limit=5, db_path="lancedb_data"):
    """Perform hybrid search combining semantic and keyword search."""
    print(f"\nHybrid search - Text: '{text_query}', Keyword: '{keyword}'")
    
    # Get embedding for query
    query_embedding = get_embedding(text_query)
    
    # Connect to database
    db = lancedb.connect(db_path)
    table = db.open_table("chunks")
    
    # Perform hybrid search
    if keyword:
        try:
            # Try newer API with where clause
            results = table.search(query_embedding).where(f"text LIKE '%{keyword}%'").limit(limit).to_pandas()
        except (AttributeError, TypeError):
            # Fall back: filter after search
            all_results = table.search(query_embedding).to_pandas()
            results = all_results[all_results['text'].str.contains(keyword, case=False)].head(limit)
    else:
        # Just semantic search
        results = table.search(query_embedding).limit(limit).to_pandas()
    
    # Format and display results
    for i, row in results.iterrows():
        print(f"\n--- Result {i+1} (Score: {1.0 - float(row['_distance']):.4f}) ---")
        print(f"Document: {row['metadata']['filename']}")
        if row['metadata'].get('title'):
            print(f"Section: {row['metadata']['title']}")
        print("\nExcerpt:")
        # Handle text safely for console output
        text_excerpt = row['text'][:300]
        # Replace problematic characters
        text_excerpt = ''.join(c if ord(c) < 128 else '?' for c in text_excerpt)
        print(f"{text_excerpt}...")
        print("-" * 80)
    
    return results

def get_expanded_context(result_row, context_window=2, db_path="lancedb_data"):
    """Get expanded context by retrieving adjacent chunks from the same document."""
    print(f"\nExpanding context for chunk from '{result_row['metadata']['filename']}'")
    
    # Extract metadata from the result
    filename = result_row['metadata']['filename']
    current_chunk_id = result_row['metadata']['chunk_id']
    try:
        chunk_id = int(current_chunk_id)
    except ValueError:
        print(f"Cannot expand context: non-numeric chunk ID '{current_chunk_id}'")
        return None
    
    # Connect to database
    db = lancedb.connect(db_path)
    table = db.open_table("chunks")
    
    # Calculate range of chunks to retrieve
    start_chunk = max(0, chunk_id - context_window)
    end_chunk = chunk_id + context_window
    
    # Query for chunks in the same document with adjacent IDs
    try:
        all_chunks = table.to_pandas()
        doc_chunks = all_chunks[all_chunks['metadata'].apply(
            lambda x: x['filename'] == filename and 
                      start_chunk <= int(x['chunk_id']) <= end_chunk
        )]
        
        # Sort by chunk ID
        doc_chunks['chunk_num'] = doc_chunks['metadata'].apply(lambda x: int(x['chunk_id']))
        doc_chunks = doc_chunks.sort_values('chunk_num')
        
        if len(doc_chunks) > 0:
            print(f"Found {len(doc_chunks)} chunks for expanded context")
            
            # Combine chunks into a single context
            expanded_texts = []
            for _, row in doc_chunks.iterrows():
                chunk_header = f"--- Chunk {row['metadata']['chunk_id']} " \
                              f"(Page {''.join(map(str, row['metadata']['page_numbers']))}) ---\n"
                # Handle text safely for console output
                chunk_text = row['text']
                # Replace problematic characters
                chunk_text = ''.join(c if ord(c) < 128 else '?' for c in chunk_text)
                expanded_texts.append(chunk_header + chunk_text)
            
            expanded_text = "\n\n".join(expanded_texts)
            
            print("\nExpanded Context:")
            print(f"{expanded_text[:500]}...")
            print("-" * 80)
            
            return expanded_text
        else:
            print("No adjacent chunks found")
            return None
            
    except Exception as e:
        print(f"Error retrieving expanded context: {str(e)}")
        return None

def show_database_stats(db_path="lancedb_data"):
    """Show statistics about the documents in the database."""
    print("\n=== Database Statistics ===")
    
    # Connect to database
    db = lancedb.connect(db_path)
    
    # Check if chunks table exists
    if "chunks" not in db.table_names():
        print("Error: 'chunks' table not found in the database")
        return
    
    # Load all chunks
    table = db.open_table("chunks")
    try:
        chunks = table.to_pandas()
        
        # Extract and count unique documents
        documents = [row['metadata']['filename'] for _, row in chunks.iterrows()]
        doc_counter = Counter(documents)
        
        print(f"Total Documents: {len(doc_counter)}")
        print(f"Total Chunks: {len(chunks)}")
        print("\nDocument Statistics:")
        
        # Display document stats
        for i, (doc, count) in enumerate(doc_counter.most_common()):
            # Get page range for this document
            doc_chunks = chunks[chunks['metadata'].apply(lambda x: x['filename'] == doc)]
            all_pages = []
            for _, row in doc_chunks.iterrows():
                all_pages.extend(row['metadata']['page_numbers'])
            unique_pages = sorted(set(all_pages))
            
            print(f"{i+1}. {doc}")
            print(f"   - Chunks: {count}")
            print(f"   - Pages: {len(unique_pages)} (range: {min(unique_pages)}-{max(unique_pages)})")
            
        print("\n" + "=" * 50)
    
    except Exception as e:
        print(f"Error retrieving database statistics: {str(e)}")

def main():
    """Main function to demonstrate search capabilities."""
    parser = argparse.ArgumentParser(description="Search the document database")
    parser.add_argument("--query", "-q", help="Semantic search query")
    parser.add_argument("--keyword", "-k", help="Optional keyword for hybrid search")
    parser.add_argument("--limit", "-l", type=int, default=3, help="Number of results to return")
    parser.add_argument("--expand", "-e", action="store_true", help="Expand context for first result")
    parser.add_argument("--stats", "-s", action="store_true", help="Show database statistics")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Show database statistics if requested
    if args.stats:
        show_database_stats()
        return
    
    # If no query is provided, use some examples
    if not args.query:
        print("No query provided. Trying example queries...")
        example_queries = [
            "What are the main components of a multi-agent system?",
            "Explain the concept of test-time scaling",
            "How does bootstrapped reasoning work?"
        ]
        
        for query in example_queries:
            results = semantic_search(query, limit=args.limit)
            
            # Try expanding context for the first result
            if args.expand and not results.empty:
                get_expanded_context(results.iloc[0])
            
            print("\n" + "=" * 80 + "\n")
    else:
        # Perform search with the provided query
        if args.keyword:
            results = hybrid_search(args.query, args.keyword, limit=args.limit)
        else:
            results = semantic_search(args.query, limit=args.limit)
        
        # Try expanding context for the first result
        if args.expand and not results.empty:
            get_expanded_context(results.iloc[0])

if __name__ == "__main__":
    main()
