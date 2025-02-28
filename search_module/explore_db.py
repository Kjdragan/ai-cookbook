import os
import sys
import pandas as pd
import json
import numpy as np
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import lancedb and related utilities
import lancedb

# Log LanceDB version for compatibility information
lancedb_version = getattr(lancedb, "__version__", "unknown")
print(f"Using LanceDB version: {lancedb_version}")

# Import embedding generation from the provider
from search_module.providers.lancedb_provider import LanceDBSearchProvider


# Helper function to make data JSON serializable
def json_serializable(item):
    """
    Convert numpy arrays and other non-serializable types to Python types.
    
    This function recursively processes complex data structures to ensure
    all elements are JSON serializable, handling various NumPy types and
    other special cases that might cause issues with different LanceDB versions.
    
    Args:
        item: Any Python or NumPy object to convert
        
    Returns:
        JSON serializable version of the input
    """
    if item is None:
        return None
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, dict):
        return {key: json_serializable(value) for key, value in item.items()}
    elif isinstance(item, list) or isinstance(item, tuple):
        return [json_serializable(elem) for elem in item]
    elif isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, (np.bool_)):
        return bool(item)
    elif hasattr(item, 'isoformat'):  # Handle datetime objects
        return item.isoformat()
    elif hasattr(item, '__dict__'):  # Handle custom objects
        return json_serializable(item.__dict__)
    else:
        # Try to convert anything else to a string if all else fails
        try:
            json.dumps(item)  # Test if it's already serializable
            return item
        except (TypeError, OverflowError):
            return str(item)


def print_search_results(results):
    """Print search results in a formatted way."""
    if not results:
        print("No results found.")
        return
        
    print(f"Found {len(results)} results:")
    
    try:
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result.score:.4f}")
            
            # Handle different result formats
            if hasattr(result, 'title') and result.title:
                print(f"Title: {result.title}")
            
            if hasattr(result, 'path') and result.path:
                print(f"Path: {result.path}")
                
            if hasattr(result, 'metadata'):
                meta = result.metadata
                if isinstance(meta, dict):
                    for key, value in meta.items():
                        if key not in ['vector']:  # Skip vector field
                            print(f"{key}: {value}")
            
            # Always print text content
            if hasattr(result, 'text') and result.text:
                print("\nContent:")
                print("-" * 80)
                print(result.text[:500] + ("..." if len(result.text) > 500 else ""))
                print("-" * 80)
            
    except Exception as e:
        print(f"Error displaying results: {str(e)}")


def run_search_query(query, limit=5):
    """Run a search query against the LanceDB chunks table."""
    print(f"\n\nRunning search for: '{query}'")
    print("=" * 80)
    
    # Connect to the database
    lancedb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lancedb_data")
    
    try:
        # Create a LanceDB provider instance
        provider = LanceDBSearchProvider(db_path=lancedb_path)
        
        # Use the provider's search method with LanceDB v0.20 approach
        results = provider.search(query, limit=limit)
        
        # Print results
        print_search_results(results)
            
    except Exception as e:
        print(f"Error during search: {str(e)}")
        
        # Check if API key is available (without showing it)
        if os.getenv("OPENAI_API_KEY"):
            print("- OpenAI API key is set in environment")
        else:
            print("- OpenAI API key is NOT set in environment variable")


def explore_lancedb():
    """Explore LanceDB database contents and schema."""
    
    # Load environment variables
    load_dotenv()
    
    print("=" * 80)
    print("LanceDB Explorer")
    print("=" * 80)
    
    # Use the standardized path to lancedb_data
    lancedb_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lancedb_data")
    print(f"\nConnecting to LanceDB at: {lancedb_path}")
    
    db = lancedb.connect(lancedb_path)
    
    # List available tables
    tables = db.table_names()
    print(f"\nAvailable tables: {tables}")
    
    if "chunks" not in tables:
        print("Error: 'chunks' table not found")
        return
    
    # Open the chunks table
    chunks_table = db.open_table("chunks")
    
    # Get schema information
    schema = chunks_table.schema
    print("\nTable schema:")
    for field in schema:
        print(f"  - {field.name}: {field.type}")
    
    # Get sample data to understand metadata structure
    print("\nFetching sample data...")
    try:
        # Try newer API first
        sample_data = chunks_table.limit(1).to_pandas()
    except AttributeError:
        # Fall back to older API
        sample_data = chunks_table.to_pandas().head(1)
    
    if len(sample_data) == 0:
        print("No data found in the table")
        return
        
    # Display vector dimensions
    vector_dim = len(sample_data.iloc[0]['vector'])
    print(f"\nVector dimensions: {vector_dim}")
    
    # Extract metadata keys
    if 'metadata' in sample_data.columns:
        metadata = sample_data.iloc[0]['metadata']
        print("\nMetadata structure:")
        # Convert metadata to JSON serializable format
        metadata_serializable = json_serializable(metadata)
        print(json.dumps(metadata_serializable, indent=2))
    
    # Get table statistics
    try:
        # Try newer API first
        table_stats = chunks_table.stats()
        print(f"\nTable statistics:")
        print(f"  - Row count: {table_stats.get('count', 'Unknown')}")
    except AttributeError:
        # Fall back to older API or alternative approach
        try:
            # Try to count rows
            row_count = len(chunks_table.to_pandas())
            print(f"\nTable statistics:")
            print(f"  - Row count: {row_count}")
        except Exception as e:
            print(f"\nCould not get table statistics: {str(e)}")
    
    # Show examples of queries that could be run
    print("\nExample search queries that could be run:")
    print("""
    # Basic vector search
    results = table.search(query_vector).to_pandas()
    
    # Hybrid search with text filtering
    results = table.search(query_vector).where("text LIKE '%keyword%'").to_pandas()
    
    # Metadata filtering
    results = table.search(query_vector).where("metadata.doc_type = 'pdf'").to_pandas()
    """)
    
    # Print out a raw example of a metadata record to help understand the structure
    print("\nRaw row data (first record):")
    row_dict = {}
    for column in sample_data.columns:
        if column == 'vector':
            row_dict[column] = f"<vector with {len(sample_data.iloc[0][column])} dimensions>"
        else:
            row_dict[column] = json_serializable(sample_data.iloc[0][column])
    print(json.dumps(row_dict, indent=2))
    
    # Show unique document titles in the database
    try:
        # Get all rows but only select metadata column to reduce memory usage
        all_data = chunks_table.to_pandas()[['metadata']]
        
        # Extract titles from metadata
        titles = [row['metadata']['title'] for row in all_data.to_dict('records')]
        unique_titles = sorted(set(titles))
        
        print(f"\nUnique documents in database ({len(unique_titles)}):")
        for title in unique_titles:
            # Count chunks per document
            chunks_count = titles.count(title)
            print(f"  - {title} ({chunks_count} chunks)")
    except Exception as e:
        print(f"\nCould not extract document titles: {str(e)}")
    
    print("\nLanceDB exploration complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LanceDB Explorer Tool")
    parser.add_argument("--search", type=str, help="Run a search query against the database")
    parser.add_argument("--limit", type=int, default=5, help="Limit the number of search results (default: 5)")
    
    args = parser.parse_args()
    
    if args.search:
        run_search_query(args.search, args.limit)
    else:
        explore_lancedb()
