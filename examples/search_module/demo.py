import os
import sys
import time
from pathlib import Path
import json
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our search components
from search_module.search_client import SearchClient
from search_module.providers.lancedb_provider import LanceDBSearchProvider


def format_result(result, index):
    """Format a search result for display."""
    # Format metadata for display
    import numpy as np
    import json
    
    def numpy_to_python(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [numpy_to_python(item) for item in obj]
        else:
            return obj
    
    # Process metadata to convert all numpy types
    processed_metadata = numpy_to_python(result.metadata)
    
    try:
        metadata_str = json.dumps(processed_metadata, indent=2)
    except TypeError as e:
        # Fallback to a simplified version if JSON serialization fails
        print(f"Warning: Could not serialize metadata: {e}")
        simplified_metadata = {k: str(v) for k, v in processed_metadata.items()}
        metadata_str = json.dumps(simplified_metadata, indent=2)
    
    # Truncate text if too long
    text = result.text
    if len(text) > 200:
        text = text[:200] + "..."
    
    # Create formatted output
    output = [
        f"Result {index + 1} (Score: {result.score:.4f})",
        f"Provider: {result.provider}",
        f"Text: {text}",
        f"Metadata: {metadata_str}",
        "-" * 80
    ]
    
    return "\n".join(output)


def run_search_demo():
    """Run a demonstration of the search capabilities."""
    # Load environment variables
    load_dotenv()
    
    print("=" * 80)
    print("Document Search Module Demo")
    print("=" * 80)
    
    # Initialize the LanceDB provider
    try:
        print("\nInitializing LanceDB search provider...")
        lancedb_provider = LanceDBSearchProvider()
        print(f"[OK] LanceDB provider initialized successfully")
        
        # Initialize the search client
        print("\nInitializing search client...")
        search_client = SearchClient(providers=[lancedb_provider])
        print(f"[OK] Search client initialized with providers: {search_client.get_available_providers()}")
        
        # Wait for user input
        input("\nPress Enter to continue with basic vector search...")
        
        # Basic vector search demo
        query = "What are the main advantages of vector databases?"
        print(f"\n1. Basic Vector Search")
        print(f"Query: '{query}'")
        print("Searching...")
        
        start_time = time.time()
        results = search_client.search(query, limit=3)
        search_time = time.time() - start_time
        
        print(f"Found {len(results)} results in {search_time:.3f} seconds:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(format_result(result, i))
        
        # Wait for user input
        input("\nPress Enter to continue with hybrid search...")
        
        # Hybrid search demo
        query = "What is Docling used for in document processing?"
        print(f"\n2. Hybrid Search (Vector + Keywords)")
        print(f"Query: '{query}'")
        print(f"Auto-extracted keywords will be used")
        print("Searching...")
        
        start_time = time.time()
        results = search_client.hybrid_search(
            query, 
            extract_auto_keywords=True,
            limit=3
        )
        search_time = time.time() - start_time
        
        print(f"Found {len(results)} results in {search_time:.3f} seconds:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(format_result(result, i))
        
        # Wait for user input
        input("\nPress Enter to continue with filtered search...")
        
        # Filtered search demo
        query = "doc_type:pdf OCR and text extraction"
        print(f"\n3. Filtered Search")
        print(f"Query with filters: '{query}'")
        print("This automatically extracts 'doc_type:pdf' as a filter")
        print("Searching...")
        
        start_time = time.time()
        results = search_client.hybrid_search(query, limit=3)
        search_time = time.time() - start_time
        
        print(f"Found {len(results)} results in {search_time:.3f} seconds:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            print(format_result(result, i))
        
        # Document similarity - need to use a document that exists in the DB
        # This depends on what's actually in your database
        try:
            # First, get a list of documents that exist in the database
            print("\nLooking for available documents...")
            
            # Try to find a document in the database to use for similarity search
            sample_results = search_client.search("document", limit=1)
            
            if sample_results and len(sample_results) > 0:
                document_id = sample_results[0].metadata.get("filename")
                
                if document_id:
                    # Wait for user input
                    input(f"\nPress Enter to continue with document similarity search for '{document_id}'...")
                    
                    # Document similarity demo
                    print(f"\n4. Document Similarity Search")
                    print(f"Finding documents similar to: '{document_id}'")
                    print("Searching...")
                    
                    start_time = time.time()
                    results = search_client.similar_documents(document_id, limit=3)
                    search_time = time.time() - start_time
                    
                    print(f"Found {len(results)} results in {search_time:.3f} seconds:")
                    print("-" * 80)
                    
                    for i, result in enumerate(results):
                        print(format_result(result, i))
            else:
                print("\nCould not find a sample document for similarity search.")
        except Exception as e:
            print(f"\nSkipping document similarity search: {str(e)}")
        
        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(run_search_demo())
