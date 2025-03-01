"""
Demo script for testing the LlamaIndex search provider integration
with the existing LanceDB data.
"""

import os
import sys
from dotenv import load_dotenv
import time
import logging
import argparse

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llamaindex_demo")

# Load environment variables
load_dotenv()

# Import our search providers - using direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from providers.llamaindex import LlamaIndexProvider
from providers.lancedb_provider import LanceDBSearchProvider

def run_search_comparison(query, limit=5, use_query_transform=True):
    """Run the same search with both providers and compare results."""
    
    logger.info(f"Running search comparison for query: '{query}'")
    logger.info("-" * 80)
    
    # Initialize both providers
    lance_provider = LanceDBSearchProvider(
        db_path="lancedb_data",
        table_name="chunks",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072
    )
    
    llama_provider = LlamaIndexProvider(
        db_path="lancedb_data",
        table_name="chunks",
        embedding_dim=3072,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        use_query_transform=use_query_transform
    )
    
    # Run LanceDB search
    logger.info("Running LanceDB search...")
    start_time = time.time()
    lance_results = lance_provider.vector_search(query, top_k=limit)
    lance_time = time.time() - start_time
    logger.info(f"LanceDB search completed in {lance_time:.2f} seconds")
    
    # Run LlamaIndex search
    logger.info("Running LlamaIndex search...")
    start_time = time.time()
    llama_results = llama_provider.vector_search(query, top_k=limit)
    llama_time = time.time() - start_time
    logger.info(f"LlamaIndex search completed in {llama_time:.2f} seconds")
    
    # Print results comparison
    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS COMPARISON FOR: '{query}'")
    print("=" * 80)
    
    print("\nLANCEDB RESULTS:")
    print("-" * 80)
    for i, result in enumerate(lance_results):
        print(f"Result {i+1}: (Score: {result.score:.4f})")
        print(f"Document ID: {result.document_id}")
        print(f"Metadata: {result.metadata}")
        print(f"Text: {result.text[:150]}..." if len(result.text) > 150 else f"Text: {result.text}")
        print("-" * 40)
    
    print("\nLLAMAINDEX RESULTS:")
    print("-" * 80)
    for i, result in enumerate(llama_results):
        print(f"Result {i+1}: (Score: {result.score:.4f})")
        print(f"Document ID: {result.document_id}")
        print(f"Metadata: {result.metadata}")
        print(f"Text: {result.text[:150]}..." if len(result.text) > 150 else f"Text: {result.text}")
        print("-" * 40)
    
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"LanceDB search time: {lance_time:.2f} seconds")
    print(f"LlamaIndex search time: {llama_time:.2f} seconds")
    print(f"Difference: {llama_time - lance_time:.2f} seconds")


def run_hybrid_search_comparison(query, keyword, limit=5):
    """Run the same hybrid search with both providers and compare results."""
    
    logger.info(f"Running hybrid search comparison for query: '{query}' with keyword: '{keyword}'")
    logger.info("-" * 80)
    
    # Initialize both providers
    lance_provider = LanceDBSearchProvider(
        db_path="lancedb_data",
        table_name="chunks",
        embedding_model="text-embedding-3-large",
        embedding_dim=3072
    )
    
    llama_provider = LlamaIndexProvider(
        db_path="lancedb_data",
        table_name="chunks",
        embedding_dim=3072,
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )
    
    # Run LanceDB hybrid search
    logger.info("Running LanceDB hybrid search...")
    start_time = time.time()
    lance_results = lance_provider.hybrid_search(query, keyword, top_k=limit)
    lance_time = time.time() - start_time
    logger.info(f"LanceDB hybrid search completed in {lance_time:.2f} seconds")
    
    # Run LlamaIndex hybrid search
    logger.info("Running LlamaIndex hybrid search...")
    start_time = time.time()
    llama_results = llama_provider.hybrid_search(query, keyword, top_k=limit)
    llama_time = time.time() - start_time
    logger.info(f"LlamaIndex hybrid search completed in {llama_time:.2f} seconds")
    
    # Print results comparison
    print("\n" + "=" * 80)
    print(f"HYBRID SEARCH RESULTS COMPARISON FOR: '{query}' + keyword '{keyword}'")
    print("=" * 80)
    
    print("\nLANCEDB RESULTS:")
    print("-" * 80)
    for i, result in enumerate(lance_results):
        print(f"Result {i+1}: (Score: {result.score:.4f})")
        print(f"Document ID: {result.document_id}")
        print(f"Metadata: {result.metadata}")
        print(f"Text: {result.text[:150]}..." if len(result.text) > 150 else f"Text: {result.text}")
        print("-" * 40)
    
    print("\nLLAMAINDEX RESULTS:")
    print("-" * 80)
    for i, result in enumerate(llama_results):
        print(f"Result {i+1}: (Score: {result.score:.4f})")
        print(f"Document ID: {result.document_id}")
        print(f"Metadata: {result.metadata}")
        print(f"Text: {result.text[:150]}..." if len(result.text) > 150 else f"Text: {result.text}")
        print("-" * 40)
    
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"LanceDB hybrid search time: {lance_time:.2f} seconds")
    print(f"LlamaIndex hybrid search time: {llama_time:.2f} seconds")
    print(f"Difference: {llama_time - lance_time:.2f} seconds")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test LlamaIndex integration with LanceDB")
    
    # Define available commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Run semantic search comparison")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--no-transform", action="store_true", 
                              help="Disable query transformation")
    
    # Hybrid search command
    hybrid_parser = subparsers.add_parser("hybrid", help="Run hybrid search comparison")
    hybrid_parser.add_argument("query", help="Semantic search query")
    hybrid_parser.add_argument("keyword", help="Keyword to filter results by")
    hybrid_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    # Similar documents command
    similar_parser = subparsers.add_parser("similar", help="Find similar documents")
    similar_parser.add_argument("document_id", help="Document ID to find similar documents for")
    similar_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.command == "search":
        run_search_comparison(
            query=args.query, 
            limit=args.limit,
            use_query_transform=not args.no_transform
        )
    elif args.command == "hybrid":
        run_hybrid_search_comparison(
            query=args.query,
            keyword=args.keyword,
            limit=args.limit
        )
    elif args.command == "similar":
        # Add similar document search when implemented
        print("Similar document search not yet implemented")
    else:
        print("Please specify a command: search, hybrid, or similar")
        sys.exit(1)
