import os
import sys
import lancedb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import argparse
from datetime import datetime
import numpy as np

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "docling-playground"))

from datapipeline import DataPipeline

console = Console()

def format_result(result, index):
    """Format a search result for display."""
    text_preview = result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"]
    
    filename = result.get("filename", "Unknown")
    page_info = f"Page(s): {result.get('page_numbers', 'Unknown')}"
    distance = f"Relevance Score: {1.0 - float(result.get('_distance', 0)):.4f}"
    
    return Panel(
        f"[bold cyan]Result #{index+1}[/bold cyan]: [bold]{filename}[/bold]\n"
        f"[dim]{page_info} | {distance}[/dim]\n\n"
        f"{text_preview}",
        expand=False
    )

def search_database(query, keyword=None, limit=5, use_hybrid=True):
    """
    Search the LanceDB database with hybrid search capabilities.
    
    Args:
        query: Semantic search query to find similar content
        keyword: Optional keyword for text-based filtering
        limit: Maximum number of results to return
        use_hybrid: If True, combine semantic search with keyword filtering
    """
    console.print(f"[bold green]Querying database with:[/bold green]")
    console.print(f"[bold]Semantic Query:[/bold] {query}")
    if keyword and use_hybrid:
        console.print(f"[bold]Keyword Filter:[/bold] '{keyword}'")
    
    # Initialize DataPipeline for embedding generation
    pipeline = DataPipeline()
    
    # Connect to LanceDB
    db_path = os.path.join(current_dir, "lancedb_data")
    console.print(f"[dim]Connecting to database at: {db_path}[/dim]")
    db = lancedb.connect(db_path)
    
    # Check if table exists
    if "chunks" not in db.table_names():
        console.print("[bold red]Error: 'chunks' table not found in database.[/bold red]")
        return
    
    # Open table
    table = db.open_table("chunks")
    console.print(f"[dim]Found table with {table.count_rows()} rows[/dim]")
    
    # Generate embedding for the query
    query_embedding = pipeline._generate_embedding(query)
    if not query_embedding or len(query_embedding) != 3072:
        console.print("[bold red]Error: Failed to generate valid embedding for query.[/bold red]")
        return
    
    # Perform search
    start_time = datetime.now()
    
    try:
        # Create base search query with vector
        search_query = table.search(query_embedding, vector_column_name="vector")
        
        # Add keyword filter if provided
        if keyword and use_hybrid:
            # Use SQL filter for text
            search_query = search_query.where(f"text LIKE '%{keyword}%'")
        
        # Execute search with limit
        try:
            # Try newer LanceDB API
            results = search_query.limit(limit).to_pandas()
        except (TypeError, AttributeError):
            # Fall back to older API
            results = search_query.to_pandas().head(limit)
            
    except Exception as e:
        console.print(f"[bold red]Error during search: {str(e)}[/bold red]")
        
        # Try alternative API if the first approach fails
        try:
            console.print("[yellow]Trying alternative search approach...[/yellow]")
            # Get all rows and perform manual filtering
            all_results = table.to_pandas()
            
            # Calculate vector similarity manually (not ideal but fallback)
            all_results['_distance'] = all_results['vector'].apply(
                lambda v: np.linalg.norm(np.array(v) - np.array(query_embedding))
            )
            
            # Apply keyword filter if needed
            if keyword and use_hybrid:
                all_results = all_results[all_results['text'].str.contains(keyword, case=False)]
                
            # Sort by distance and take top results
            results = all_results.sort_values('_distance').head(limit)
            
        except Exception as e2:
            console.print(f"[bold red]Alternative search also failed: {str(e2)}[/bold red]")
            return
    
    end_time = datetime.now()
    search_time = (end_time - start_time).total_seconds()
    
    # Display results
    if len(results) == 0:
        console.print("[yellow]No results found for your query.[/yellow]")
        return
    
    console.print(f"[green]Found {len(results)} results in {search_time:.2f} seconds:[/green]\n")
    
    for i, result in enumerate(results.to_dict('records')):
        console.print(format_result(result, i))
        
    # Display metadata table
    metadata_table = Table(title="Search Metadata")
    metadata_table.add_column("Property", style="cyan")
    metadata_table.add_column("Value", style="green")
    
    metadata_table.add_row("Database Location", db_path)
    metadata_table.add_row("Total Documents in DB", str(table.count_rows()))
    metadata_table.add_row("Search Time", f"{search_time:.2f} seconds")
    metadata_table.add_row("Results Returned", str(len(results)))
    metadata_table.add_row("Vector Dimension", str(len(query_embedding)))
    
    console.print("\n")
    console.print(metadata_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the document database with hybrid semantic and keyword capabilities")
    parser.add_argument("query", help="Semantic search query to find similar content")
    parser.add_argument("--keyword", "-k", help="Optional keyword for text filtering")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results to return")
    parser.add_argument("--semantic-only", action="store_true", help="Use only semantic search without keyword filtering")
    
    args = parser.parse_args()
    
    search_database(
        query=args.query,
        keyword=args.keyword,
        limit=args.limit,
        use_hybrid=not args.semantic_only
    )
