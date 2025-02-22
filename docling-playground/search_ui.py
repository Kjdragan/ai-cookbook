import streamlit as st
import lancedb
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import os
from openai import OpenAI
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from dotenv import load_dotenv
import pyarrow as pa
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

def initialize_embedding_func():
    """Initialize OpenAI embedding function."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
        
    embedding_func = get_registry().get("openai").create(
        name="text-embedding-3-large",
        api_key=api_key
    )
    return embedding_func

def initialize_db():
    """Initialize the database connection and create FTS index if needed."""
    try:
        # Connect to the database
        db = lancedb.connect("lancedb")
        
        # Get or create the table
        try:
            table = db.open_table("chunks")
        except Exception as e:
            st.error(f"Error opening table: {str(e)}")
            # Create a sample table with the correct schema
            sample_data = {
                "text": ["Sample text"],
                "metadata": [{
                    "filename": "sample.txt",
                    "page_numbers": [1],
                    "title": "Sample"
                }],
                "vector": np.zeros(3072).tolist()  # 3072 for text-embedding-3-large
            }
            table = db.create_table("chunks", data=sample_data)
        
        # Try to create FTS index without replace=True first
        try:
            table.create_fts_index("text")
            st.success("Created/Updated FTS index on text column")
        except Exception as e:
            st.warning(f"Could not create FTS index: {str(e)}")
            try:
                # If that fails, try with replace=True
                table.create_fts_index("text", replace=True)
                st.success("Created/Updated FTS index on text column (with replace)")
            except Exception as e2:
                st.error(f"Failed to create FTS index: {str(e2)}")
                # Continue without FTS - hybrid search will still work but may be less effective
                pass
        
        return table
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
        return None

def search_documents(
    table,
    query: str,
    k: int = 5,
    query_type: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Search documents using hybrid search (vector + full-text).
    
    Args:
        table: LanceDB table
        query: Search query
        k: Number of results to return
        query_type: Type of search ("hybrid", "vector", or "fts")
    """
    try:
        # Create search query based on type
        if query_type == "hybrid":
            # For hybrid search, we need both vector and text
            embedding_func = initialize_embedding_func()
            vector = embedding_func.generate_embeddings([query])[0]
            results = (
                table.search(query_type="hybrid")
                .vector(vector)
                .text(query)
                .limit(k)
                .to_pandas()
            )
        elif query_type == "vector":
            # For vector search, we need to create the embedding
            embedding_func = initialize_embedding_func()
            vector = embedding_func.generate_embeddings([query])[0]
            results = (
                table.search(vector)
                .limit(k)
                .to_pandas()
            )
        else:  # fts
            # For full-text search, just use the text query
            results = (
                table.search(query, query_type="fts")
                .limit(k)
                .to_pandas()
            )
            
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def display_results(results: pd.DataFrame):
    """Display search results in a nice format."""
    if results.empty:
        st.warning("No results found.")
        return
        
    st.subheader(f"Found {len(results)} results")
    for _, row in results.iterrows():
        # Get score based on available columns
        score = "N/A"
        for score_col in ["_distance", "_score", "_relevance"]:
            if score_col in row and pd.notna(row[score_col]):
                score = f"{row[score_col]:.4f}"
                break
            
        with st.expander(f"Score: {score}"):
            # Display metadata if available
            meta = row.get("metadata", {})
            if meta:
                st.markdown(f"**Source**: {meta.get('filename', 'Unknown')}")
                st.markdown(f"**Page**: {meta.get('page_numbers', ['N/A'])[0]}")
                if meta.get("title"):
                    st.markdown(f"**Section**: {meta['title']}")
                
            # Display text content
            st.markdown("**Content:**")
            st.markdown(row.get("text", "No content available"))

def main():
    st.set_page_config(
        page_title="Document Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("📚 Document Search")
    
    # Initialize components
    table = initialize_db()
    
    # Search interface
    with st.form("search_form"):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            query = st.text_input("Enter your search query:", placeholder="What would you like to find?")
        
        with col2:
            search_type = st.selectbox(
                "Search Type",
                ["Hybrid", "Vector", "Full-text"],
                index=0
            )
        
        with col3:
            k = st.number_input("Results", min_value=1, max_value=20, value=5)
            
        submitted = st.form_submit_button("🔍 Search")
    
    # Perform search when form is submitted
    if submitted and query:
        with st.spinner("Searching..."):
            results = search_documents(
                table,
                query,
                k=k,
                query_type=search_type.lower()
            )
            
            if isinstance(results, pd.DataFrame):
                display_results(results)
            else:
                st.error("Search failed. Please try again.")

if __name__ == "__main__":
    main()
