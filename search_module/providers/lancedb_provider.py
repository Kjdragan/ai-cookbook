import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# Import base search classes
from .base import SearchProvider, SearchResult

def convert_to_search_results(df_results, query=None, logger=None):
    """
    Convert pandas DataFrame results to SearchResult objects.
    
    Args:
        df_results: Pandas DataFrame of search results
        query: The original search query
        logger: Logger instance
        
    Returns:
        List of SearchResult objects
    """
    from .base import SearchResult
    search_results = []
    
    if df_results is None or len(df_results) == 0:
        if logger:
            logger.warning("No results found")
        return search_results
    
    try:
        for _, row in df_results.iterrows():
            # Extract text and metadata
            text = row.get('text', '')
            
            # Extract metadata, excluding vector field and any other large fields
            metadata = {}
            for col in df_results.columns:
                if col not in ['text', 'vector', '_distance']:
                    metadata[col] = row.get(col)
            
            # Get the distance score
            score = row.get('_distance', 1.0)  # Default to 1.0 if no distance
            
            # Try to determine source
            source = None
            if 'path' in metadata:
                source = metadata['path']
            elif 'url' in metadata:
                source = metadata['url']
            elif 'source' in metadata:
                source = metadata['source']
            else:
                source = "unknown"
            
            # Create and add the result
            result = SearchResult(text=text, score=score, source=source, metadata=metadata)
            search_results.append(result)
            
    except Exception as e:
        if logger:
            logger.error(f"Error converting search results: {str(e)}")
    
    return search_results


class LanceDBSearchProvider(SearchProvider):
    """
    Search implementation using LanceDB's vector database.
    """
    
    def __init__(self, 
                db_path: str = "lancedb_data", 
                table_name: str = "chunks", 
                embedding_model: str = "text-embedding-3-large",
                embedding_dim: int = 3072):
        """
        Initialize the LanceDB search provider.
        
        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table to search
            embedding_model: Name of the embedding model, e.g. "text-embedding-3-large"
            embedding_dim: Dimension of the embedding vectors
        """
        super().__init__(name="lancedb")
        
        self.logger = logging.getLogger("search.lancedb")
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        
        # Connect to LanceDB and initialize embedding model
        self._connect_to_lancedb()
        self._initialize_embedding_model()
        
    def _connect_to_lancedb(self):
        """Connect to LanceDB database and open table."""
        self.logger.info(f"Connecting to LanceDB at {self.db_path}")
        self.db = lancedb.connect(self.db_path)
        
        # Get LanceDB version for logging
        lancedb_version = getattr(lancedb, "__version__", "unknown")
        self.logger.info(f"Using LanceDB version: {lancedb_version}")
        
        # Open the table if it exists
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
        else:
            self.logger.warning(f"Table {self.table_name} does not exist")
            self.table = None
            
    def _initialize_embedding_model(self):
        """Initialize the embedding model for LanceDB v0.20."""
        self.logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        
        # Ensure environment variables are loaded
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Try multiple initialization approaches
        try:
            # First attempt: Try using variable_store
            registry = get_registry()
            
            if hasattr(registry, 'variable_store'):
                self.logger.info("Using variable_store for API key")
                registry.variable_store.set("OPENAI_API_KEY", api_key)
                self.embedding_model = registry.get("openai").create(
                    name=self.embedding_model_name
                )
                self.logger.info("Successfully initialized embedding model with variable_store")
                return
                
            # Second attempt: Check if we can use environment variable reference instead of hardcoded value
            self.logger.info("Trying with environment variable reference")
            try:
                self.embedding_model = registry.get("openai").create(
                    name=self.embedding_model_name,
                    api_key="$env:OPENAI_API_KEY"  # Use environment variable reference
                )
                self.logger.info("Successfully initialized embedding model with env var reference")
                return
            except Exception as e:
                self.logger.warning(f"Environment variable reference approach failed: {str(e)}")
            
            # Last resort: Fall back to direct OpenAI client
            self.logger.info("Falling back to direct OpenAI client")
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Test if the client works by making a sample embedding request
            response = client.embeddings.create(
                input="Test embedding",
                model=self.embedding_model_name
            )
            
            # If we get here, client is working
            self.embedding_model = client
            self.is_direct_client = True  # Flag to indicate we're using direct client
            self.logger.info("Successfully initialized direct OpenAI client")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise ValueError(f"Could not initialize embedding model: {str(e)}")
    
    def search(self, query: str, limit: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform a vector-based semantic search.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional parameters:
                - nprobes: Number of clusters to search (default: 10)
                - refine_factor: Refine the top results with exact search (default: 10)
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        self.logger.info(f"Vector search for query: '{query}', limit={limit}")
        
        try:
            # Generate embedding for the query
            if hasattr(self, 'is_direct_client') and self.is_direct_client:
                # Using direct OpenAI client
                response = self.embedding_model.embeddings.create(
                    input=query,
                    model=self.embedding_model_name
                )
                query_embedding = response.data[0].embedding
            else:
                # Using LanceDB registry model
                query_embedding = self.embedding_model.generate_embeddings([query])[0]
            
            # Get search parameters with sensible defaults
            nprobes = kwargs.get('nprobes', 10)
            refine_factor = kwargs.get('refine_factor', 10)
            
            # Ensure table exists before attempting search
            if not self.table:
                self.logger.error(f"Table {self.table_name} does not exist")
                return []
                
            # Execute search with LanceDB v0.20+ approach
            df_results = self.table.search(query_embedding).limit(limit).to_pandas()
            
            # Convert to standardized results
            search_results = convert_to_search_results(df_results, query, self.logger)
            
            duration = time.time() - start_time
            self._log_search_metrics(query, len(search_results), duration, "vector")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return []
    
    def hybrid_search(
        self, 
        query: str, 
        keywords: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5, 
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a hybrid search combining vector search with keyword filtering.
        
        Args:
            query: The semantic search query
            keywords: Optional list of keywords for text filtering
            filters: Optional dictionary of metadata filters
            limit: Maximum number of results to return
            **kwargs: Additional parameters passed to the search method
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        self.logger.info(
            f"Hybrid search for query: '{query}', keywords={keywords}, filters={filters}, limit={limit}"
        )
        
        try:
            # Generate embedding for the query using the same approach as search method
            query_embedding = self.embedding_model.generate_embeddings([query])[0]
            
            # Start with a search query
            search_query = self.table.search(query_embedding)
            
            # Prepare condition strings for both keywords and filters
            conditions = []
            
            # Apply text filtering if keywords are provided
            if keywords and len(keywords) > 0:
                keyword_conditions = []
                for keyword in keywords:
                    # Escape single quotes in keywords
                    safe_keyword = keyword.replace("'", "''")
                    keyword_conditions.append(f"text LIKE '%{safe_keyword}%'")
                
                text_filter = " OR ".join(keyword_conditions)
                conditions.append(f"({text_filter})")
            
            # Apply metadata filtering if filters are provided
            if filters and len(filters) > 0:
                filter_conditions = []
                
                for key, value in filters.items():
                    # Handle potential SQL injection by careful string construction
                    if isinstance(value, str):
                        # Escape single quotes in string values
                        safe_value = value.replace("'", "''")
                        
                        if key.startswith("metadata."):
                            filter_conditions.append(f"{key} = '{safe_value}'")
                        else:
                            filter_conditions.append(f"metadata.{key} = '{safe_value}'")
                    else:
                        # For non-string types
                        if key.startswith("metadata."):
                            filter_conditions.append(f"{key} = {value}")
                        else:
                            filter_conditions.append(f"metadata.{key} = {value}")
                
                filter_string = " AND ".join(filter_conditions)
                conditions.append(f"({filter_string})")
            
            # Combine all conditions with AND
            if conditions:
                combined_conditions = " AND ".join(conditions)
                
                # Apply the combined condition to the search query
                try:
                    search_query = search_query.where(combined_conditions)
                except Exception as e:
                    self.logger.error(f"Error applying conditions: {str(e)}")
                    # If where fails, we'll continue with just the vector search
            
            # Execute the search with LanceDB v0.20+ approach
            df_results = search_query.limit(limit).to_pandas()
            
            # Convert to standard result format
            search_results = convert_to_search_results(
                df_results, 
                query, 
                self.logger
            )
            
            duration = time.time() - start_time
            self._log_search_metrics(query, len(search_results), duration, "hybrid")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    def similar_documents(
        self, 
        document_id: str,
        limit: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document by ID.
        
        Args:
            document_id: Identifier for the document (typically filename)
            limit: Maximum number of results to return
            **kwargs: Additional parameters passed to the search method
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        self.logger.info(f"Finding documents similar to: '{document_id}', limit={limit}")
        
        try:
            # First, find a sample vector from the specified document
            try:
                # Try newer API first
                document_chunk = (
                    self.table.where(f"metadata.filename = '{document_id}'")
                    .limit(1)
                    .to_pandas()
                )
            except AttributeError:
                # Fall back to older API
                document_chunk = (
                    self.table.where(f"metadata.filename = '{document_id}'")
                    .to_pandas()
                    .head(1)
                )
            
            if len(document_chunk) == 0:
                self.logger.warning(f"Document '{document_id}' not found in database")
                return []
            
            # Use the vector from the first chunk for similarity search
            vector = document_chunk.iloc[0]['vector']
            
            # Perform similarity search, excluding chunks from the same document
            try:
                # Try newer API first
                results = (
                    self.table.search(vector)
                    .where(f"metadata.filename != '{document_id}'")
                    .limit(limit)
                    .to_pandas()
                )
            except AttributeError:
                # Fall back to older API
                results = (
                    self.table.search(vector)
                    .where(f"metadata.filename != '{document_id}'")
                    .to_pandas()
                    .head(limit)
                )
            
            # Convert to standard result format
            search_results = convert_to_search_results(
                results, 
                f"Similar to document: {document_id}",
                self.logger
            )
            
            duration = time.time() - start_time
            self._log_search_metrics(
                f"document_similarity:{document_id}", 
                len(search_results), 
                duration, 
                "document_similarity"
            )
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Document similarity search failed: {str(e)}")
            return []
    
    def _manual_vector_search(self, query_embedding, limit):
        """
        Manual vector search implementation as a last resort fallback.
        Used when all other API approaches fail.
        
        Args:
            query_embedding: The vector to search for
            limit: Maximum number of results to return
            
        Returns:
            Pandas DataFrame with search results
        """
        try:
            # Get all data from the table
            all_data = self.table.to_pandas()
            
            # Calculate distances manually
            distances = []
            for _, row in all_data.iterrows():
                vector = row['vector']
                # Calculate Euclidean distance
                distance = np.linalg.norm(np.array(vector) - np.array(query_embedding))
                distances.append(distance)
            
            # Add distances to the DataFrame
            all_data['_distance'] = distances
            
            # Sort by distance and take top results
            return all_data.sort_values('_distance').head(limit)
        except Exception as e:
            self.logger.error(f"Manual vector search failed: {str(e)}")
            raise e
