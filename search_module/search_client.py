import os
import time
import logging
from typing import List, Dict, Any, Optional, Union, Type
from pathlib import Path
import json

# Import providers
from .providers.base import SearchProvider, SearchResult
from .providers.lancedb_provider import LanceDBSearchProvider

# Import utilities
from .utils.query_processing import (
    extract_keywords,
    extract_metadata_and_query,
    clean_query
)


class SearchClient:
    """
    Main search client that orchestrates search across different providers.
    """
    
    def __init__(
        self,
        providers: Optional[List[SearchProvider]] = None,
        default_provider: Optional[str] = None
    ):
        """
        Initialize the search client.
        
        Args:
            providers: List of search providers to use
            default_provider: Name of the default provider to use
        """
        # Configure logging
        self.logger = logging.getLogger("search.client")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Set up providers
        self.providers = {}
        if providers:
            for provider in providers:
                self.providers[provider.name] = provider
        
        # Use LanceDB provider by default if none provided
        if not self.providers:
            try:
                lancedb_provider = LanceDBSearchProvider()
                self.providers[lancedb_provider.name] = lancedb_provider
                self.logger.info(f"Initialized default LanceDB provider")
            except Exception as e:
                self.logger.error(f"Failed to initialize default LanceDB provider: {str(e)}")
        
        # Set default provider
        if default_provider and default_provider in self.providers:
            self.default_provider = default_provider
        elif self.providers:
            self.default_provider = next(iter(self.providers.keys()))
        else:
            raise ValueError("No search providers available")
        
        self.logger.info(f"SearchClient initialized with {len(self.providers)} providers")
        self.logger.info(f"Default provider: {self.default_provider}")
    
    def search(
        self,
        query: str,
        provider: Optional[str] = None,
        limit: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a semantic search.
        
        Args:
            query: Search query
            provider: Name of the provider to use (default: use default_provider)
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        # Clean the query
        clean_query_text = clean_query(query)
        
        # Select the provider
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            self.logger.warning(f"Provider '{provider_name}' not found, using '{self.default_provider}'")
            provider_name = self.default_provider
        
        provider_instance = self.providers[provider_name]
        
        self.logger.info(f"Searching with provider '{provider_name}': '{clean_query_text}'")
        
        # Execute the search
        results = provider_instance.search(clean_query_text, limit=limit, **kwargs)
        
        search_time = time.time() - start_time
        self.logger.info(
            f"Search completed in {search_time:.3f}s, found {len(results)} results"
        )
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        provider: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        extract_auto_keywords: bool = True,
        limit: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a hybrid semantic + keyword search.
        
        Args:
            query: Search query
            provider: Name of the provider to use (default: use default_provider)
            keywords: List of keywords to filter by
            filters: Dictionary of metadata filters
            extract_auto_keywords: Whether to automatically extract keywords from the query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        # Process the query to extract metadata filters if not explicitly provided
        if not filters:
            clean_query_text, extracted_filters = extract_metadata_and_query(query)
            filters = extracted_filters
        else:
            clean_query_text = clean_query(query)
        
        # Extract keywords if requested and not explicitly provided
        if extract_auto_keywords and not keywords:
            keywords = extract_keywords(clean_query_text)
            self.logger.info(f"Auto-extracted keywords: {keywords}")
        
        # Select the provider
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            self.logger.warning(f"Provider '{provider_name}' not found, using '{self.default_provider}'")
            provider_name = self.default_provider
        
        provider_instance = self.providers[provider_name]
        
        self.logger.info(
            f"Hybrid search with provider '{provider_name}': '{clean_query_text}', "
            f"keywords={keywords}, filters={filters}"
        )
        
        # Execute the search
        results = provider_instance.hybrid_search(
            clean_query_text, 
            keywords=keywords,
            filters=filters,
            limit=limit, 
            **kwargs
        )
        
        search_time = time.time() - start_time
        self.logger.info(
            f"Hybrid search completed in {search_time:.3f}s, found {len(results)} results"
        )
        
        return results
    
    def similar_documents(
        self,
        document_id: str,
        provider: Optional[str] = None,
        limit: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document by ID.
        
        Args:
            document_id: Identifier for the document (typically filename)
            provider: Name of the provider to use (default: use default_provider)
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        # Select the provider
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            self.logger.warning(f"Provider '{provider_name}' not found, using '{self.default_provider}'")
            provider_name = self.default_provider
        
        provider_instance = self.providers[provider_name]
        
        self.logger.info(
            f"Finding similar documents to '{document_id}' with provider '{provider_name}'"
        )
        
        # Execute the search
        results = provider_instance.similar_documents(
            document_id,
            limit=limit, 
            **kwargs
        )
        
        search_time = time.time() - start_time
        self.logger.info(
            f"Similar document search completed in {search_time:.3f}s, found {len(results)} results"
        )
        
        return results
    
    def add_provider(self, provider: SearchProvider):
        """
        Add a new search provider to the client.
        
        Args:
            provider: SearchProvider instance to add
        """
        self.providers[provider.name] = provider
        self.logger.info(f"Added provider: {provider.name}")
    
    def set_default_provider(self, provider_name: str):
        """
        Set the default search provider.
        
        Args:
            provider_name: Name of the provider to set as default
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")
        
        self.default_provider = provider_name
        self.logger.info(f"Default provider set to: {provider_name}")
    
    def get_available_providers(self) -> List[str]:
        """
        Get a list of available provider names.
        
        Returns:
            List of provider names
        """
        return list(self.providers.keys())
