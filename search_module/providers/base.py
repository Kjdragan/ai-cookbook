from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
import time
from datetime import datetime


class SearchResult:
    """
    Standard result format for all search providers.
    """
    def __init__(
        self,
        text: str,
        score: float,
        source: str = None,
        metadata: Dict[str, Any] = None,
        provider: str = None,
        vector_distance: Optional[float] = None
    ):
        self.text = text
        self.score = score  # Lower is better for distance metrics
        self.source = source
        self.metadata = metadata or {}
        self.provider = provider  # Which provider returned this result
        self.vector_distance = vector_distance  # Raw vector distance if applicable
        
        # Add convenience properties for common metadata fields
        self.title = self.metadata.get('title', '')
        self.path = self.metadata.get('path', '')
        self.id = self.metadata.get('id', '')
        self.url = self.metadata.get('url', '')
        self.doc_type = self.metadata.get('doc_type', '')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
            "provider": self.provider,
            "vector_distance": self.vector_distance,
            "title": self.title,
            "path": self.path,
            "id": self.id,
            "url": self.url,
            "doc_type": self.doc_type
        }
        
    def __repr__(self):
        return f"SearchResult(score={self.score:.4f}, source={self.source})"


class SearchProvider(ABC):
    """
    Abstract base class for search providers.
    All search implementations must implement this interface.
    """
    
    def __init__(self, name: str):
        """
        Initialize the search provider.
        
        Args:
            name: Unique identifier for this provider
        """
        self.name = name
        self.logger = logging.getLogger(f"search.{name}")
        
        # Configure logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def search(self, query: str, limit: int = 5, **kwargs) -> List[SearchResult]:
        """
        Perform a semantic search using vector embeddings.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
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
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    @abstractmethod
    def similar_documents(
        self, 
        document_id: str,
        limit: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Find documents similar to a given document by ID.
        
        Args:
            document_id: Identifier of the document to find similar matches for
            limit: Maximum number of results to return
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    def _log_search_metrics(
        self, 
        query: str, 
        result_count: int, 
        duration: float,
        search_type: str = "vector"
    ):
        """
        Log search performance metrics.
        
        Args:
            query: The search query
            result_count: Number of results returned
            duration: Search duration in seconds
            search_type: Type of search performed
        """
        self.logger.info(
            f"Search ({search_type}) - Query: '{query}' - Results: {result_count} - Time: {duration:.3f}s"
        )
