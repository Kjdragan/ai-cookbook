"""
LlamaIndex search provider implementation.
Integrates with existing LanceDB infrastructure for vector search.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import lancedb
from dotenv import load_dotenv

# Import LlamaIndex components
from llama_index_vector_stores_lancedb import LanceDBVectorStore
from llama_index_core.embeddings import OpenAIEmbedding
from llama_index_core.schema import TextNode, Document, NodeWithScore
from llama_index_core.indices.vector_store import VectorStoreIndex
from llama_index_core.retrievers import VectorIndexRetriever
from llama_index_core.response_synthesizers import CompactAndRefine
from llama_index_core.query_engine import RetrieverQueryEngine

# Import search provider base classes
from ..search_provider import SearchProvider, SearchResult
from .llm_config import LLMFactory
from .query_transformation import QueryTransformer

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class LlamaIndexProvider(SearchProvider):
    """
    LlamaIndex implementation of the SearchProvider interface.
    Uses LlamaIndex as a query layer on top of LanceDB.
    """
    
    def __init__(
        self,
        db_path: str = "lancedb_data",
        table_name: str = "documents",
        embedding_dim: int = 3072,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        hybrid_alpha: float = 0.5,
        use_query_transform: bool = True,
        **kwargs
    ):
        """
        Initialize the LlamaIndex search provider.
        
        Args:
            db_path: Path to LanceDB database
            table_name: Name of the LanceDB table
            embedding_dim: Dimension of the embedding vectors
            llm_provider: LLM provider to use (openai or deepseek)
            llm_model: LLM model to use
            hybrid_alpha: Weight of the semantic search component in hybrid search
            use_query_transform: Whether to use query transformation
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.hybrid_alpha = hybrid_alpha
        self.use_query_transform = use_query_transform
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        
        # Create connection to LanceDB
        self._init_lancedb()
        
        # Set up LlamaIndex components
        self._init_llamaindex(llm_provider, llm_model)
        
    def _init_lancedb(self):
        """Initialize connection to LanceDB."""
        try:
            self.db = lancedb.connect(self.db_path)
            
            # Check if table exists
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
                self.logger.info(f"Connected to existing LanceDB table: {self.table_name}")
            else:
                self.logger.warning(f"Table {self.table_name} not found in LanceDB database")
                self.table = None
            
        except Exception as e:
            self.logger.error(f"Error connecting to LanceDB: {e}")
            self.db = None
            self.table = None
    
    def _init_llamaindex(self, llm_provider: str, llm_model: str):
        """Initialize LlamaIndex components."""
        try:
            # Create LLM instance
            self.llm = LLMFactory.create_llm(
                provider=llm_provider,
                model_name=llm_model,
                temperature=0.0
            )
            
            # Create query transformer
            self.query_transformer = QueryTransformer(self.llm)
            
            # Set up embedding model
            self.embed_model = OpenAIEmbedding(
                model_name="text-embedding-3-large",
                dimensions=self.embedding_dim
            )
            
            # Create vector store and index if table exists
            if self.table is not None:
                self.vector_store = LanceDBVectorStore(
                    uri=self.db_path,
                    table_name=self.table_name,
                    embed_dim=self.embedding_dim
                )
                
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model
                )
                
                self.logger.info("LlamaIndex components initialized successfully")
            else:
                self.vector_store = None
                self.index = None
                self.logger.warning("LlamaIndex index not created due to missing table")
            
        except Exception as e:
            self.logger.error(f"Error initializing LlamaIndex components: {e}")
            self.llm = None
            self.query_transformer = None
            self.embed_model = None
            self.vector_store = None
            self.index = None
    
    def _node_to_search_result(self, node: NodeWithScore) -> SearchResult:
        """
        Convert a LlamaIndex node to a SearchResult.
        
        Args:
            node: LlamaIndex node with score
            
        Returns:
            SearchResult object
        """
        # Extract metadata from node
        metadata = node.node.metadata or {}
        
        # Get text content
        text = node.node.text or ""
        
        # Get score (1 - distance for compatibility with existing provider)
        score = 1.0 - (node.score or 0.0)
        
        # Create SearchResult
        return SearchResult(
            document_id=metadata.get("document_id", metadata.get("id", "")),
            text=text,
            metadata=metadata,
            score=score
        )
    
    def vector_search(
        self, 
        query: str, 
        top_k: int = 5, 
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform vector search using LlamaIndex.
        
        Args:
            query: The search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            self.logger.error("Vector search failed: LlamaIndex components not initialized")
            return []
        
        try:
            # Transform query if enabled
            if self.use_query_transform:
                transformed_query = self.query_transformer.rewrite_query(query)
                self.logger.info(f"Transformed query: {transformed_query}")
            else:
                transformed_query = query
            
            # Create retriever with metadata filtering if provided
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k,
            )
            
            # Apply metadata filter if provided
            # This is handled at the query level since LanceDBVectorStore doesn't support metadata filtering directly
            if filter_metadata:
                # We'll implement basic filtering here by retrieving more results and then filtering
                retriever.similarity_top_k = top_k * 3  # Retrieve more to allow for filtering
            
            # Perform retrieval
            nodes = retriever.retrieve(transformed_query)
            
            # Apply metadata filtering if needed
            if filter_metadata:
                filtered_nodes = []
                for node in nodes:
                    meta = node.node.metadata or {}
                    match = True
                    for key, value in filter_metadata.items():
                        if key not in meta or meta[key] != value:
                            match = False
                            break
                    if match:
                        filtered_nodes.append(node)
                nodes = filtered_nodes[:top_k]  # Limit to top_k after filtering
            
            # Convert to SearchResult objects
            return [self._node_to_search_result(node) for node in nodes]
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return []
    
    def hybrid_search(
        self, 
        query: str, 
        keyword: str = "", 
        top_k: int = 5, 
        alpha: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Semantic search query
            keyword: Keyword search term
            top_k: Number of results to return
            alpha: Weight of semantic vs keyword search (0-1), higher means more weight to semantic
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        if self.table is None:
            self.logger.error("Hybrid search failed: LanceDB table not available")
            return []
        
        try:
            # Use provided alpha or default
            alpha = alpha if alpha is not None else self.hybrid_alpha
            
            # If no keyword provided, fall back to vector search
            if not keyword:
                return self.vector_search(query, top_k, filter_metadata)
            
            # Transform query if enabled
            if self.use_query_transform:
                transformed_query = self.query_transformer.rewrite_query(query)
            else:
                transformed_query = query
            
            # Get embeddings for the query
            query_embedding = self.embed_model.get_text_embedding(transformed_query)
            
            # Construct LanceDB query with hybrid search
            lance_query = self.table.search(query_embedding, query_type="vector")
            
            # Add text search component
            lance_query = lance_query.where(f"text LIKE '%{keyword}%'")
            
            # Apply metadata filter if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    lance_query = lance_query.where(f"{key} = '{value}'")
            
            # Execute query and limit results
            results = lance_query.limit(top_k).to_pandas()
            
            # Convert to SearchResult objects
            search_results = []
            for _, row in results.iterrows():
                result = SearchResult(
                    document_id=row.get("document_id", row.get("id", "")),
                    text=row.get("text", ""),
                    metadata={k: v for k, v in row.items() 
                             if k not in ["vector", "_distance", "text", "document_id", "id"]},
                    score=1.0 - row.get("_distance", 0.0)  # Convert distance to score
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def similar_document_search(
        self, 
        document_id: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Find documents similar to the specified document.
        
        Args:
            document_id: ID of the document to find similar documents for
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        if self.table is None:
            self.logger.error("Similar document search failed: LanceDB table not available")
            return []
        
        try:
            # Find the document with the given ID
            source_doc = self.table.where(f"document_id = '{document_id}'").to_pandas()
            
            if len(source_doc) == 0:
                # Try alternate ID field
                source_doc = self.table.where(f"id = '{document_id}'").to_pandas()
            
            if len(source_doc) == 0:
                self.logger.error(f"Document with ID {document_id} not found")
                return []
            
            # Get the vector for the source document
            source_vector = source_doc.iloc[0]["vector"]
            
            # Search for similar documents
            lance_query = self.table.search(source_vector, query_type="vector")
            
            # Apply metadata filter if provided
            if filter_metadata:
                for key, value in filter_metadata.items():
                    lance_query = lance_query.where(f"{key} = '{value}'")
            
            # Get top_k+1 results (to account for the source document itself)
            results = lance_query.limit(top_k + 1).to_pandas()
            
            # Filter out the source document
            results = results[results["document_id"] != document_id]
            if len(results) == len(results):  # No rows were filtered, try alternate ID field
                results = results[results["id"] != document_id]
            
            # Limit to top_k
            results = results.head(top_k)
            
            # Convert to SearchResult objects
            search_results = []
            for _, row in results.iterrows():
                result = SearchResult(
                    document_id=row.get("document_id", row.get("id", "")),
                    text=row.get("text", ""),
                    metadata={k: v for k, v in row.items() 
                             if k not in ["vector", "_distance", "text", "document_id", "id"]},
                    score=1.0 - row.get("_distance", 0.0)  # Convert distance to score
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in similar document search: {e}")
            return []
