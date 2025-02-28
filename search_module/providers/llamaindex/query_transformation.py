"""
Query transformation utilities for the LlamaIndex search provider.
Implements techniques like query rewriting, expansion and HyDE.
"""

import logging
from typing import Optional, List, Dict, Any

from llama_index_core.llms import LLM
from llama_index_core.schema import QueryBundle
from llama_index_core.query_pipeline import CustomQueryComponent

logger = logging.getLogger(__name__)

class QueryTransformer:
    """
    Implements various query transformation techniques to improve search quality.
    """
    
    def __init__(self, llm: Optional[LLM] = None):
        """
        Initialize the query transformer.
        
        Args:
            llm: LLM instance to use for transformations
        """
        self.llm = llm
        
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query to make it more effective for retrieval.
        Uses an LLM to expand and clarify the query.
        
        Args:
            query: The original query string
            
        Returns:
            Transformed query string
        """
        if not self.llm:
            logger.warning("No LLM provided, returning original query")
            return query
        
        # Prompt template for query rewriting
        prompt = f"""You are an expert search query optimizer. Your task is to rewrite a search query to make it more effective for retrieval from a vector database.
The goal is to add important context and clarify any ambiguities while preserving the original intent.

Original query: {query}

Rewritten query:"""
        
        try:
            response = self.llm.complete(prompt)
            rewritten_query = response.text.strip()
            
            # Simple safety check to ensure we got a reasonable response
            if len(rewritten_query) < 3 or len(rewritten_query) > 500:
                logger.warning(f"Rewritten query seems problematic, using original. Got: {rewritten_query}")
                return query
                
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return query
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into multiple related queries to improve recall.
        
        Args:
            query: The original query string
            
        Returns:
            List of expanded query strings
        """
        if not self.llm:
            logger.warning("No LLM provided, returning original query only")
            return [query]
        
        # Prompt template for query expansion
        prompt = f"""You are an expert in information retrieval. Your task is to expand a search query into 3 alternative phrasings or related queries that will help retrieve relevant information from a document database.
        
Original query: {query}

Generate exactly 3 alternative queries, one per line:"""
        
        try:
            response = self.llm.complete(prompt)
            lines = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Make sure we have at least some results
            if not lines:
                return [query]
                
            # Limit to at most 3 expanded queries plus the original
            expanded_queries = [query] + lines[:3]
            return expanded_queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Implement Hypothetical Document Embedding (HyDE) by generating
        a synthetic document that would answer the query.
        
        Args:
            query: The original query string
            
        Returns:
            A hypothetical document text
        """
        if not self.llm:
            logger.warning("No LLM provided, returning empty document")
            return ""
        
        # Prompt template for HyDE
        prompt = f"""You are an expert in creating hypothetical document passages that would be the perfect answer to a given question or query. Your task is to create a passage that would contain the information needed to address the query below.
        
Query: {query}

Write a short, factual passage that would address this query (1-3 paragraphs):"""
        
        try:
            response = self.llm.complete(prompt)
            hyde_doc = response.text.strip()
            return hyde_doc
            
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            return ""


class HyDEQueryTransformer(CustomQueryComponent):
    """
    HyDE query transformer component for LlamaIndex query pipelines.
    """
    
    def __init__(self, llm: LLM):
        """
        Initialize the HyDE transformer.
        
        Args:
            llm: LLM instance to use for transformations
        """
        self.transformer = QueryTransformer(llm)
        
    def _run(self, query_bundle: QueryBundle, **kwargs) -> QueryBundle:
        """
        Transform a query using the HyDE technique.
        
        Args:
            query_bundle: The original query bundle
            
        Returns:
            Transformed query bundle
        """
        query_str = query_bundle.query_str
        hyde_doc = self.transformer.generate_hypothetical_document(query_str)
        
        # Combine original query with HyDE document
        new_query = f"{query_str}\n\nContext: {hyde_doc}"
        
        # Return new query bundle
        return QueryBundle(
            query_str=new_query,
            custom_embedding_strs=[hyde_doc]
        )
