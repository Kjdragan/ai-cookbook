"""
Citation tracking system for LlamaIndex results.

This module provides tools to track and format citations for retrieved documents
and their usage in generated responses.
"""

import re
import logging
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle, Document
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.llms import LLM
from llama_index.core.response import Response

logger = logging.getLogger(__name__)


class CitationStyle(str, Enum):
    """Citation formatting styles."""
    
    INLINE = "inline"  # Citations inline in text [1]
    FOOTNOTE = "footnote"  # Footnotes with numbers
    ENDNOTE = "endnote"  # Sources listed at the end
    AUTHOR_DATE = "author_date"  # Author-date style (Smith, 2023)
    NONE = "none"  # No citations


@dataclass
class CitationConfig:
    """Configuration for citation tracking and formatting."""
    
    # How to format citations
    style: CitationStyle = CitationStyle.ENDNOTE
    
    # Whether to include page numbers in citations when available
    include_page_numbers: bool = True
    
    # Whether to deduplicate repeated citations
    deduplicate_citations: bool = True
    
    # Minimum confidence score for a source to be included (0.0-1.0)
    min_confidence_score: float = 0.0
    
    # Maximum number of citations to include per response
    max_citations: int = 20
    
    # Format string for citation text
    # Available variables: {id}, {title}, {author}, {date}, {source}, {page}, {url}
    citation_format: str = "{id}. {title}, {author} ({date}){page}"
    
    # Placeholder text for missing metadata fields
    missing_metadata_placeholder: str = "N/A"
    
    # Whether to add special tokens during LLM citation processing
    use_llm_citation_markup: bool = True
    
    # Special tokens for citation markup (if use_llm_citation_markup=True)
    citation_start_token: str = "[cite:"
    citation_end_token: str = "]"


class CitationTracker:
    """
    System for tracking citations from source documents to generated responses.
    
    This component tracks which source documents are used in responses and
    formats citations according to the specified style.
    """
    
    def __init__(
        self,
        config: Optional[CitationConfig] = None,
    ):
        """
        Initialize the citation tracker.
        
        Args:
            config: Citation tracking configuration
        """
        self.config = config or CitationConfig()
        
        # Track source documents and their citations
        self.sources: Dict[str, Dict[str, Any]] = {}
        
        # Track citations used in the most recent response
        self.current_citations: List[str] = []
        
        # Map of node IDs to source IDs
        self.node_to_source_map: Dict[str, str] = {}
    
    def register_source_nodes(self, nodes: List[NodeWithScore]) -> None:
        """
        Register source nodes for citation tracking.
        
        Args:
            nodes: List of source nodes
        """
        for node in nodes:
            if not node.node:
                continue
                
            # Generate source ID if not already tracked
            node_id = self._get_node_id(node.node)
            
            # Check if we already have this node
            if node_id in self.node_to_source_map:
                continue
                
            # Create source citation
            source = self._extract_source_metadata(node.node)
            source_id = self._generate_source_id(source)
            
            # Store the source
            self.sources[source_id] = source
            
            # Map node ID to source ID
            self.node_to_source_map[node_id] = source_id
            
            logger.debug(f"Registered source: {source_id} - {source.get('title', 'Untitled')}")
    
    def process_response(self, response: Response) -> Response:
        """
        Process a response to add citations.
        
        Args:
            response: The response to process
            
        Returns:
            Processed response with citations
        """
        if not response:
            return response
            
        # Reset current citations
        self.current_citations = []
        
        # Collect source node information
        if hasattr(response, "source_nodes") and response.source_nodes:
            self.register_source_nodes(response.source_nodes)
            
            # Track citations used in this response
            for node in response.source_nodes:
                if not node.node:
                    continue
                    
                # Skip nodes with low confidence if configured
                if (self.config.min_confidence_score > 0 and 
                    node.score is not None and 
                    node.score < self.config.min_confidence_score):
                    continue
                    
                # Get the source ID for this node
                node_id = self._get_node_id(node.node)
                source_id = self.node_to_source_map.get(node_id)
                
                if source_id and source_id not in self.current_citations:
                    self.current_citations.append(source_id)
                    
                    # Apply maximum citations limit
                    if len(self.current_citations) >= self.config.max_citations:
                        break
        
        # Format the response text with citations
        if response.response:
            response.response = self._format_response_with_citations(response.response)
            
        return response
    
    def get_citation_text(self) -> str:
        """
        Get formatted citation text for all sources used in the current response.
        
        Returns:
            Formatted citation text
        """
        if not self.current_citations:
            return ""
            
        citations = []
        
        for idx, source_id in enumerate(self.current_citations, 1):
            source = self.sources.get(source_id, {})
            
            # Format the citation
            citation = self._format_citation(idx, source)
            citations.append(citation)
            
        # Join all citations
        if self.config.style == CitationStyle.FOOTNOTE:
            return "\n\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(citations, 1))
        else:
            return "\n\n**Sources:**\n" + "\n".join(citations)
    
    def _extract_source_metadata(self, node: TextNode) -> Dict[str, Any]:
        """
        Extract metadata from a node for citation purposes.
        
        Args:
            node: The source node
            
        Returns:
            Dict containing source metadata
        """
        # Initialize with empty fields
        source = {
            "title": self.config.missing_metadata_placeholder,
            "author": self.config.missing_metadata_placeholder,
            "date": self.config.missing_metadata_placeholder,
            "source": self.config.missing_metadata_placeholder,
            "url": self.config.missing_metadata_placeholder,
            "page": "",
            "text_snippet": node.text[:100] + "..." if len(node.text) > 100 else node.text
        }
        
        # Extract metadata from node if available
        if hasattr(node, "metadata") and node.metadata:
            metadata = node.metadata
            
            # Common metadata fields
            for field in ["title", "author", "date", "source", "url"]:
                if field in metadata and metadata[field]:
                    source[field] = metadata[field]
            
            # Handle page numbers
            if self.config.include_page_numbers:
                page = metadata.get("page") or metadata.get("page_number")
                if page:
                    source["page"] = f", page {page}"
            
            # Try to extract from document metadata if available
            if "document" in metadata and hasattr(metadata["document"], "metadata"):
                doc_metadata = metadata["document"].metadata
                for field in ["title", "author", "date", "source", "url"]:
                    if (field in doc_metadata and doc_metadata[field] and 
                        source[field] == self.config.missing_metadata_placeholder):
                        source[field] = doc_metadata[field]
        
        return source
    
    def _generate_source_id(self, source: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a source.
        
        Args:
            source: Source metadata
            
        Returns:
            Unique source ID
        """
        # Create a key using available metadata
        key_parts = [
            source.get("title", ""),
            source.get("author", ""),
            source.get("date", ""),
            source.get("source", ""),
            source.get("text_snippet", "")
        ]
        
        # Create a hash
        key = "|".join(str(part) for part in key_parts if part)
        hash_obj = hashlib.md5(key.encode())
        
        return hash_obj.hexdigest()[:10]
    
    def _get_node_id(self, node: TextNode) -> str:
        """
        Get a unique ID for a node.
        
        Args:
            node: The node
            
        Returns:
            Unique node ID
        """
        if hasattr(node, "node_id") and node.node_id:
            return node.node_id
            
        # Fallback to hash
        text_sample = node.text[:100]
        hash_obj = hashlib.md5(text_sample.encode())
        
        return hash_obj.hexdigest()
    
    def _format_citation(self, index: int, source: Dict[str, Any]) -> str:
        """
        Format a citation for a source.
        
        Args:
            index: Citation index
            source: Source metadata
            
        Returns:
            Formatted citation string
        """
        # Replace variables in the citation format
        formatted = self.config.citation_format.format(
            id=index,
            title=source.get("title", self.config.missing_metadata_placeholder),
            author=source.get("author", self.config.missing_metadata_placeholder),
            date=source.get("date", self.config.missing_metadata_placeholder),
            source=source.get("source", self.config.missing_metadata_placeholder),
            page=source.get("page", ""),
            url=source.get("url", self.config.missing_metadata_placeholder)
        )
        
        return formatted
    
    def _format_response_with_citations(self, response_text: str) -> str:
        """
        Format response text with citations according to the configured style.
        
        Args:
            response_text: Original response text
            
        Returns:
            Response text with citations
        """
        if not self.current_citations:
            return response_text
            
        # Different formatting based on citation style
        if self.config.style == CitationStyle.INLINE:
            # Add citation list at the end
            citations_text = self.get_citation_text()
            return response_text + citations_text
            
        elif self.config.style == CitationStyle.FOOTNOTE:
            # Add footnotes
            citations_text = self.get_citation_text()
            return response_text + citations_text
            
        elif self.config.style == CitationStyle.ENDNOTE:
            # Add endnotes
            citations_text = self.get_citation_text()
            return response_text + citations_text
            
        elif self.config.style == CitationStyle.AUTHOR_DATE:
            # Add author-date style citations
            citations_text = self.get_citation_text()
            return response_text + citations_text
            
        elif self.config.style == CitationStyle.NONE:
            # No citations
            return response_text
            
        else:
            # Default to endnote
            citations_text = self.get_citation_text()
            return response_text + citations_text


class CitationPostprocessor(BaseNodePostprocessor):
    """
    Postprocessor that adds citation tracking to retrieved nodes.
    
    This processor adds citation metadata to nodes and registers them
    with the citation tracker.
    """
    
    def __init__(
        self,
        citation_tracker: CitationTracker,
    ):
        """
        Initialize the citation postprocessor.
        
        Args:
            citation_tracker: Citation tracker instance
        """
        self.citation_tracker = citation_tracker
        
    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        Process nodes to add citation information.
        
        Args:
            nodes: List of nodes to process
            query_bundle: Optional query bundle
            
        Returns:
            Processed nodes
        """
        # Register the nodes with the citation tracker
        self.citation_tracker.register_source_nodes(nodes)
        
        # Return the nodes unchanged
        return nodes
        

class CitationResponseSynthesizer(BaseSynthesizer):
    """
    Wrapper for response synthesizers that adds citation tracking.
    
    This synthesizer wraps another synthesizer and adds citation tracking
    to the generated responses.
    """
    
    def __init__(
        self,
        response_synthesizer: BaseSynthesizer,
        citation_tracker: CitationTracker,
    ):
        """
        Initialize the citation response synthesizer.
        
        Args:
            response_synthesizer: Base synthesizer to wrap
            citation_tracker: Citation tracker instance
        """
        super().__init__()
        self.base_synthesizer = response_synthesizer
        self.citation_tracker = citation_tracker
        
    def _get_prompts(self):
        """Get prompts from the base synthesizer."""
        if hasattr(self.base_synthesizer, "_get_prompts"):
            return self.base_synthesizer._get_prompts()
        return {}
        
    def _update_prompts(self, **kwargs: Any):
        """Update prompts in the base synthesizer."""
        if hasattr(self.base_synthesizer, "_update_prompts"):
            self.base_synthesizer._update_prompts(**kwargs)
            
    async def aget_response(
        self,
        query_str: str,
        text_chunks: List[str],
        **kwargs: Any
    ) -> Response:
        """Async version of get_response."""
        # Convert to NodeWithScore format if needed
        if not isinstance(text_chunks[0], NodeWithScore):
            nodes = [
                NodeWithScore(node=TextNode(text=chunk), score=1.0)
                for chunk in text_chunks
            ]
        else:
            nodes = text_chunks
            
        # Register nodes and get response
        self.citation_tracker.register_source_nodes(nodes)
        
        # Use base synthesizer for async response
        if hasattr(self.base_synthesizer, "aget_response"):
            response = await self.base_synthesizer.aget_response(query_str, text_chunks, **kwargs)
        else:
            # Fall back to synchronous method
            response = self.base_synthesizer.get_response(query_str, text_chunks, **kwargs)
            
        # Add citations
        processed_response = self.citation_tracker.process_response(response)
        return processed_response
        
    def get_response(
        self,
        query_str: str,
        text_chunks: List[str],
        **kwargs: Any
    ) -> Response:
        """
        Get a response with citation tracking.
        
        Args:
            query_str: Query string
            text_chunks: Text chunks or nodes to use
            **kwargs: Additional arguments
            
        Returns:
            Response with citations
        """
        # Convert to NodeWithScore format if needed
        if not text_chunks or not isinstance(text_chunks[0], NodeWithScore):
            nodes = [
                NodeWithScore(node=TextNode(text=chunk), score=1.0)
                for chunk in text_chunks
            ]
        else:
            nodes = text_chunks
            
        # Register nodes and get response
        self.citation_tracker.register_source_nodes(nodes)
        
        # Use the base synthesizer
        response = self.base_synthesizer.get_response(query_str, text_chunks, **kwargs)
        
        # Add citations
        processed_response = self.citation_tracker.process_response(response)
        return processed_response
        
    def synthesize(
        self,
        query: Union[str, QueryBundle],
        nodes: List[NodeWithScore],
        **kwargs: Any
    ) -> Response:
        """
        Synthesize a response with citation tracking.
        
        Args:
            query: Query string or bundle
            nodes: Source nodes
            **kwargs: Additional arguments for the base synthesizer
            
        Returns:
            Response with citations
        """
        # Register the source nodes with the citation tracker
        self.citation_tracker.register_source_nodes(nodes)
        
        # Use the base synthesizer to generate the response
        response = self.base_synthesizer.synthesize(query, nodes, **kwargs)
        
        # Process the response to add citations
        processed_response = self.citation_tracker.process_response(response)
        
        return processed_response
