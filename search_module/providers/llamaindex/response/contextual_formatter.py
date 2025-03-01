"""
Contextual response formatter for LlamaIndex responses.

This module provides tools to format response content based on query type,
client needs, and content characteristics.
"""

import re
import logging
import json
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field

from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.core.response import Response
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


class ResponseFormat(str, Enum):
    """Output formats for responses."""
    
    TEXT = "text"  # Plain text
    MARKDOWN = "markdown"  # Markdown formatting
    HTML = "html"  # HTML formatting
    JSON = "json"  # JSON structure
    CSV = "csv"  # CSV formatting


class VerbosityLevel(str, Enum):
    """Verbosity levels for response formatting."""
    
    CONCISE = "concise"  # Brief answers
    NORMAL = "normal"  # Standard detail level
    DETAILED = "detailed"  # Comprehensive answers
    AUTO = "auto"  # Automatically determined based on query


@dataclass
class FormatterConfig:
    """Configuration for contextual response formatting."""
    
    # Default output format
    default_format: ResponseFormat = ResponseFormat.MARKDOWN
    
    # Default verbosity level
    default_verbosity: VerbosityLevel = VerbosityLevel.AUTO
    
    # Map of client types to preferred formats
    client_format_map: Dict[str, ResponseFormat] = field(default_factory=lambda: {
        "api": ResponseFormat.JSON,
        "ui": ResponseFormat.MARKDOWN,
        "cli": ResponseFormat.TEXT,
        "notebook": ResponseFormat.MARKDOWN
    })
    
    # Maximum response length (in characters)
    max_response_length: int = 10000
    
    # Whether to automatically determine the appropriate verbosity
    auto_verbosity: bool = True
    
    # LLM configuration for auto formatting
    llm_model_name: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    
    # Whether to include metadata in the response
    include_metadata: bool = True
    
    # Structure for JSON responses
    json_structure: Dict[str, Any] = field(default_factory=lambda: {
        "response": "",
        "sources": [],
        "metadata": {}
    })


class ContextualFormatter:
    """
    Formats responses based on query context and client needs.
    
    This formatter adjusts response formatting, verbosity, and structure
    based on query characteristics and client requirements.
    """
    
    def __init__(
        self,
        config: Optional[FormatterConfig] = None,
        llm: Optional[LLM] = None,
    ):
        """
        Initialize the contextual formatter.
        
        Args:
            config: Formatter configuration
            llm: LLM instance for auto formatting
        """
        self.config = config or FormatterConfig()
        
        # Initialize LLM if needed for auto formatting
        self.llm = None
        if self.config.auto_verbosity:
            self.llm = llm or OpenAI(
                model=self.config.llm_model_name,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
    
    def format_response(
        self,
        response: Response,
        query_bundle: Optional[QueryBundle] = None,
        output_format: Optional[ResponseFormat] = None,
        verbosity: Optional[VerbosityLevel] = None,
        client_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Format a response based on context and configuration.
        
        Args:
            response: The response to format
            query_bundle: Optional query bundle for context
            output_format: Desired output format, overrides defaults
            verbosity: Desired verbosity level, overrides defaults
            client_type: Client type (api, ui, cli, notebook)
            metadata: Additional metadata to include
            
        Returns:
            Formatted response (string or dict depending on format)
        """
        if not response:
            return "" if output_format != ResponseFormat.JSON else self._empty_json_response()
            
        # Determine output format
        format_to_use = self._determine_output_format(output_format, client_type)
        
        # Determine verbosity level
        verbosity_to_use = self._determine_verbosity(verbosity, query_bundle)
        
        # Get the response text
        response_text = response.response or ""
        
        # Format the response
        if format_to_use == ResponseFormat.JSON:
            return self._format_as_json(
                response, response_text, query_bundle, verbosity_to_use, metadata
            )
        elif format_to_use == ResponseFormat.MARKDOWN:
            return self._format_as_markdown(
                response, response_text, query_bundle, verbosity_to_use
            )
        elif format_to_use == ResponseFormat.HTML:
            return self._format_as_html(
                response, response_text, query_bundle, verbosity_to_use
            )
        elif format_to_use == ResponseFormat.CSV:
            return self._format_as_csv(
                response, response_text, query_bundle, verbosity_to_use
            )
        else:
            # Default to plain text
            return self._format_as_text(
                response, response_text, query_bundle, verbosity_to_use
            )
    
    def _determine_output_format(
        self,
        requested_format: Optional[ResponseFormat],
        client_type: Optional[str]
    ) -> ResponseFormat:
        """
        Determine the appropriate output format.
        
        Args:
            requested_format: Explicitly requested format
            client_type: Type of client
            
        Returns:
            Determined output format
        """
        # Explicit format takes precedence
        if requested_format is not None:
            return requested_format
            
        # Client type preference next
        if client_type and client_type in self.config.client_format_map:
            return self.config.client_format_map[client_type]
            
        # Default format
        return self.config.default_format
    
    def _determine_verbosity(
        self,
        requested_verbosity: Optional[VerbosityLevel],
        query_bundle: Optional[QueryBundle]
    ) -> VerbosityLevel:
        """
        Determine the appropriate verbosity level.
        
        Args:
            requested_verbosity: Explicitly requested verbosity
            query_bundle: Query bundle for context
            
        Returns:
            Determined verbosity level
        """
        # Explicit verbosity takes precedence
        if requested_verbosity is not None and requested_verbosity != VerbosityLevel.AUTO:
            return requested_verbosity
            
        # Auto-determine verbosity if enabled and we have a query
        if self.config.auto_verbosity and query_bundle and self.llm:
            return self._auto_determine_verbosity(query_bundle)
            
        # Default verbosity
        if self.config.default_verbosity != VerbosityLevel.AUTO:
            return self.config.default_verbosity
        else:
            return VerbosityLevel.NORMAL
    
    def _auto_determine_verbosity(self, query_bundle: QueryBundle) -> VerbosityLevel:
        """
        Automatically determine appropriate verbosity based on query.
        
        Args:
            query_bundle: Query bundle
            
        Returns:
            Determined verbosity level
        """
        if not self.llm:
            return VerbosityLevel.NORMAL
            
        query = query_bundle.query_str
        
        prompt = f"""Analyze the following query and determine the appropriate verbosity level for the response:

Query: {query}

Please determine if the query requires:
1. CONCISE response: Brief, direct answer (e.g., factual questions, simple requests)
2. NORMAL response: Standard level of detail (e.g., explanations, overviews)
3. DETAILED response: Comprehensive, in-depth answer (e.g., complex topics, analysis requests)

Respond with only one word: CONCISE, NORMAL, or DETAILED.
"""
        
        try:
            llm_response = self.llm.complete(prompt)
            response_text = llm_response.text.strip().upper()
            
            if "CONCISE" in response_text:
                return VerbosityLevel.CONCISE
            elif "DETAILED" in response_text:
                return VerbosityLevel.DETAILED
            else:
                return VerbosityLevel.NORMAL
                
        except Exception as e:
            logger.error(f"Error determining verbosity: {e}")
            return VerbosityLevel.NORMAL
    
    def _format_as_json(
        self,
        response: Response,
        response_text: str,
        query_bundle: Optional[QueryBundle],
        verbosity: VerbosityLevel,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format response as JSON.
        
        Args:
            response: Original response object
            response_text: Response text
            query_bundle: Query context
            verbosity: Verbosity level
            additional_metadata: Additional metadata to include
            
        Returns:
            JSON-formatted response as a dictionary
        """
        # Create a copy of the JSON structure
        result = dict(self.config.json_structure)
        
        # Add response text (truncated if needed)
        if len(response_text) > self.config.max_response_length:
            response_text = response_text[:self.config.max_response_length] + "..."
        result["response"] = response_text
        
        # Add sources if available
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for idx, node in enumerate(response.source_nodes):
                if not node.node:
                    continue
                    
                source = {
                    "id": idx + 1,
                    "text": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                    "score": node.score if node.score is not None else None,
                }
                
                # Add metadata if available
                if hasattr(node.node, "metadata") and node.node.metadata:
                    source["metadata"] = node.node.metadata
                    
                sources.append(source)
        
        result["sources"] = sources
        
        # Add metadata
        if self.config.include_metadata:
            # Start with existing metadata if any
            metadata = {}
            
            if hasattr(response, "metadata") and response.metadata:
                metadata.update(response.metadata)
                
            # Add additional metadata
            if additional_metadata:
                metadata.update(additional_metadata)
                
            # Add query info
            if query_bundle:
                metadata["query"] = query_bundle.query_str
                
                if hasattr(query_bundle, "metadata") and query_bundle.metadata:
                    metadata["query_metadata"] = query_bundle.metadata
            
            # Add verbosity
            metadata["verbosity"] = verbosity
            
            result["metadata"] = metadata
        
        return result
    
    def _format_as_markdown(
        self,
        response: Response,
        response_text: str,
        query_bundle: Optional[QueryBundle],
        verbosity: VerbosityLevel
    ) -> str:
        """
        Format response as Markdown.
        
        Args:
            response: Original response object
            response_text: Response text
            query_bundle: Query context
            verbosity: Verbosity level
            
        Returns:
            Markdown-formatted response text
        """
        # Truncate if needed
        if len(response_text) > self.config.max_response_length:
            response_text = response_text[:self.config.max_response_length] + "..."
            
        # For detailed verbosity, add extra information
        if verbosity == VerbosityLevel.DETAILED:
            # Add sources if available
            if hasattr(response, "source_nodes") and response.source_nodes:
                response_text += "\n\n### Sources\n"
                
                for idx, node in enumerate(response.source_nodes):
                    if not node.node:
                        continue
                        
                    # Extract source info
                    snippet = node.node.text[:150] + "..." if len(node.node.text) > 150 else node.node.text
                    score = f" (Score: {node.score:.2f})" if node.score is not None else ""
                    
                    # Format source reference
                    source_info = f"{idx+1}. {snippet}{score}"
                    response_text += f"\n{source_info}\n"
        
        return response_text
    
    def _format_as_html(
        self,
        response: Response,
        response_text: str,
        query_bundle: Optional[QueryBundle],
        verbosity: VerbosityLevel
    ) -> str:
        """
        Format response as HTML.
        
        Args:
            response: Original response object
            response_text: Response text
            query_bundle: Query context
            verbosity: Verbosity level
            
        Returns:
            HTML-formatted response text
        """
        # Truncate if needed
        if len(response_text) > self.config.max_response_length:
            response_text = response_text[:self.config.max_response_length] + "..."
            
        # Escape HTML special characters
        response_text = (
            response_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
        
        # Convert newlines to <br> tags
        response_text = response_text.replace("\n", "<br>")
        
        # Wrap in div
        html = f'<div class="response">{response_text}</div>'
        
        # For detailed verbosity, add extra information
        if verbosity == VerbosityLevel.DETAILED:
            # Add sources if available
            if hasattr(response, "source_nodes") and response.source_nodes:
                html += '<div class="sources"><h3>Sources</h3><ol>'
                
                for idx, node in enumerate(response.source_nodes):
                    if not node.node:
                        continue
                        
                    # Extract source info
                    snippet = node.node.text[:150] + "..." if len(node.node.text) > 150 else node.node.text
                    snippet = (
                        snippet
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                        .replace("'", "&#39;")
                    )
                    
                    score = f" (Score: {node.score:.2f})" if node.score is not None else ""
                    
                    # Format source reference
                    html += f'<li>{snippet}{score}</li>'
                
                html += '</ol></div>'
        
        return html
    
    def _format_as_text(
        self,
        response: Response,
        response_text: str,
        query_bundle: Optional[QueryBundle],
        verbosity: VerbosityLevel
    ) -> str:
        """
        Format response as plain text.
        
        Args:
            response: Original response object
            response_text: Response text
            query_bundle: Query context
            verbosity: Verbosity level
            
        Returns:
            Plain text formatted response
        """
        # Truncate if needed
        if len(response_text) > self.config.max_response_length:
            response_text = response_text[:self.config.max_response_length] + "..."
            
        # For detailed verbosity, add extra information
        if verbosity == VerbosityLevel.DETAILED:
            # Add sources if available
            if hasattr(response, "source_nodes") and response.source_nodes:
                response_text += "\n\nSources:\n"
                
                for idx, node in enumerate(response.source_nodes):
                    if not node.node:
                        continue
                        
                    # Extract source info
                    snippet = node.node.text[:150] + "..." if len(node.node.text) > 150 else node.node.text
                    score = f" (Score: {node.score:.2f})" if node.score is not None else ""
                    
                    # Format source reference
                    source_info = f"{idx+1}. {snippet}{score}"
                    response_text += f"\n{source_info}\n"
        
        return response_text
    
    def _format_as_csv(
        self,
        response: Response,
        response_text: str,
        query_bundle: Optional[QueryBundle],
        verbosity: VerbosityLevel
    ) -> str:
        """
        Format response as CSV.
        
        Args:
            response: Original response object
            response_text: Response text
            query_bundle: Query context
            verbosity: Verbosity level
            
        Returns:
            CSV-formatted response
        """
        # This is a simple implementation that assumes the response
        # contains CSV-formatted data or can be directly used as CSV
        
        # Check if the response already appears to be CSV formatted
        if "," in response_text and "\n" in response_text:
            # Already seems to be CSV format
            return response_text
            
        # If not CSV, try to interpret as a table and convert to CSV
        lines = response_text.split("\n")
        csv_lines = []
        
        # Process potential table structure
        for line in lines:
            # Remove leading/trailing whitespace and table formatting characters
            cleaned_line = line.strip().strip("|").strip()
            
            # Skip separator lines (e.g., |------|------|)
            if re.match(r'^[\-\+\s]*$', cleaned_line):
                continue
                
            # Split by pipe for markdown tables or multiple spaces for text tables
            if "|" in cleaned_line:
                cells = [cell.strip() for cell in cleaned_line.split("|")]
            else:
                cells = [cell.strip() for cell in re.split(r'\s{2,}', cleaned_line)]
                
            # Skip empty lines
            if not any(cells):
                continue
                
            # Escape quotes and join with commas
            csv_cells = [f'"{cell.replace("\"", "\"\"")}"' for cell in cells]
            csv_lines.append(",".join(csv_cells))
            
        # Return as CSV
        return "\n".join(csv_lines)
    
    def _empty_json_response(self) -> Dict[str, Any]:
        """
        Create an empty JSON response.
        
        Returns:
            Empty JSON response structure
        """
        result = dict(self.config.json_structure)
        result["response"] = ""
        result["sources"] = []
        result["metadata"] = {"status": "no_response_generated"}
        
        return result
