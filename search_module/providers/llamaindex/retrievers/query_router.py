"""
Query router for LlamaIndex retrieval methods.

This module provides functionality to route queries to the most appropriate
retrieval method based on query characteristics and patterns.
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries that can be detected and routed."""
    
    KEYWORD = "keyword"  # Simple keyword-based queries
    SEMANTIC = "semantic"  # Semantic/conceptual queries
    FACTOID = "factoid"  # Fact-based questions (who, what, when, where)
    COMPLEX = "complex"  # Complex, multi-part questions
    UNKNOWN = "unknown"  # Unknown or unclassified query type


class RoutingStrategy(str, Enum):
    """Strategies for routing queries to retrievers."""
    
    RULE_BASED = "rule_based"  # Use heuristic rules to determine routing
    LLM_BASED = "llm_based"  # Use an LLM to classify and route queries
    HYBRID = "hybrid"  # Combine rule-based and LLM-based approaches


@dataclass
class QueryRouterConfig:
    """Configuration for QueryRouter."""
    
    # Strategy to use for routing
    strategy: RoutingStrategy = RoutingStrategy.HYBRID
    
    # LLM configuration for LLM-based routing
    llm_model_name: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 200
    
    # Whether to include query analysis in metadata
    include_analysis_metadata: bool = True
    
    # Confidence threshold for LLM classification (0.0-1.0)
    confidence_threshold: float = 0.7
    
    # Maximum query length for rule-based analysis
    max_rule_query_length: int = 1000
    
    # Keywords that suggest specific query types (for rule-based routing)
    keyword_indicators: Dict[str, List[str]] = field(default_factory=lambda: {
        QueryType.KEYWORD: [
            "find", "search", "list", "show", "get"
        ],
        QueryType.SEMANTIC: [
            "similar", "like", "related", "about", "concept", 
            "understand", "explain", "describe"
        ],
        QueryType.FACTOID: [
            "who", "what", "when", "where", "why", "how many", 
            "how much", "which", "did"
        ],
        QueryType.COMPLEX: [
            "compare", "contrast", "difference", "analyze", "evaluate",
            "relationship", "impact", "effect", "cause", "synthesize", 
            "integrate", "and", "both", "multiple"
        ]
    })


class QueryRouter:
    """
    Routes queries to the most appropriate retriever based on query characteristics.
    
    The router analyzes incoming queries and directs them to the retriever
    that is most likely to produce the best results for that query type.
    """
    
    def __init__(
        self,
        retrievers: Dict[str, BaseRetriever],
        default_retriever_name: str,
        config: Optional[QueryRouterConfig] = None,
        llm: Optional[LLM] = None,
    ):
        """
        Initialize the query router.
        
        Args:
            retrievers: Dict mapping retriever names to retriever instances
            default_retriever_name: Name of the default retriever to use
                                   when no specific routing can be determined
            config: Router configuration
            llm: LLM instance for LLM-based routing
        """
        self.retrievers = retrievers
        self.config = config or QueryRouterConfig()
        
        # Validate default retriever
        if default_retriever_name not in retrievers:
            raise ValueError(
                f"Default retriever '{default_retriever_name}' not in provided retrievers: "
                f"{list(retrievers.keys())}"
            )
        self.default_retriever_name = default_retriever_name
        
        # Initialize LLM if needed
        if self.config.strategy in [RoutingStrategy.LLM_BASED, RoutingStrategy.HYBRID]:
            self.llm = llm or OpenAI(
                model=self.config.llm_model_name,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        else:
            self.llm = None
            
        # Create mapping from query types to retriever names
        self.query_type_map: Dict[QueryType, str] = {}
        
        # Default mappings if no specific ones are provided
        # These can be overridden using the set_query_type_mapping method
        if retrievers and len(retrievers) > 1:
            retriever_names = list(retrievers.keys())
            self.query_type_map = {
                QueryType.KEYWORD: retriever_names[0],
                QueryType.SEMANTIC: retriever_names[0],
                QueryType.FACTOID: retriever_names[0],
                QueryType.COMPLEX: retriever_names[0],
                QueryType.UNKNOWN: self.default_retriever_name
            }
    
    def set_query_type_mapping(self, query_type: QueryType, retriever_name: str) -> None:
        """
        Set the mapping from a query type to a retriever name.
        
        Args:
            query_type: The query type to map
            retriever_name: The name of the retriever to use for this query type
        """
        if retriever_name not in self.retrievers:
            raise ValueError(
                f"Retriever '{retriever_name}' not in provided retrievers: "
                f"{list(self.retrievers.keys())}"
            )
        
        self.query_type_map[query_type] = retriever_name
    
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes using the appropriate retriever for the query.
        
        Args:
            query_bundle: The query bundle
            
        Returns:
            List of retrieved nodes with scores
        """
        # Analyze the query to determine the best retriever
        query_analysis = self._analyze_query(query_bundle)
        query_type = query_analysis["query_type"]
        confidence = query_analysis["confidence"]
        
        # Determine which retriever to use
        retriever_name = self.query_type_map.get(
            query_type, self.default_retriever_name
        )
        retriever = self.retrievers[retriever_name]
        
        logger.info(
            f"Routing query '{query_bundle.query_str[:50]}...' to {retriever_name} "
            f"(detected as {query_type}, confidence: {confidence:.2f})"
        )
        
        # Add metadata to the query bundle if applicable
        if self.config.include_analysis_metadata:
            if query_bundle.metadata is None:
                query_bundle.metadata = {}
            query_bundle.metadata["query_router"] = query_analysis
        
        # Retrieve using the selected retriever
        return retriever.retrieve(query_bundle)
    
    def _analyze_query(self, query_bundle: QueryBundle) -> Dict[str, Any]:
        """
        Analyze query to determine its type and characteristics.
        
        Args:
            query_bundle: Query bundle to analyze
            
        Returns:
            Dict containing analysis results
        """
        query = query_bundle.query_str
        
        # Apply the selected strategy
        if self.config.strategy == RoutingStrategy.RULE_BASED:
            return self._rule_based_analysis(query)
        elif self.config.strategy == RoutingStrategy.LLM_BASED:
            return self._llm_based_analysis(query)
        elif self.config.strategy == RoutingStrategy.HYBRID:
            return self._hybrid_analysis(query)
        else:
            # Default to rule-based
            return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> Dict[str, Any]:
        """
        Analyze query using rule-based heuristics.
        
        Args:
            query: Query string
            
        Returns:
            Dict containing analysis results
        """
        # Trim query if too long for efficient analysis
        query_for_analysis = query[:self.config.max_rule_query_length].lower()
        
        # Check for multiple question marks (complex query indicator)
        question_mark_count = query_for_analysis.count("?")
        
        # Count words as a complexity metric
        word_count = len(query_for_analysis.split())
        
        # Check for query type indicators
        query_type_scores = {qtype: 0.0 for qtype in QueryType}
        
        # Remove common words that might skew the analysis
        stopwords = ["the", "a", "an", "in", "on", "at", "to", "for"]
        query_words = query_for_analysis.split()
        
        # Check against keyword indicators
        for query_type, indicators in self.config.keyword_indicators.items():
            for indicator in indicators:
                if indicator.lower() in query_for_analysis:
                    query_type_scores[query_type] += 1.0
                    
                    # Higher score for indicators at the beginning of the query
                    if query_for_analysis.startswith(indicator.lower()):
                        query_type_scores[query_type] += 1.0
        
        # Apply additional heuristics
        
        # Complex queries tend to be longer
        if word_count > 15:
            query_type_scores[QueryType.COMPLEX] += (word_count - 15) * 0.1
            
        # Multiple question marks often indicate complex queries
        if question_mark_count > 1:
            query_type_scores[QueryType.COMPLEX] += question_mark_count * 0.5
            
        # Short queries with specific terms are often keyword searches
        if word_count < 5 and query_type_scores[QueryType.KEYWORD] == 0:
            query_type_scores[QueryType.KEYWORD] += 1.0
            
        # Very short queries (1-3 words) are typically keywords
        if word_count <= 3:
            query_type_scores[QueryType.KEYWORD] += 2.0
            
        # Find the query type with the highest score
        best_query_type = max(
            query_type_scores.items(), 
            key=lambda x: x[1]
        )[0]
        
        # If no clear type, default to unknown
        if query_type_scores[best_query_type] == 0:
            best_query_type = QueryType.UNKNOWN
            confidence = 0.0
        else:
            # Calculate confidence as normalized score
            total_score = sum(query_type_scores.values())
            confidence = query_type_scores[best_query_type] / total_score if total_score > 0 else 0.0
            
            # Cap confidence for rule-based at 0.9
            confidence = min(confidence, 0.9)
        
        return {
            "query_type": best_query_type,
            "confidence": confidence,
            "method": "rule_based",
            "scores": query_type_scores
        }
    
    def _llm_based_analysis(self, query: str) -> Dict[str, Any]:
        """
        Analyze query using an LLM for classification.
        
        Args:
            query: Query string
            
        Returns:
            Dict containing analysis results
        """
        if not self.llm:
            logger.warning("LLM-based analysis requested but no LLM provided, using rule-based")
            return self._rule_based_analysis(query)
            
        prompt = f"""Classify the following query into exactly one of these categories:
1. KEYWORD: Simple keyword-based queries (e.g., "find documents about climate change")
2. SEMANTIC: Conceptual/semantic queries requiring understanding (e.g., "explain the relationship between economics and politics")
3. FACTOID: Fact-based questions (e.g., "who invented the telephone?", "when was Python created?")
4. COMPLEX: Multi-part or analytical questions (e.g., "compare the economic policies of the US and China and their impact on global trade")
5. UNKNOWN: If the query doesn't clearly fit any category

Query: {query}

Respond in the following JSON format:
{{
  "query_type": "KEYWORD|SEMANTIC|FACTOID|COMPLEX|UNKNOWN",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "brief explanation for the classification"
}}
"""
        try:
            # Get LLM response
            llm_response = self.llm.complete(prompt)
            response_text = llm_response.text
            
            # Extract JSON part from response
            json_match = re.search(r"({.*})", response_text, re.DOTALL)
            
            if json_match:
                import json
                json_str = json_match.group(1)
                result = json.loads(json_str)
                
                # Normalize the query type to enum
                query_type_str = result.get("query_type", "UNKNOWN").upper()
                
                # Map the string to the enum
                query_type_map = {qt.name: qt for qt in QueryType}
                query_type = query_type_map.get(query_type_str, QueryType.UNKNOWN)
                
                # Get confidence
                confidence = float(result.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Ensure in range [0,1]
                
                return {
                    "query_type": query_type,
                    "confidence": confidence,
                    "method": "llm_based",
                    "reasoning": result.get("reasoning", "")
                }
            else:
                logger.warning(f"Could not parse JSON from LLM response: {response_text}")
                return {
                    "query_type": QueryType.UNKNOWN,
                    "confidence": 0.0,
                    "method": "llm_based_failed",
                    "error": "Could not parse LLM response"
                }
                
        except Exception as e:
            logger.error(f"Error in LLM-based analysis: {e}")
            # Fall back to rule-based analysis
            rule_result = self._rule_based_analysis(query)
            rule_result["method"] = "llm_failed_rule_fallback"
            rule_result["error"] = str(e)
            return rule_result
    
    def _hybrid_analysis(self, query: str) -> Dict[str, Any]:
        """
        Use both rule-based and LLM-based analysis and combine results.
        
        Args:
            query: Query string
            
        Returns:
            Dict containing analysis results
        """
        # Get both analyses
        rule_result = self._rule_based_analysis(query)
        llm_result = self._llm_based_analysis(query)
        
        # If LLM confidence is high enough, prefer it
        if (llm_result["confidence"] >= self.config.confidence_threshold and 
            llm_result["query_type"] != QueryType.UNKNOWN):
            return {
                "query_type": llm_result["query_type"],
                "confidence": llm_result["confidence"],
                "method": "hybrid_llm_preferred",
                "rule_analysis": rule_result,
                "llm_analysis": llm_result
            }
        
        # If rule confidence is higher, prefer it
        if rule_result["confidence"] > llm_result["confidence"]:
            return {
                "query_type": rule_result["query_type"],
                "confidence": rule_result["confidence"],
                "method": "hybrid_rule_preferred",
                "rule_analysis": rule_result,
                "llm_analysis": llm_result
            }
        
        # Otherwise, use LLM result
        return {
            "query_type": llm_result["query_type"],
            "confidence": llm_result["confidence"],
            "method": "hybrid_llm_default",
            "rule_analysis": rule_result,
            "llm_analysis": llm_result
        }
    
    def route_query(self, query: Union[str, QueryBundle]) -> Tuple[str, Any]:
        """
        Route a query to the most appropriate retriever and retrieve results.
        
        Args:
            query: Query string or QueryBundle
            
        Returns:
            Tuple of (retriever_name, retrieval_result)
        """
        # Convert string query to QueryBundle if needed
        if isinstance(query, str):
            query_bundle = QueryBundle(query_str=query)
        else:
            query_bundle = query
            
        # Analyze the query to determine the best retriever
        query_analysis = self._analyze_query(query_bundle)
        query_type = query_analysis["query_type"]
        confidence = query_analysis["confidence"]
        
        # Determine which retriever to use
        retriever_name = self.query_type_map.get(query_type, self.default_retriever_name)
        retriever = self.retrievers[retriever_name]
        
        logger.info(
            f"Routing query '{query_bundle.query_str[:50]}...' to {retriever_name} "
            f"(detected as {query_type}, confidence: {confidence:.2f})"
        )
        
        # We can't add metadata directly to QueryBundle in 0.12+
        # Only store analysis in our log/class state if needed
        
        # Retrieve using the selected retriever
        retrieval_result = retriever.retrieve(query_bundle)
        
        # Create a simple result object with the nodes
        class RetrievalResult:
            def __init__(self, nodes):
                self.nodes = nodes
                
        result = RetrievalResult(retrieval_result)
        
        return retriever_name, result
