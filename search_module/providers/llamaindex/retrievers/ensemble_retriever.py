"""
Ensemble retriever implementation that combines results from multiple retrievers.

This module provides a flexible framework for combining results from different
retrieval strategies, with configurable weighting and result normalization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import numpy as np
from collections import defaultdict

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines results from multiple retrievers.
    
    This retriever takes a list of retrievers, runs all of them on the input query,
    and combines the results using configurable weighting and normalization strategies.
    """
    
    def __init__(
        self,
        retrievers: Union[List[BaseRetriever], Dict[str, BaseRetriever]],
        weights: Optional[Union[List[float], Dict[str, float]]] = None,
        normalization: str = "min_max",
        normalization_method: Optional[str] = None,  # Alias for normalization
        combination_method: str = "weighted_sum",
        deduplication_threshold: float = 0.9,
        top_k: int = 10,
    ):
        """
        Initialize the ensemble retriever.
        
        Args:
            retrievers: List or Dict of retrievers to ensemble
            weights: Optional weights for each retriever (defaults to equal weights)
            normalization: Score normalization method ("min_max", "rank", "softmax", "none")
            normalization_method: Alias for normalization
            combination_method: Method for combining scores ("weighted_sum", "max", "reciprocal_rank")
            deduplication_threshold: Similarity threshold for deduplication (0.0-1.0)
            top_k: Number of results to return
        """
        super().__init__()
        
        # Handle dict or list input for retrievers
        if isinstance(retrievers, dict):
            self.retriever_names = list(retrievers.keys())
            self.retrievers = list(retrievers.values())
        else:
            self.retriever_names = [f"retriever_{i}" for i in range(len(retrievers))]
            self.retrievers = retrievers
            
        # Handle dict or list input for weights
        if weights is None:
            self.weights = [1.0] * len(self.retrievers)
        elif isinstance(weights, dict):
            if not isinstance(retrievers, dict):
                raise ValueError("If weights is a dict, retrievers must also be a dict")
            self.weights = [weights.get(name, 1.0) for name in self.retriever_names]
        else:
            self.weights = weights
            
        if len(self.weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")
            
        # Handle normalization method parameter alias
        self.normalization = normalization_method if normalization_method is not None else normalization
        
        self.combination_method = combination_method
        self.deduplication_threshold = deduplication_threshold
        self.top_k = top_k
        
        # Validate parameters
        valid_norms = ["min_max", "minmax", "rank", "softmax", "none"]
        if self.normalization not in valid_norms:
            raise ValueError(f"Normalization must be one of {valid_norms}")
            
        valid_combos = ["weighted_sum", "max", "reciprocal_rank"]
        if self.combination_method not in valid_combos:
            raise ValueError(f"Combination method must be one of {valid_combos}")
            
        logger.info(f"Initialized EnsembleRetriever with {len(self.retrievers)} retrievers")
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve nodes from all retrievers and combine results.
        
        Args:
            query_bundle: Query bundle containing the query string and additional info
            
        Returns:
            List of nodes with combined scores
        """
        logger.info(f"Running ensemble retrieval with {len(self.retrievers)} retrievers")
        
        # Run all retrievers and collect results
        all_nodes = []
        retriever_results = []
        
        for i, retriever in enumerate(self.retrievers):
            retriever_name = self.retriever_names[i] if hasattr(self, 'retriever_names') else f"retriever_{i}"
            logger.debug(f"Running retriever {i+1}/{len(self.retrievers)}: {retriever_name}")
            try:
                nodes = retriever.retrieve(query_bundle)
                retriever_results.append(nodes)
                logger.debug(f"Retriever {retriever_name} returned {len(nodes)} results")
            except Exception as e:
                logger.error(f"Error in retriever {retriever_name}: {e}")
                retriever_results.append([])
        
        # Normalize scores within each retriever
        normalized_results = self._normalize_scores(retriever_results)
        
        # Combine results from all retrievers
        combined_nodes = self._combine_results(normalized_results)
        
        # Deduplicate results
        deduplicated_nodes = self._deduplicate_results(combined_nodes)
        
        # Return top k results
        return deduplicated_nodes[:self.top_k]
    
    def _normalize_scores(self, retriever_results: List[List[NodeWithScore]]) -> List[List[NodeWithScore]]:
        """
        Normalize scores within each retriever's results.
        
        Args:
            retriever_results: List of results from each retriever
            
        Returns:
            List of results with normalized scores
        """
        normalized_results = []
        
        for nodes in retriever_results:
            if not nodes:
                normalized_results.append([])
                continue
                
            # Extract scores
            scores = np.array([node.score or 0.0 for node in nodes])
            
            # Apply normalization
            if self.normalization == "min_max" and len(scores) > 1:
                min_score, max_score = np.min(scores), np.max(scores)
                if max_score > min_score:  # Avoid division by zero
                    normalized_scores = (scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(scores)
                    
            elif self.normalization == "rank":
                # Higher rank (1st, 2nd, etc.) gets higher score
                ranks = np.argsort(np.argsort(-scores))  # Double argsort for ranking
                normalized_scores = 1.0 - (ranks / max(1, len(ranks) - 1))
                
            elif self.normalization == "softmax":
                exp_scores = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
                normalized_scores = exp_scores / np.sum(exp_scores)
                
            else:  # "none" or fallback
                normalized_scores = scores
                
            # Create new nodes with normalized scores
            normalized_nodes = []
            for i, node in enumerate(nodes):
                normalized_node = NodeWithScore(
                    node=node.node,
                    score=float(normalized_scores[i])
                )
                normalized_nodes.append(normalized_node)
                
            normalized_results.append(normalized_nodes)
            
        return normalized_results
    
    def _combine_results(self, normalized_results: List[List[NodeWithScore]]) -> List[NodeWithScore]:
        """
        Combine results from multiple retrievers using the specified method.
        
        Args:
            normalized_results: List of normalized results from each retriever
            
        Returns:
            Combined list of nodes with scores
        """
        # Create a dictionary to track nodes and their scores from different retrievers
        node_scores = defaultdict(list)
        node_weights = defaultdict(list)
        node_objects = {}
        
        # Collect scores for each unique node
        for retriever_idx, nodes in enumerate(normalized_results):
            weight = self.weights[retriever_idx]
            
            for node in nodes:
                # Use node ID or content as the key
                node_id = self._get_node_id(node)
                
                # Store score and retriever weight
                node_scores[node_id].append(node.score or 0.0)
                node_weights[node_id].append(weight)
                
                # Keep track of the original node object
                if node_id not in node_objects:
                    node_objects[node_id] = node
        
        # Combine scores for each unique node
        combined_nodes = []
        
        for node_id, scores in node_scores.items():
            weights = node_weights[node_id]
            original_node = node_objects[node_id]
            
            # Apply combination method
            if self.combination_method == "weighted_sum":
                # Weighted sum of scores
                combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                
            elif self.combination_method == "max":
                # Maximum score across retrievers
                combined_score = max(scores)
                
            elif self.combination_method == "reciprocal_rank":
                # Reciprocal rank fusion
                k = 60  # Constant to prevent extremely small values from dominating
                combined_score = sum(1.0 / (k + (1.0 - s) * 100) for s in scores)
                
            else:
                # Default to weighted sum
                combined_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            
            # Create new node with combined score
            combined_node = NodeWithScore(
                node=original_node.node,
                score=combined_score
            )
            combined_nodes.append(combined_node)
        
        # Sort by score (descending)
        return sorted(combined_nodes, key=lambda x: x.score or 0.0, reverse=True)
    
    def _deduplicate_results(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Remove duplicate results based on similarity threshold.
        
        Args:
            nodes: List of nodes to deduplicate
            
        Returns:
            Deduplicated list of nodes
        """
        if not nodes:
            return []
            
        # Simple deduplication based on node ID
        seen_ids = set()
        deduplicated = []
        
        for node in nodes:
            node_id = self._get_node_id(node)
            
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                deduplicated.append(node)
        
        return deduplicated
    
    def _get_node_id(self, node: NodeWithScore) -> str:
        """
        Get a unique identifier for a node.
        
        Args:
            node: Node to get ID for
            
        Returns:
            Unique identifier string
        """
        # Try to get node ID from metadata
        metadata = node.node.metadata or {}
        
        if "id" in metadata:
            return str(metadata["id"])
        elif "doc_id" in metadata:
            return str(metadata["doc_id"])
        elif "document_id" in metadata:
            return str(metadata["document_id"])
        
        # Fall back to text content hash
        return str(hash(node.node.text))
    
    def __str__(self) -> str:
        """Return string representation of the retriever."""
        retriever_names = [type(r).__name__ for r in self.retrievers]
        return f"EnsembleRetriever(retrievers={retriever_names}, weights={self.weights})"
