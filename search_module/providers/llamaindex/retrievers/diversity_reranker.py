"""
Diversity-aware reranking implementation for LlamaIndex.

This module provides tools to rerank retrieval results to improve diversity
and reduce redundancy while maintaining relevance.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)


@dataclass
class DiversityRerankConfig:
    """Configuration for diversity reranking."""
    
    # Tradeoff between relevance (1.0) and diversity (0.0)
    # Higher values prioritize relevance, lower values prioritize diversity
    lambda_factor: float = 0.5
    
    # Maximum number of candidates to consider
    max_candidates: int = 100
    
    # Whether to use MMR (Maximum Marginal Relevance) or DPP (Determinantal Point Process)
    method: str = "mmr"  # "mmr" or "dpp"
    
    # Similarity metric to use for diversity calculation
    similarity_metric: str = "cosine"  # "cosine", "dot_product", "euclidean"
    
    # Parameters for specific methods
    mmr_diversity_weight: float = 0.3  # Diversity weight for MMR
    
    # Whether to calculate diversity based on content or embeddings
    use_embeddings: bool = True
    
    # Optional config for embedding model (if not provided through reranker)
    embedding_model_name: str = "text-embedding-3-large"
    embedding_dim: int = 3072


class DiversityReranker:
    """
    Reranker that promotes diversity in retrieved results.
    
    This reranker uses techniques like Maximum Marginal Relevance (MMR) 
    or Determinantal Point Process (DPP) to select diverse subsets of
    retrieved documents while maintaining relevance.
    """
    
    def __init__(
        self,
        config: Optional[DiversityRerankConfig] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        """
        Initialize the diversity reranker.
        
        Args:
            config: Reranking configuration
            embed_model: Embedding model to use for embedding calculation
        """
        self.config = config or DiversityRerankConfig()
        
        # Initialize embedding model if needed
        if self.config.use_embeddings:
            self.embed_model = embed_model or OpenAIEmbedding(
                model_name=self.config.embedding_model_name,
                dimensions=self.config.embedding_dim
            )
        else:
            self.embed_model = None
            
        # Validate configuration
        valid_methods = ["mmr", "dpp"]
        if self.config.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
            
        valid_metrics = ["cosine", "dot_product", "euclidean"]
        if self.config.similarity_metric not in valid_metrics:
            raise ValueError(f"Similarity metric must be one of {valid_metrics}")
    
    def rerank(
        self, 
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        top_n: Optional[int] = None
    ) -> List[NodeWithScore]:
        """
        Rerank nodes to improve diversity.
        
        Args:
            nodes: Initial ranked list of nodes
            query_bundle: Optional query bundle for relevance calculation
            top_n: Number of results to return (defaults to all)
            
        Returns:
            Reranked list of nodes
        """
        if not nodes:
            return []
            
        # Limit number of candidates for efficiency
        nodes = nodes[:self.config.max_candidates]
        
        # Set default top_n if not provided
        if top_n is None:
            top_n = len(nodes)
            
        # Choose reranking method
        if self.config.method == "mmr":
            reranked = self._mmr_rerank(nodes, query_bundle, top_n)
        elif self.config.method == "dpp":
            reranked = self._dpp_rerank(nodes, query_bundle, top_n)
        else:
            # Fallback to original ranking
            reranked = nodes[:top_n]
            
        return reranked
    
    def _mmr_rerank(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle],
        top_n: int
    ) -> List[NodeWithScore]:
        """
        Rerank using Maximum Marginal Relevance (MMR).
        
        MMR aims to reduce redundancy while maintaining relevance by selecting
        documents that maximize a weighted combination of relevance and diversity.
        
        Args:
            nodes: Initial ranked list of nodes
            query_bundle: Query bundle for relevance calculation
            top_n: Number of results to return
            
        Returns:
            Reranked list of nodes
        """
        logger.debug(f"Reranking {len(nodes)} nodes using MMR, targeting {top_n} results")
        
        if len(nodes) <= 1:
            return nodes
            
        if top_n > len(nodes):
            top_n = len(nodes)
            
        # Calculate embeddings for nodes if using embeddings
        if self.config.use_embeddings and self.embed_model:
            node_texts = [node.node.text for node in nodes]
            embeddings = self._get_embeddings(node_texts)
            
            # Also get query embedding if available
            query_embedding = None
            if query_bundle:
                query_text = query_bundle.query_str
                query_embedding = self._get_embeddings([query_text])[0]
        else:
            # Use text-based similarity
            embeddings = None
            query_embedding = None
            
        # Initial relevance scores from the retriever (normalized to [0, 1])
        relevance_scores = np.array([node.score or 0.0 for node in nodes])
        
        # Selected indices and remaining candidates
        selected_indices = []
        remaining_indices = list(range(len(nodes)))
        
        # Select nodes iteratively using MMR
        for _ in range(min(top_n, len(nodes))):
            if not remaining_indices:
                break
                
            # If this is the first selection or no embeddings, choose based on relevance
            if not selected_indices or embeddings is None:
                # Select the most relevant node
                next_idx = max(remaining_indices, key=lambda i: relevance_scores[i])
            else:
                # Calculate MMR scores
                mmr_scores = []
                
                lambda_val = self.config.lambda_factor
                diversity_weight = self.config.mmr_diversity_weight
                
                for i in remaining_indices:
                    # Relevance component (lambda * rel_score)
                    relevance_component = lambda_val * relevance_scores[i]
                    
                    # Diversity component ((1-lambda) * min_dist_to_selected)
                    max_similarity = 0.0
                    
                    for j in selected_indices:
                        similarity = self._calculate_similarity(
                            embeddings[i], embeddings[j], 
                            self.config.similarity_metric
                        )
                        max_similarity = max(max_similarity, similarity)
                    
                    diversity_component = (1.0 - lambda_val) * diversity_weight * (1.0 - max_similarity)
                    
                    # Combined MMR score
                    mmr_score = relevance_component + diversity_component
                    mmr_scores.append((i, mmr_score))
                
                # Select the node with the highest MMR score
                next_idx = max(mmr_scores, key=lambda x: x[1])[0]
            
            # Move the selected index from remaining to selected
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        # Create the reranked list
        reranked_nodes = [nodes[i] for i in selected_indices]
        
        return reranked_nodes
    
    def _dpp_rerank(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle],
        top_n: int
    ) -> List[NodeWithScore]:
        """
        Rerank using Determinantal Point Process (DPP).
        
        DPP is a probabilistic model that captures negative correlations and
        is used to select diverse subsets from a ground set.
        
        Args:
            nodes: Initial ranked list of nodes
            query_bundle: Query bundle for relevance calculation
            top_n: Number of results to return
            
        Returns:
            Reranked list of nodes
        """
        logger.debug(f"Reranking {len(nodes)} nodes using DPP, targeting {top_n} results")
        
        if len(nodes) <= 1:
            return nodes
            
        if top_n > len(nodes):
            top_n = len(nodes)
            
        # Calculate embeddings for nodes if using embeddings
        if self.config.use_embeddings and self.embed_model:
            node_texts = [node.node.text for node in nodes]
            embeddings = self._get_embeddings(node_texts)
        else:
            # Fall back to MMR if embeddings are not available
            logger.warning("DPP requires embeddings, falling back to MMR")
            return self._mmr_rerank(nodes, query_bundle, top_n)
            
        # Get relevance scores
        relevance_scores = np.array([node.score or 0.0 for node in nodes])
        
        # Create quality vector (diagonal of kernel matrix)
        quality = relevance_scores
        
        # Create similarity matrix
        n = len(nodes)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self._calculate_similarity(
                        embeddings[i], embeddings[j], 
                        self.config.similarity_metric
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Create kernel matrix L = diag(quality) * similarity * diag(quality)
        kernel = np.outer(quality, quality) * similarity_matrix
        
        # Apply lambda factor to balance diversity and relevance
        kernel = self.config.lambda_factor * np.diag(np.diag(kernel)) + \
                 (1 - self.config.lambda_factor) * kernel
        
        # Greedy DPP algorithm for subset selection
        selected_indices = []
        remaining_indices = list(range(n))
        
        # Select the most relevant item first
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select subsequent items
        for _ in range(min(top_n - 1, n - 1)):
            if not remaining_indices:
                break
                
            # Calculate marginal gains for each remaining item
            gains = []
            
            for idx in remaining_indices:
                # Create a new candidate set including this item
                candidate_set = selected_indices + [idx]
                
                # Calculate determinant for this subset (proportional to diversity)
                sub_kernel = kernel[np.ix_(candidate_set, candidate_set)]
                
                # Use the determinant as the diversity measure
                try:
                    det = np.linalg.det(sub_kernel)
                    if det < 0:  # Handle numerical issues
                        det = 0
                except:
                    det = 0
                
                gains.append((idx, det))
            
            # Select the item with the highest determinant
            next_idx = max(gains, key=lambda x: x[1])[0]
            
            # Add to selected and remove from remaining
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        # Create the reranked list
        reranked_nodes = [nodes[i] for i in selected_indices]
        
        return reranked_nodes
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        try:
            # Handle empty texts
            texts = [t if t.strip() else "empty" for t in texts]
            
            # Get embeddings from the model
            embeddings = self.embed_model.get_text_embedding_batch(texts)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.config.embedding_dim] * len(texts)
    
    def _calculate_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
        metric: str
    ) -> float:
        """
        Calculate similarity between two vectors using the specified metric.
        
        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric to use
            
        Returns:
            Similarity score (higher means more similar)
        """
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        if metric == "cosine":
            # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return np.dot(a, b) / (norm_a * norm_b)
            
        elif metric == "dot_product":
            # Dot product similarity
            return np.dot(a, b)
            
        elif metric == "euclidean":
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(a - b)
            # Convert to similarity (higher means more similar)
            return 1.0 / (1.0 + distance)
            
        else:
            # Default to cosine
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return np.dot(a, b) / (norm_a * norm_b)
