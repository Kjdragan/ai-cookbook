"""
Module for configuring different LLMs for LlamaIndex integration.
Provides a unified interface for using OpenAI and Deepseek models.
"""

import os
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Import Llama Index components
from llama_index_core.llms import LLM
from llama_index_llms_openai import OpenAI

class LLMFactory:
    """Factory class for creating LLM instances with different providers."""
    
    @staticmethod
    def create_llm(
        provider: str = "openai", 
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None
    ) -> LLM:
        """
        Create an LLM instance based on the specified provider.
        
        Args:
            provider: The LLM provider (openai or deepseek)
            model_name: The model name to use
            temperature: The temperature for generation
            max_tokens: Maximum tokens to generate
            additional_kwargs: Any additional parameters to pass to the LLM
            
        Returns:
            An LLM instance compatible with LlamaIndex
        """
        provider = provider.lower()
        additional_kwargs = additional_kwargs or {}
        
        # Set up OpenAI-compatible client for either provider
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment variables")
                
            return OpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                additional_kwargs=additional_kwargs
            )
            
        elif provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                logger.warning("DEEPSEEK_API_KEY not found in environment variables")
                
            # Use OpenAI client with custom base URL for Deepseek
            return OpenAI(
                model=model_name,
                api_key=api_key,
                api_base="https://api.deepseek.com",
                api_type="openai",
                temperature=temperature,
                max_tokens=max_tokens,
                additional_kwargs=additional_kwargs
            )
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
    @staticmethod
    def get_available_models(provider: str = "openai") -> list:
        """
        Get a list of recommended models for the specified provider.
        
        Args:
            provider: The LLM provider (openai or deepseek)
            
        Returns:
            List of recommended model names
        """
        provider = provider.lower()
        
        if provider == "openai":
            return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
        elif provider == "deepseek":
            return ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]
        else:
            return []
