"""
Simple test script to verify the LlamaIndex integration.
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_llamaindex")

# Load environment variables
load_dotenv()

try:
    # Test imports
    from llama_index.core.llms import LLM
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.lancedb import LanceDBVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    
    logger.info("Successfully imported LlamaIndex packages!")
    
    # Test LLM
    try:
        llm = OpenAI(model="gpt-4o")
        response = llm.complete("Hello, LlamaIndex!")
        logger.info(f"LLM response: {response}")
    except Exception as e:
        logger.error(f"Error using OpenAI LLM: {e}")
    
    # Log environment details
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    
except ImportError as e:
    logger.error(f"Import error: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
