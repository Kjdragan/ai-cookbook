"""
Search providers package for the Docling search module.
Exports available search providers.
"""

from .lancedb_provider import LanceDBSearchProvider

# Import LlamaIndex provider if dependencies are available
try:
    from .llamaindex import LlamaIndexProvider
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
