from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import logging
from datetime import datetime

# Import our search components
from .search_client import SearchClient
from .providers.base import SearchResult
from .providers.lancedb_provider import LanceDBSearchProvider


# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"search_api_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("search.api")


# Pydantic models for API requests and responses
class SearchRequest(BaseModel):
    query: str
    provider: Optional[str] = None
    limit: int = Field(5, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None


class DocumentSimilarityRequest(BaseModel):
    document_id: str
    provider: Optional[str] = None
    limit: int = Field(5, ge=1, le=50)


class SearchResultModel(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float
    provider: str
    vector_distance: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchResultModel]
    time_taken: float
    total_results: int
    query: str


# Initialize the app and search client
app = FastAPI(
    title="Document Search API",
    description="API for searching document chunks using vector embeddings",
    version="1.0.0"
)

# Initialize the search client as a global variable
search_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global search_client
    
    logger.info("Initializing search API")
    
    # Create a LanceDB provider
    try:
        lancedb_provider = LanceDBSearchProvider()
        
        # Initialize the search client with the LanceDB provider
        search_client = SearchClient(providers=[lancedb_provider])
        
        logger.info("Search API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize search API: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Document Search API",
        "version": "1.0.0",
        "providers": search_client.get_available_providers(),
        "default_provider": search_client.default_provider
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform a semantic search on document chunks.
    """
    try:
        import time
        start_time = time.time()
        
        # Execute the search
        results = search_client.search(
            query=request.query,
            provider=request.provider,
            limit=request.limit
        )
        
        # Convert results to Pydantic models
        result_models = [
            SearchResultModel(
                text=r.text,
                metadata=r.metadata,
                score=r.score,
                provider=r.provider,
                vector_distance=r.vector_distance
            )
            for r in results
        ]
        
        time_taken = time.time() - start_time
        
        return SearchResponse(
            results=result_models,
            time_taken=time_taken,
            total_results=len(results),
            query=request.query
        )
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/hybrid-search", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """
    Perform a hybrid search combining vector search with keyword filtering.
    """
    try:
        import time
        start_time = time.time()
        
        # Execute the hybrid search
        results = search_client.hybrid_search(
            query=request.query,
            provider=request.provider,
            keywords=request.keywords,
            filters=request.filters,
            limit=request.limit
        )
        
        # Convert results to Pydantic models
        result_models = [
            SearchResultModel(
                text=r.text,
                metadata=r.metadata,
                score=r.score,
                provider=r.provider,
                vector_distance=r.vector_distance
            )
            for r in results
        ]
        
        time_taken = time.time() - start_time
        
        return SearchResponse(
            results=result_models,
            time_taken=time_taken,
            total_results=len(results),
            query=request.query
        )
    
    except Exception as e:
        logger.error(f"Hybrid search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@app.post("/similar-documents", response_model=SearchResponse)
async def similar_documents(request: DocumentSimilarityRequest):
    """
    Find documents similar to a given document by ID.
    """
    try:
        import time
        start_time = time.time()
        
        # Execute the document similarity search
        results = search_client.similar_documents(
            document_id=request.document_id,
            provider=request.provider,
            limit=request.limit
        )
        
        # Convert results to Pydantic models
        result_models = [
            SearchResultModel(
                text=r.text,
                metadata=r.metadata,
                score=r.score,
                provider=r.provider,
                vector_distance=r.vector_distance
            )
            for r in results
        ]
        
        time_taken = time.time() - start_time
        
        return SearchResponse(
            results=result_models,
            time_taken=time_taken,
            total_results=len(results),
            query=f"similar-to:{request.document_id}"
        )
    
    except Exception as e:
        logger.error(f"Document similarity search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document similarity search failed: {str(e)}")


@app.get("/providers")
async def get_providers():
    """
    Get a list of available search providers.
    """
    return {
        "providers": search_client.get_available_providers(),
        "default": search_client.default_provider
    }


def run_api(host="0.0.0.0", port=8000):
    """Run the API server."""
    uvicorn.run("search_module.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run_api()
