import re
from typing import List, Dict, Any, Set, Tuple, Optional


def extract_keywords(query: str, min_length: int = 4) -> List[str]:
    """
    Extract meaningful keywords from a query.
    
    Args:
        query: Input query string
        min_length: Minimum length of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    # Convert to lowercase
    query = query.lower()
    
    # Common stopwords to exclude
    stopwords = {
        'the', 'and', 'or', 'but', 'this', 'that', 'these', 'those', 'with', 'from',
        'for', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'not', 'don', 'should', 'would', 'can', 'cannot',
        'could', 'has', 'have', 'had', 'having', 'was', 'were', 'been', 'being', 'what',
        'when', 'where', 'which', 'who', 'whom', 'how', 'why', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'some', 'such', 'only', 'own', 'same', 'than',
        'too', 'very', 'just', 'shall', 'should', 'other'
    }
    
    # Split the query into words and filter
    words = re.findall(r'\b[a-z0-9]+\b', query)
    
    # Filter out stopwords and short words
    keywords = [
        word for word in words 
        if word not in stopwords and len(word) >= min_length
    ]
    
    return keywords


def parse_metadata_filters(filter_query: str) -> Dict[str, Any]:
    """
    Parse a filter query string into a metadata filter dictionary.
    
    Examples:
        "doc_type:pdf date>2023-01-01" -> {"doc_type": "pdf", "date": {">": "2023-01-01"}}
        "filename:report.pdf" -> {"filename": "report.pdf"}
    
    Args:
        filter_query: String of filter conditions
        
    Returns:
        Dictionary of metadata field filters
    """
    if not filter_query:
        return {}
    
    filters = {}
    pattern = r'([a-zA-Z0-9_\.]+)([:<>=]+)([^"\s]+|"[^"]*")'
    
    # Find all filter expressions
    matches = re.findall(pattern, filter_query)
    
    for field, operator, value in matches:
        # Remove quotes from quoted values
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        
        # Convert numeric values
        if value.isdigit():
            value = int(value)
        elif re.match(r'^-?\d+\.\d+$', value):
            value = float(value)
        
        # Handle different operators
        if operator == ':':
            filters[field] = value
        elif operator in ['>', '<', '>=', '<=']:
            if field not in filters:
                filters[field] = {}
            filters[field][operator] = value
    
    return filters


def clean_query(query: str) -> str:
    """
    Clean and normalize a search query.
    
    Args:
        query: Raw input query
        
    Returns:
        Cleaned query string
    """
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove special characters that might interfere with search
    query = re.sub(r'[^\w\s\?\.,:;]', '', query)
    
    return query


def extract_metadata_and_query(combined_query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract metadata filters and the main query from a combined query string.
    
    Example:
        "doc_type:pdf find information about machine learning"
        -> ("find information about machine learning", {"doc_type": "pdf"})
    
    Args:
        combined_query: Query potentially containing metadata filters
        
    Returns:
        Tuple of (clean_query, metadata_filters)
    """
    # Extract potential filter patterns
    filter_pattern = r'([a-zA-Z0-9_\.]+:[^"\s]+|[a-zA-Z0-9_\.]+:"[^"]*")'
    filters = re.findall(filter_pattern, combined_query)
    
    # Remove filters from the query
    clean_query_text = combined_query
    for f in filters:
        clean_query_text = clean_query_text.replace(f, '').strip()
    
    # Clean up the query
    clean_query_text = clean_query(clean_query_text)
    
    # Parse the filters
    filter_query = ' '.join(filters)
    metadata_filters = parse_metadata_filters(filter_query)
    
    return clean_query_text, metadata_filters
