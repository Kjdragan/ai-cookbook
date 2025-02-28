# LanceDB Search Reference

## Core Search Operations

### Basic Vector Search
```python
# Create embedding for query
query_embedding = embedding_model.embed_query("What is machine learning?")

# Perform vector search
results = table.search(query_embedding).limit(10).to_pandas()
```

### Hybrid Search (Vector + Keyword)
```python
# Combine vector search with text filtering
results = table.search(query_embedding).where("text LIKE ?", "%machine learning%").limit(10).to_pandas()
```

### Metadata Filtering
```python
# Filter by metadata fields
results = (
    table.search(query_embedding)
    .where("metadata.doc_type = 'pdf' AND metadata.page_numbers[0] > 5")
    .limit(10)
    .to_pandas()
)
```

### Multiple Conditions
```python
# Complex filtering with multiple conditions
results = (
    table.search(query_embedding)
    .where(
        "(metadata.doc_type = 'pdf' OR metadata.doc_type = 'html') AND text LIKE ?", 
        "%machine learning%"
    )
    .limit(10)
    .to_pandas()
)
```

## Advanced Features

### Reranking
LanceDB provides distance scores which can be used for initial ranking, but custom reranking can be implemented:

```python
# Get results with distance scores
results = table.search(query_embedding).limit(20).to_pandas()

# Custom reranking function
def rerank_results(results, query):
    # Calculate custom scores based on relevance to query
    for i, row in results.iterrows():
        keyword_match_score = sum(keyword in row.text.lower() for keyword in query.lower().split())
        # Combine with vector distance
        custom_score = (1 / (1 + row._distance)) * (1 + 0.2 * keyword_match_score)
        results.at[i, 'custom_score'] = custom_score
    
    # Rerank based on custom score
    return results.sort_values('custom_score', ascending=False)

# Apply reranking
reranked_results = rerank_results(results, query)
```

### Pagination
```python
# Implement pagination for large result sets
page_size = 10
page_number = 2  # 0-indexed

results = (
    table.search(query_embedding)
    .limit(page_size)
    .offset(page_size * page_number)
    .to_pandas()
)
```

## Performance Optimization

### Approximate Nearest Neighbor (ANN) Settings
```python
# Configure ANN search parameters
results = (
    table.search(
        query_embedding,
        nprobes=20,  # Number of clusters to search (higher = more accurate, slower)
        refine_factor=10  # Refine the top results with exact search
    )
    .limit(10)
    .to_pandas()
)
```

### Batch Queries
```python
# Batch process multiple queries for efficiency
query_embeddings = [
    embedding_model.embed_query("What is machine learning?"),
    embedding_model.embed_query("How does deep learning work?")
]

batch_results = []
for query_embedding in query_embeddings:
    results = table.search(query_embedding).limit(10).to_pandas()
    batch_results.append(results)
```

## Error Handling

```python
try:
    results = table.search(query_embedding).limit(10).to_pandas()
except Exception as e:
    logger.error(f"Search failed: {str(e)}")
    # Fallback to simple text search if vector search fails
    results = table.where("text LIKE ?", f"%{query}%").limit(10).to_pandas()
```

## Best Practices

1. **Optimize Vector Dimensions**: LanceDB performs best with vectors of dimension 768-3072
2. **Index Configuration**: For large datasets, configure indexing parameters:
   ```python
   table.create_index(num_partitions=256, num_sub_vectors=96)
   ```
3. **Connection Management**: Reuse LanceDB connections when possible
4. **Query Monitoring**: Log search times and result counts for optimization
