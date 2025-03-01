# Docling Document Query System

This document explains how to use the document query system to search through processed documents.

## Query System Features

The document query system enables several powerful search capabilities:

1. **Semantic Search**: Search for documents based on meaning, not just keywords
2. **Hybrid Search**: Combine semantic search with keyword filtering
3. **Context Expansion**: Retrieve adjacent chunks for more comprehensive context
4. **Database Statistics**: View information about processed documents

## Prerequisites

Before using the query system, make sure:

1. You have processed documents using the document pipeline
2. Your OpenAI API key is configured in the `.env` file
3. You have installed all dependencies with `uv add`

## Example Commands

### Basic Semantic Search

```bash
uv run python query_documents.py --query "What is bootstrapped reasoning?" --limit 3
```

### Hybrid Search (Semantic + Keyword)

```bash
uv run python query_documents.py --query "How do multi-agent systems work?" --keyword "feedback" --limit 3
```

### Expanded Context Search

```bash
uv run python query_documents.py --query "Explain test-time scaling" --expand
```

### Database Statistics

```bash
uv run python query_documents.py --stats
```

## Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--query` | `-q` | Semantic search query |
| `--keyword` | `-k` | Optional keyword for hybrid search |
| `--limit` | `-l` | Number of results to return (default: 3) |
| `--expand` | `-e` | Expand context for first result |
| `--stats` | `-s` | Show database statistics |

## Performance Notes

1. The first query may be slower as it initializes connections
2. Embedding generation typically takes 1-2 seconds
3. Search operations are very fast (usually < 0.1 seconds)
4. GPU acceleration doesn't significantly affect query performance
   (but greatly improves document processing)

## Technical Details

- Uses OpenAI's `text-embedding-3-large` model (3072 dimensions)
- Stores vectors in LanceDB for efficient similarity search
- Handles Unicode encoding safely for Windows console output
- Falls back gracefully between different LanceDB API versions

## Troubleshooting

If you encounter issues:

1. Check that your documents were properly processed
2. Verify your `.env` file contains a valid OpenAI API key
3. Make sure the database exists at `lancedb_data` directory
4. If no results are returned, try broader queries or keywords
