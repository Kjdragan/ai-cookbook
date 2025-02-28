import lancedb
import json
import os
from pathlib import Path
import numpy as np

def json_serializable(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, dict):
        return {key: json_serializable(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [json_serializable(elem) for elem in item]
    else:
        return item

def view_database():
    # Connect to the LanceDB database
    import os
    
    # Use the standardized path to lancedb_data
    lancedb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "lancedb_data")
    db = lancedb.connect(lancedb_path)

    # Open the chunks table
    table = db.open_table("chunks")

    # Fetch all chunks; converting to pandas then to dict records
    chunks = table.to_pandas().to_dict(orient='records')

    # Print the number of chunks
    print(f"Total chunks in the database: {len(chunks)}")

    # Print each chunk's metadata and text (truncated for readability)
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1}:")
        metadata = json_serializable(chunk.get('metadata', {}))
        print("Metadata:", json.dumps(metadata, indent=4))
        print("Text:", chunk.get('text', '')[:200] + '...')  # Truncate text for readability
        print("Vector length:", len(chunk.get('vector', [])))

if __name__ == "__main__":
    view_database()