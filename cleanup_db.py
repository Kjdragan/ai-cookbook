"""
Temporary script to delete database files safely.
"""
import os
import sqlite3
import time
import shutil

# Close any open database connections
db_paths = ["processed_files.db"]
for db_path in db_paths:
    if os.path.exists(db_path):
        try:
            # Try to open and close the database to release locks
            conn = sqlite3.connect(db_path)
            conn.close()
            print(f"Successfully closed connection to {db_path}")
        except Exception as e:
            print(f"Error closing connection to {db_path}: {e}")

# Sleep to allow processes to complete
time.sleep(1)

# Remove the database files
if os.path.exists("processed_files.db"):
    try:
        os.remove("processed_files.db")
        print("Successfully removed processed_files.db")
    except Exception as e:
        print(f"Error removing processed_files.db: {e}")

# Remove the LanceDB directory if it exists
if os.path.exists("lancedb_data"):
    try:
        shutil.rmtree("lancedb_data")
        print("Successfully removed lancedb_data directory")
    except Exception as e:
        print(f"Error removing lancedb_data directory: {e}")

print("Database cleanup complete")
