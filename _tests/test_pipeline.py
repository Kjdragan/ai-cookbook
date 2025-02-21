import sys
from pathlib import Path

# Add the docling-playground directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "docling-playground"))

from datapipeline import DataPipeline
import time

def main():
    print("Starting pipeline test...")
    pipeline = DataPipeline()
    
    # Keep the script running to allow file monitoring
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down pipeline...")

if __name__ == "__main__":
    main()
