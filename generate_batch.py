#!/usr/bin/env python3
"""
Simple batch generation script to generate samples without CLI timeout issues.
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.graph.workflow import run_generation

def generate_batch(batch_size: int, output_dir: str):
    """Generate a batch of samples."""
    settings = Settings.from_env()

    print(f"Starting generation of {batch_size} samples...")
    print(f"Output directory: {output_dir}")

    start_time = time.time()

    try:
        records = run_generation(settings, count=batch_size, output_dir=Path(output_dir))
        elapsed = time.time() - start_time

        print(f"Generation completed in {elapsed:.2f} seconds")
        print(f"Generated {len(records)} records")

        output_path = Path(output_dir)
        if (output_path / "dataset.jsonl").exists():
            with open(output_path / "dataset.jsonl", 'r') as f:
                lines = [line for line in f if line.strip()]
            print(f"Dataset file contains {len(lines)} lines")

        return {"generated_records": len(records)}

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_batch.py <batch_size> <output_dir>")
        print("Example: python generate_batch.py 1000 output/test_batch")
        sys.exit(1)
    
    batch_size = int(sys.argv[1])
    output_dir = sys.argv[2]
    
    generate_batch(batch_size, output_dir)