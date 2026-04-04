#!/usr/bin/env python3
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.evaluation.benchmark_suite import BenchmarkSuite

def main():
    settings = Settings.from_env()
    suite = BenchmarkSuite(settings)
    
    # Load the test dataset we generated
    dataset_path = Path("test_upgrade/dataset.jsonl")
    if not dataset_path.exists():
        print("Test dataset not found at test_upgrade/dataset.jsonl")
        return 1
    
    records = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} records from test dataset")
    
    # Run the full audit
    audit_results = suite.run_dataset_audit(records)
    
    # Print results
    print("\nBENCHMARK RESULTS:")
    print(json.dumps(audit_results, indent=2))
    
    # Save to file
    results_file = Path("test_upgrade/benchmark_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Check if gates pass
    if audit_results["release_gates"]["overall_pass"]:
        print("✅ All release gates PASSED")
        return 0
    else:
        print("❌ Some release gates FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())