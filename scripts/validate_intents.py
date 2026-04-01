#!/usr/bin/env python3
"""
Intent Validation Script for TMF921 Dataset

This script validates generated TMF921 intents against the official schema
to ensure they are properly formed and compliant.
"""

import json
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.validation.jsonschema_validator import TMFJsonSchemaValidator


def validate_dataset(dataset_path=None, split="train"):
    """
    Validate a dataset of TMF921 intents.
    
    Args:
        dataset_path: Path to local dataset or Hugging Face dataset name
        split: Dataset split to validate (default: "train")
    
    Returns:
        dict: Validation results
    """
    from datasets import load_dataset
    
    settings = Settings.from_env(Path.cwd())
    validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
    
    # Load dataset
    if dataset_path:
        # Load from local file
        if Path(dataset_path).exists():
            if dataset_path.endswith(".jsonl"):
                data = []
                with open(dataset_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                # Convert to format expected by validation
                dataset = data
            else:
                dataset = load_dataset(dataset_path, split=split)
        else:
            # Path provided but doesn't exist, treat as HF dataset name
            dataset = load_dataset(dataset_path, split=split)
    else:
        # Load from Hugging Face Hub (default)
        dataset = load_dataset("nraptisss/TMF921-Intents", split=split)
    
    # Track validation results
    total_samples = len(dataset)
    valid_samples = 0
    schema_valid_count = 0
    tio_compliant_count = 0
    validation_errors = []
    
    print(f"Validating {total_samples} intents...")
    
    # Initialize settings properly
    settings = Settings.from_env(Path.cwd())
    validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
    
    for i, example in enumerate(dataset):
        if i % 100 == 0 and i > 0:
            print(f"  Processed {i}/{total_samples} samples...")
        
        try:
            # Extract the intent payload - it's already a complete Intent_FVO
            tmf921_intent = example["tmf921_intent"]
            nl_intent = example["nl_intent"]
            
            # The tmf921_intent is already a complete Intent_FVO structure
            # Validate it directly without wrapping
            validation = validator.validate(tmf921_intent)
            
            if validation.valid:
                valid_samples += 1
                
                # Check specific compliance metrics from metadata
                if example["metadata"].get("schema_validity") == 1.0:
                    schema_valid_count += 1
                if example["metadata"].get("tio_compliance") == 1.0:
                    tio_compliant_count += 1
            else:
                validation_errors.append({
                    "sample_index": i,
                    "errors": validation.errors,
                    "nl_intent": nl_intent[:100] + "..." if len(nl_intent) > 100 else nl_intent
                })
                
        except Exception as e:
            validation_errors.append({
                "sample_index": i,
                "error": str(e),
                "nl_intent": example["nl_intent"][:100] + "..." if len(example["nl_intent"]) > 100 else example["nl_intent"]
            })
    
    # Calculate results
    results = {
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "validity_rate": valid_samples / total_samples if total_samples > 0 else 0,
        "schema_valid_count": schema_valid_count,
        "schema_valid_rate": schema_valid_count / total_samples if total_samples > 0 else 0,
        "tio_compliant_count": tio_compliant_count,
        "tio_compliant_rate": tio_compliant_count / total_samples if total_samples > 0 else 0,
        "validation_errors_count": len(validation_errors),
        "first_few_errors": validation_errors[:5] if validation_errors else []
    }
    
    return results


def main():
    """Main validation function."""
    print("TMF921 Intent Validation Script")
    print("=" * 50)
    
    # Validate the Hugging Face dataset
    print("\nValidating dataset from Hugging Face Hub: nraptisss/TMF921-Intents")
    results = validate_dataset()
    
    # Clean up any validation result files
    import os
    for f in os.listdir('.'):
        if f.startswith('validation_results_') and f.endswith('.json'):
            try:
                os.remove(f)
            except:
                pass
    
    print("\nVALIDATION RESULTS:")
    print("-" * 30)
    print(f"Total samples: {results['total_samples']}")
    print(f"Valid samples: {results['valid_samples']}")
    print(f"Overall validity rate: {results['validity_rate']:.2%}")
    print(f"Schema valid count: {results['schema_valid_count']}")
    print(f"Schema validity rate: {results['schema_valid_rate']:.2%}")
    print(f"TIO compliant count: {results['tio_compliant_count']}")
    print(f"TIO compliance rate: {results['tio_compliant_rate']:.2%}")
    print(f"Validation errors: {results['validation_errors_count']}")
    
    if results['validation_errors_count'] > 0:
        print("\nFirst few validation errors:")
        for error in results['first_few_errors'][:3]:
            print(f"  Sample {error['sample_index']}: {error.get('error', 'Schema validation failed')}")
            if 'errors' in error:
                for err in error['errors'][:2]:  # Show first 2 errors
                    print(f"    - {err}")
    
    # Save results to file
    import json
    from datetime import datetime
    results_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return appropriate exit code
    if results['validity_rate'] >= 0.95:  # 95% validity threshold
        print("\n✅ VALIDATION PASSED: Dataset meets quality thresholds")
        return 0
    else:
        print("\n❌ VALIDATION FAILED: Dataset below quality thresholds")
        return 1


if __name__ == "__main__":
    sys.exit(main())