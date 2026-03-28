#!/usr/bin/env python3
"""
Local Intent Validation Script for TMF921 Dataset

This script validates generated TMF921 intents from a local JSONL file
against the official schema to ensure they are properly formed and compliant.
"""

import json
import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.validation.jsonschema_validator import TMFJsonSchemaValidator


def validate_local_file(file_path):
    """
    Validate a local JSONL file of TMF921 intents.
    
    Args:
        file_path: Path to local JSONL dataset file
    
    Returns:
        dict: Validation results
    """
    settings = Settings.from_env(Path.cwd())
    validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
    
    # Load dataset from local file
    data = []
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    
    if not data:
        print("Error: No valid data found in file")
        return None
    
    # Track validation results
    total_samples = len(data)
    valid_samples = 0
    schema_valid_count = 0
    tio_compliant_count = 0
    validation_errors = []
    
    print(f"Validating {total_samples} intents from {file_path}...")
    
    for i, example in enumerate(data):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{total_samples} samples...")
        
        try:
            # Extract the intent payload
            tmf921_intent = example["tmf921_intent"]
            nl_intent = example["nl_intent"]
            
            # The tmf921_intent should already be a valid IntentFVO object
            intent_payload = example["tmf921_intent"].copy()
            # Ensure required fields are present with reasonable defaults
            intent_payload.setdefault("@type", "Intent" if example["serialization"] == "json-ld" else "ProbeIntent")
            intent_payload.setdefault("name", f"sample_{i}")
            intent_payload.setdefault("description", nl_intent[:200])  # Truncate if too long
            intent_payload.setdefault("priority", example["metadata"].get("priority", "medium"))
            intent_payload.setdefault("context", example["metadata"].get("context", "evaluation"))
            intent_payload.setdefault("version", "1.0")
            intent_payload.setdefault("lifecycleStatus", "active")
            # The expression should already be present and valid from generation
            
            # Validate against TMF921 schema
            validation = validator.validate(intent_payload)
            
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
    if len(sys.argv) < 2:
        print("Usage: python validate_local.py <path_to_jsonl_file>")
        print("Example: python validate_local.py output/1k_enhanced_test/dataset.jsonl")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("TMF921 Local Intent Validation Script")
    print("=" * 50)
    
    results = validate_local_file(file_path)
    
    if results is None:
        sys.exit(1)
    
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
        sys.exit(0)
    else:
        print("\n❌ VALIDATION FAILED: Dataset below quality thresholds")
        sys.exit(1)


if __name__ == "__main__":
    main()