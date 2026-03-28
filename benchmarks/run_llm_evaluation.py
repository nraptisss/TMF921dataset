#!/usr/bin/env python3
"""
LLM Baseline Evaluation for TMF921 Intents Dataset

This script evaluates how well locally loaded LLMs perform on the TMF921 intents dataset.
Since we cannot use external APIs per requirements, we test with the already downloaded models:
- Qwen3.5-9B (reasoning)
- Qwen3.5-4B (bulk)

The evaluation measures:
1. Syntax validity of generated TMF921 expressions
2. Semantic alignment between NL intents and formal expressions
3. KPI extraction accuracy
4. Schema compliance rate
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset
from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.llm import LLMRouter
from tmf921_dataset_gen.validation.jsonschema_validator import TMFJsonSchemaValidator
from tmf921_dataset_gen.validation.semantic_score import extract_kpis


def evaluate_model_performance(dataset, model_name, task_type="translation"):
    """
    Evaluate a specific model on a subset of the dataset.
    
    Args:
        dataset: Hugging Face dataset object
        model_name: Name of the model to evaluate
        task_type: Either "translation" (NL to TMF921) or "reverse" (TMF921 to NL)
    
    Returns:
        dict: Evaluation metrics
    """
    settings = Settings()
    router = LLMRouter(settings)
    validator = TMFJsonSchemaValidator(settings, "Intent_FVO")
    
    # Track metrics
    total_samples = min(50, len(dataset))  # Evaluate on first 50 samples for speed
    syntax_valid = 0
    schema_compliant = 0
    semantic_scores = []
    kpi_accuracy_scores = []
    
    print(f"Evaluating {model_name} on {total_samples} samples...")
    
    for i in range(total_samples):
        example = dataset[i]
        nl_intent = example["nl_intent"]
        expected_tmf921 = example["tmf921_intent"]
        expected_serialization = example["serialization"]
        
        if task_type == "translation":
            # Test: NL Intent -> TMF921 Expression
            prompt = f"""
            Convert this natural language network intent to a formal TMF921 Intent_FVO expression.
            Preserve all numeric values, units, and constraints exactly.
            Return ONLY the TMF921 expression in {expected_serialization} format.
            
            Natural Language Intent:
            {nl_intent}
            """.strip()
            
            try:
                if model_name == "Qwen3.5-9B":
                    response = router.generate_text(prompt, model=settings.reasoning_model, temperature=0.1, max_tokens=512)
                elif model_name == "Qwen3.5-4B":
                    response = router.generate_text(prompt, model=settings.bulk_model, temperature=0.1, max_tokens=512)
                else:
                    continue
                    
                # Try to parse as JSON if JSON-LD expected
                if expected_serialization == "json-ld":
                    # Extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        generated_expr = json.loads(json_match.group())
                    else:
                        # Try to validate as-is
                        generated_expr = json.loads(response)
                else:
                    generated_expr = response.strip()
                    
                # Basic syntax check
                if expected_serialization == "json-ld":
                    json.loads(json.dumps(generated_expr))  # Will raise if invalid
                    syntax_valid += 1
                else:
                    # For Turtle, do basic validation
                    if "@type" in str(generated_expr) and "expressionValue" in str(generated_expr):
                        syntax_valid += 1
                
                # Semantic similarity (simplified)
                # In practice, you'd use embedding similarity or more sophisticated metrics
                generated_kpis = extract_kpis(str(generated_expr))
                expected_kpis = example["metadata"]["kpis"]
                
                # Simple KPI overlap score
                if generated_kpis and expected_kpis:
                    common_kpis = set(generated_kpis.keys()) & set(expected_kpis.keys())
                    if expected_kpis:
                        kpi_score = len(common_kpis) / len(expected_kpis)
                        kpi_accuracy_scores.append(kpi_score)
                
                # Schema validation
                if expected_serialization == "json-ld":
                    try:
                        # Build minimal valid IntentFVO for validation
                        intent_payload = {
                            "@type": "Intent",
                            "name": f"eval_{i}",
                            "description": nl_intent[:100],
                            "priority": "medium",
                            "context": "evaluation",
                            "version": "1.0",
                            "lifecycleStatus": "active",
                            "expression": generated_expr
                        }
                        validation = validator.validate(intent_payload)
                        if validation.valid:
                            schema_compliant += 1
                    except Exception:
                        pass  # Validation failed
                        
            except Exception as e:
                # Generation or parsing failed
                continue
                
        elif task_type == "reverse":
            # Test: TMF921 Expression -> NL Intent (more challenging)
            # This would require the model to interpret the formal expression
            # Skipping for now as it's more complex and less critical for our use case
            pass
    
    # Calculate metrics
    results = {
        "model": model_name,
        "task": task_type,
        "samples_evaluated": total_samples,
        "syntax_validity_rate": syntax_valid / total_samples if total_samples > 0 else 0,
        "schema_compliance_rate": schema_compliant / total_samples if total_samples > 0 else 0,
        "avg_kpi_accuracy": sum(kpi_accuracy_scores) / len(kpi_accuracy_scores) if kpi_accuracy_scores else 0,
        "eval_timestamp": datetime.now().isoformat()
    }
    
    return results


def main():
    """Main evaluation function."""
    print("Loading TMF921 Intents dataset from Hugging Face Hub...")
    dataset = load_dataset("nraptisss/TMF921-Intents", split="train")
    print(f"Loaded dataset with {len(dataset)} examples")
    
    # Results collection
    all_results = []
    
    # Evaluate Qwen3.5-9B (reasoning model)
    print("\n" + "="*60)
    print("EVALUATING QWEN3.5-9B (REASONING MODEL)")
    print("="*60)
    results_9b = evaluate_model_performance(dataset, "Qwen3.5-9B", "translation")
    all_results.append(results_9b)
    
    # Print results
    print(f"Samples evaluated: {results_9b['samples_evaluated']}")
    print(f"Syntax validity rate: {results_9b['syntax_validity_rate']:.2%}")
    print(f"Schema compliance rate: {results_9b['schema_compliance_rate']:.2%}")
    print(f"Average KPI accuracy: {results_9b['avg_kpi_accuracy']:.2%}")
    
    # Evaluate Qwen3.5-4B (bulk model) 
    print("\n" + "="*60)
    print("EVALUATING QWEN3.5-4B (BULK MODEL)")
    print("="*60)
    results_4b = evaluate_model_performance(dataset, "Qwen3.5-4B", "translation")
    all_results.append(results_4b)
    
    # Print results
    print(f"Samples evaluated: {results_4b['samples_evaluated']}")
    print(f"Syntax validity rate: {results_4b['syntax_validity_rate']:.2%}")
    print(f"Schema compliance rate: {results_4b['schema_compliance_rate']:.2%}")
    print(f"Average KPI accuracy: {results_4b['avg_kpi_accuracy']:.2%}")
    
    # Save results to file
    results_dir = Path(__file__).parent
    results_file = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"{result['model']}:")
        print(f"  Syntax Validity: {result['syntax_validity_rate']:.2%}")
        print(f"  Schema Compliance: {result['schema_compliance_rate']:.2%}")
        print(f"  KPI Accuracy: {result['avg_kpi_accuracy']:.2%}")
        print()


if __name__ == "__main__":
    main()