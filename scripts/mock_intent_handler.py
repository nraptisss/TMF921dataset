#!/usr/bin/env python3
"""
Mock Intent Handler for TMF921 Intents

DEMO SCRIPT — NOT PART OF THE GENERATION PIPELINE.

This script demonstrates how parsed TMF921 intents could be translated
into simulated network actions in an intent-based networking system.
It loads examples from the published dataset and shows a mock execution
flow. It is not used by the dataset generator, verifier, or benchmark suite.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset


class MockNetworkAction:
    """Represents a simulated network action."""
    
    def __init__(self, action_type: str, target: str, parameters: Dict[str, Any]):
        self.action_type = action_type
        self.target = target
        self.parameters = parameters
        self.timestamp = datetime.now().isoformat()
        self.status = "PENDING"
    
    def execute(self) -> Dict[str, Any]:
        """Simulate executing the network action."""
        # In a real system, this would call actual network APIs or controllers
        self.status = "SUCCESS"
        
        # Simulate different action types
        if self.action_type == "CREATE_SLICE":
            result = {
                "slice_id": f"slice-{hash(self.target) % 10000:04d}",
                "status": "ACTIVE",
                "bandwidth_allocated": self.parameters.get("throughput_mbps", 0),
                "latency_guaranteed": self.parameters.get("latency_ms", float('inf')),
                "reliability_guaranteed": self.parameters.get("reliability_percent", 0)
            }
        elif self.action_type == "MODIFY_QOS":
            result = {
                "qos_policy_id": f"qos-{hash(self.target) % 1000:03d}",
                "status": "APPLIED",
                "updated_parameters": self.parameters
            }
        elif self.action_type == "CONFIGURE_MONITORING":
            result = {
                "monitoring_session_id": f"mon-{hash(self.target) % 10000:04d}",
                "status": "ACTIVE",
                "reporting_interval": self.parameters.get("reporting_interval_seconds", 0)
            }
        else:
            result = {"status": "EXECUTED", "action": self.action_type}
        
        return {
            "action": self.action_type,
            "target": self.target,
            "execution_time": self.timestamp,
            "status": self.status,
            "result": result
        }


def parse_tmf921_intent(tmf921_intent: Dict[str, Any]) -> List[MockNetworkAction]:
    """
    Parse a TMF921 intent into simulated network actions.
    
    This is a simplified demonstration of how an intent translation layer
    might work in a real IBN system.
    """
    actions = []
    
    # Extract basic info
    intent_type = tmf921_intent.get("@type", "Intent")
    name = tmf921_intent.get("name", "unknown")
    expression = tmf921_intent.get("expression", {})
    
    # Handle JSON-LD format
    if isinstance(expression, dict) and "@graph" in expression:
        for graph_item in expression["@graph"]:
            if graph_item.get("@type") == "icm:Intent":
                # Extract expectations (what we want to achieve)
                expectations = graph_item.get("icm:hasExpectation", [])
                for exp in expectations:
                    exp_id = exp.get("@id", "")
                    exp_type = exp.get("@type", "")
                    exp_target = exp.get("icm:target", {})
                    exp_params = exp.get("icm:params", {})
                    
                    # Convert TMF921 parameters to network actions
                    if exp_type == "icm:DeliveryExpectation":
                        action = _create_delivery_action(name, exp_target, exp_params)
                        if action:
                            actions.append(action)
                    elif exp_type == "icm:ReportingExpectation":
                        action = _create_reporting_action(name, exp_target, exp_params)
                        if action:
                            actions.append(action)
    
    # Handle Turtle format (simplified)
    elif isinstance(expression, str) and "@type" in expression and "TurtleExpression" in expression:
        # In a real implementation, we'd parse the Turtle format
        # For this demo, we'll create a generic action
        actions.append(MockNetworkAction(
            action_type="GENERIC_INTENT_HANDLING",
            target=name,
            parameters={"raw_expression": expression[:200]}  # Truncate for safety
        ))
    
    return actions


def _create_delivery_action(intent_name: str, target: Dict[str, Any], params: Dict[str, Any]) -> MockNetworkAction:
    """Create a network action from a DeliveryExpectation."""
    action_type = "UNKNOWN"
    action_target = target.get("@id", "unknown_target")
    
    # Determine action type based on parameters
    has_latency = any(k.startswith("met:latency") for k in params.keys())
    has_throughput = any(k.startswith("met:throughput") for k in params.keys())
    has_reliability = any(k.startswith("met:reliability") for k in params.keys())
    has_energy = any(k.startswith("met:energyConsumption") for k in params.keys())
    
    if has_latency and has_reliability:
        action_type = "CREATE_URLLC_SLICE"
    elif has_throughput and has_reliability:
        action_type = "CREATE_EMBB_SLICE"
    elif has_energy:
        action_type = "OPTIMIZE_FOR_ENERGY"
    elif has_throughput:
        action_type = "CREATE_BASIC_CONNECTION"
    else:
        action_type = "GENERIC_CONFIGURATION"
    
    # Extract parameter values for the action
    action_params = {}
    for param_key, param_values in params.items():
        if isinstance(param_values, list) and param_values:
            # Take the first value entry
            value_entry = param_values[0]
            if isinstance(value_entry, dict):
                for k, v in value_entry.items():
                    if k != "@type":  # Skip type info
                        action_params[k] = v
    
    return MockNetworkAction(
        action_type=action_type,
        target=action_target,
        parameters=action_params
    )


def _create_reporting_action(intent_name: str, target: Dict[str, Any], params: Dict[str, Any]) -> MockNetworkAction:
    """Create a network action from a ReportingExpectation."""
    action_target = target.get("@id", "unknown_target")
    
    action_params = {}
    for param_key, param_values in params.items():
        if isinstance(param_values, list) and param_values:
            value_entry = param_values[0]
            if isinstance(value_entry, dict):
                for k, v in value_entry.items():
                    if k != "@type":
                        action_params[k] = v
    
    return MockNetworkAction(
        action_type="CONFIGURE_MONITORING",
        target=action_target,
        parameters=action_params
    )


def demo_intent_handling(dataset_limit: int = 5):
    """
    Demonstrate intent handling with a few examples from the dataset.
    
    Args:
        dataset_limit: Number of examples to process
    """
    print("TMF921 Intent Handler Demonstration")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset from Hugging Face Hub...")
    dataset = load_dataset("nraptisss/TMF921-Intents", split="train")
    
    print(f"Processing first {min(dataset_limit, len(dataset))} intents...\n")
    
    for i in range(min(dataset_limit, len(dataset))):
        example = dataset[i]
        nl_intent = example["nl_intent"]
        tmf921_intent = example["tmf921_intent"]
        serialization = example["serialization"]
        
        print(f"Example {i+1}:")
        print(f"NL Intent: {nl_intent[:100]}{'...' if len(nl_intent) > 100 else ''}")
        print(f"Serialization: {serialization}")
        
        # Parse the intent into network actions
        actions = parse_tmf921_intent(tmf921_intent)
        
        print(f"Parsed into {len(actions)} network action(s):")
        for j, action in enumerate(actions):
            result = action.execute()
            print(f"  Action {j+1}: {result['action']}")
            print(f"    Target: {result['target']}")
            print(f"    Status: {result['status']}")
            if 'result' in result and isinstance(result['result'], dict):
                for key, value in result['result'].items():
                    if key not in ['status']:
                        print(f"    {key}: {value}")
        print()


def validate_and_handle(dataset_path: str | None = None, limit: int = 50):
    """
    Validate intents from a dataset and demonstrate handling.
    
    Args:
        dataset_path: Path to dataset or None for Hugging Face
        limit: Maximum number of intents to process
    """
    from datasets import load_dataset
    
    print("Loading dataset for validation and handling demo...")
    
    if dataset_path and Path(dataset_path).exists():
        if dataset_path.endswith(".jsonl"):
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            dataset = data
        else:
            dataset = load_dataset(dataset_path, split="train")
    else:
        dataset = load_dataset("nraptisss/TMF921-Intents", split="train")
    
    total_to_process = min(limit, len(dataset))
    print(f"Processing {total_to_process} intents for validation and handling...")
    
    successful_handling = 0
    total_actions = 0
    
    for i in range(total_to_process):
        example = dataset[i]
        nl_intent = example["nl_intent"]
        tmf921_intent = example["tmf921_intent"]
        
        try:
            actions = parse_tmf921_intent(tmf921_intent)
            if actions:
                # Execute all actions
                for action in actions:
                    result = action.execute()
                    if result["status"] == "SUCCESS":
                        successful_handling += 1
                    total_actions += 1
                    
                if i % 10 == 0:
                    print(f"  Processed {i+1}/{total_to_process} intents...")
        except Exception as e:
            if i % 10 == 0:
                print(f"  Error processing intent {i+1}: {str(e)[:50]}...")
    
    print(f"\nHandling Results:")
    print(f"  Intents processed: {total_to_process}")
    print(f"  Total actions attempted: {total_actions}")
    print(f"  Successfully executed: {successful_handling}")
    print(f"  Success rate: {successful_handling/total_actions*100:.1f}%" if total_actions > 0 else "  Success rate: 0%")


def main():
    """Main demonstration function."""
    print("TMF921 Mock Intent Handler")
    print("=" * 30)
    
    # Run the demo
    demo_intent_handling(dataset_limit=3)
    
    # Run validation and handling on a larger set
    print("\n" + "=" * 50)
    print("RUNNING VALIDATION AND HANDLING ON LARGER SAMPLE")
    print("=" * 50)
    validate_and_handle(limit=20)


if __name__ == "__main__":
    main()