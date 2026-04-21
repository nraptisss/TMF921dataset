import json
import re
from pathlib import Path

def evaluate_dataset(file_path):
    results = {
        "total": 0,
        "float_device_count": 0,
        "wrong_reliability_op": 0,
        "payload_nl_mismatch": 0,
        "unrealistic_critical_reliability": 0,
        "details": []
    }

    def get_num(s):
        match = re.search(r"(\d+(?:\.\d+)?)", str(s))
        return float(match.group(1)) if match else None

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            results["total"] += 1
            data = json.loads(line)
            nl = data["nl_intent"].lower()
            payload = data["tmf921_intent"]
            
            constraints = []
            expression = payload.get("expression", {})
            if expression.get("@type") == "JsonLdExpression":
                graph = expression.get("expressionValue", {}).get("@graph", [])
                for node in graph:
                    params = node.get("icm:params", {})
                    for key, values in params.items():
                        if key == "icm:targetDescription": continue
                        for v in values:
                            op, val = next(iter(v.items()))
                            constraints.append({"key": key, "op": op, "val": val})

            # 1. Check for float device_count
            for c in constraints:
                if "deviceCount" in c["key"]:
                    val_str = c["val"].strip(" count")
                    if "." in val_str:
                        # Check if it's actually a float or just .0
                        try:
                            f_val = float(val_str)
                            if f_val != int(f_val):
                                results["float_device_count"] += 1
                                results["details"].append(f"Line {i+1}: float device_count {val_str}")
                        except ValueError:
                            pass

            # 2. Check for wrong reliability operator
            for c in constraints:
                if "reliability" in c["key"]:
                    if "atMost" in c["op"]:
                        results["wrong_reliability_op"] += 1
                        results["details"].append(f"Line {i+1}: reliability using atMost")

            # 3. Check for payload/NL mismatch
            for c in constraints:
                val_num = get_num(c["val"])
                if val_num is not None:
                    # Search for the number in NL, allowing for integer representation
                    # e.g. 500.0 -> "500"
                    search_val = str(int(val_num)) if val_num.is_integer() else str(val_num)
                    if search_val not in nl:
                        results["payload_nl_mismatch"] += 1
                        results["details"].append(f"Line {i+1}: value {search_val} not in NL")
                        break

            # 4. Unrealistic critical reliability
            if payload.get("priority") == "critical":
                for c in constraints:
                    if "reliability" in c["key"]:
                        val_num = get_num(c["val"])
                        if val_num is not None and val_num < 99.0:
                            results["unrealistic_critical_reliability"] += 1
                            results["details"].append(f"Line {i+1}: critical reliability {val_num}%")

    return results

if __name__ == "__main__":
    path = "output/final_dataset_500/dataset.jsonl"
    res = evaluate_dataset(path)
    print(json.dumps(res, indent=2))
