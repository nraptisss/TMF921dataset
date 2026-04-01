---
license: cc-by-4.0
task_categories:
  - text-classification
  - text-generation
size_categories:
  - 10K<n<100K
language:
  - en
---

# TMF921 Intents Dataset

This dataset contains 10,000 synthetically generated natural-language intents paired with their formal TMF921 v5.0 `Intent_FVO` representations. The intents were generated using the enhanced TMF921 Dataset Generator with improved realism features, running locally on an NVIDIA RTX 6000 Ada GPU with Qwen 3.5 language models.

## 🔬 **Enhancements Over Original**

This version includes significant improvements for training robust NLU models:

### 🎭 **Natural Language Template Variations**
- **Conversational**: "Hey, can we minimize a EMBB slice for the private campus? I need throughput of at least 428 Mbps..."
- **Administrative**: "Action required: minimize resource for private campus. Specs: support for 12433 devices..."
- **Business-focused**: "We need to optimize the business interface: throughput of at least 719 Mbps..."
- **Technical precise**: "Implement coordinate MMTC functionality. Constraints: support for 6645 devices..."

### 🔊 **Realistic Noise & Ambiguity**
- **KPI Omission**: ~10% random omission of KPIs to simulate incomplete operator requests
- **Value Variation**: ±5% noise on numeric values to reflect real-world imprecision
- **Missing Parameters**: Some intents intentionally lack certain specifications

## Dataset Structure

Each example contains:

- **`nl_intent`**: Natural language English description of the intent (enhanced with variations)
- **`tmf921_intent`**: Fully structured TMF921 Intent_FVO object (JSON-LD or Turtle expression)
- **`serialization`**: Either `"json-ld"` or `"turtle"` indicating the format of `tmf921_intent`
- **`metadata`**: Rich annotation object containing:
  - `taxonomy_category`: Hierarchical classification (e.g., `service/embb/energy`)
  - `kpis`: Extracted key performance indicators with values and units
  - `quality_score`: Composite quality metric (0-1)
  - `tio_compliance`: Binary flag indicating compliance with TMF Open API schema
  - `schema_validity`: Binary flag indicating validity against TMF921 Intent Management schema
  - `realism_score`: Grounding fidelity to source corpus (0-1)
  - `semantic_score`: Alignment between natural language and formal expression (0-1)
  - `validation_notes`: Any validation warnings (empty array if none)
  - `seed_id`: Identifier of the source seed document used for grounding
  - `generation_timestamp`: ISO 8601 timestamp of generation

## Generation Methodology

The intents were generated using an enhanced retrieval-augmented pipeline:

1. **Corpus Construction**: Processed TR290 documents, TMF921 specification seeds, and IDAN reference implementations into a normalized corpus.
2. **Retrieval-Augmented Generation**: 
   - Reasoning model (Qwen/Qwen3.5-9B) generates structured intermediate representations from retrieved context and seed prompts
   - Translation model (Qwen/Qwen3.5-4B) converts these into natural language descriptions and formal TMF921 expressions
3. **Enhanced Diversity Agent**: Applies template variations and noise injection for improved realism
4. **Grounded Sampling**: Outputs are constrained using retrieved passages to ensure factual grounding in source materials.

## Technical Details

- **Models** (all running locally in bfloat16 precision):
  - Reasoning: `Qwen/Qwen3.5-9B` (9 billion parameters)
  - Bulk generation: `Qwen/Qwen3.5-4B` (4 billion parameters)
  - Embeddings: `BAAI/bge-large-en-v1.5` (for retrieval)
- **Inference**: `INFERENCE_BACKEND=local-transformers` with `LOCAL_FILES_ONLY=true`
- **Validation**: Each intent passes schema validation, TIO compliance, and quality checks
- **Hardware**: NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("nraptisss/TMF921-Intents", split="train")

# Access first example
example = dataset[0]
print("Natural Language Intent:", example["nl_intent"])
print("TMF921 Intent:", example["tmf921_intent"])
print("Metadata:", example["metadata"])
```

## Citation

If you use this dataset in your work, please cite the TMF921 Dataset Generator project:

```
@misc{tmf921_dataset_generator,
  title={TMF921 Dataset Generator},
  author={Nick Raptis},
  year={2026},
  publisher={GitHub},
  journal={GitHub repository},
  url={https://github.com/nraptisss/codex-dataset}
}
```

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Contact

For questions or contributions, please refer to the [project repository](https://github.com/nraptisss/codex-dataset).