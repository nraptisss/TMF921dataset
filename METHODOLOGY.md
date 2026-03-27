# TMF921 Intent Dataset Generation Methodology

## Overview

This document describes the end-to-end process used to generate a synthetic, production-oriented dataset of natural-language intents paired with TMF921 v5.0 `Intent_FVO` payloads. The dataset was generated entirely on-premises using GPU-accelerated local LLMs from the Qwen 3.5 family, without reliance on external APIs.

The resulting dataset is available at:
- Hugging Face Hub: https://huggingface.co/datasets/nraptisss/TMF921-Intents
- Local repository: `output/large_dataset/` (contains `dataset.jsonl` and `manifest.json`)

## Motivation

The TMF921 Intent Management specification defines a standardized framework for expressing intent-based networking in telecom and enterprise environments. However, there is a scarcity of large, high-quality datasets that pair natural language intent expressions with their formal TMF921 representations. This gap hinders:
- Development and testing of intent interpretation and translation systems
- Training of machine learning models for intent recognition in telecom domains
- Benchmarking of intent-based network automation (IBNA) solutions

Our dataset addresses this need by providing 1,000 synthetically generated but specification-compliant intent pairs, covering diverse scenarios across service, resource, and business layers.

## Methodology

### 1. Environment Setup

- **Hardware**: NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM)
- **Software Stack**:
  - Ubuntu-based Linux environment
  - Python 3.13
  - PyTorch 2.11.0 with CUDA 12.4
  - Hugging Face Transformers 5.4.0
  - Accelerate, Sentence-Transformers, BitsAndBytes for efficient inference
- **Models** (all loaded locally in bfloat16 precision):
  - Reasoning: `Qwen/Qwen3.5-9B` (9 billion parameters)
  - Bulk generation: `Qwen/Qwen3.5-4B` (4 billion parameters)
  - Embeddings: `BAAI/bge-large-en-v1.5` (for retrieval-augmented generation)

### 2. Pipeline Architecture

The generation follows a modular, agent-based workflow implemented in the TMF921 Dataset Generator codebase, comprising:

#### a. Corpus Construction (`build-corpus`)
- Input sources: TR290 documents, TMF921 specification seeds, IDAN reference implementations
- Processing: PDF/DOCX extraction, text cleaning, chunking, and normalization
- Output: Structured corpus (~189 documents) stored in `artifacts/normalized/corpus/`

#### b. Retrieval-Augmented Generation (RAG) Setup
- Embedding model encodes corpus passages into vector space
- ChromaDB vector store enables semantic retrieval during generation
- Top-k=5 relevant passages retrieved per generation step to ground outputs in domain knowledge

#### c. Intent Generation (`generate`)
- **Reasoning Model (Qwen3.5-9B)**: 
  - Reads retrieved context and intent seed prompts
  - Produces structured intermediate representations (JSON/YAML) describing:
    - Intent type (Service, Resource, Business, Probe)
    - Layer (service, resource, business)
    - Key performance indicators (KPIs) and constraints
    - Priority level and lifecycle status
- **Bulk Model (Qwen3.5-4B)**:
  - Takes reasoning output and converts it into:
    - Natural language intent description (`nl_intent`)
    - Formal TMF921 Intent_FVO expression (`tmf921_intent`) in either JSON-LD or Turtle serialization
- Temperature: 0.7 (balanced creativity/consistency)
- Max new tokens: 1024
- Top-p: 0.9 (nucleus sampling)

#### d. Grounded Sampling (`sample`)
- Uses the same generation pipeline but constrains outputs using retrieved corpus passages
- Ensures factual grounding in source materials (TR290, seeds, IDAN references)
- Outputs directed to user-specified directory (e.g., `output/large_dataset/`)

### 3. Key Technical Choices & Justifications

| Choice | Reasoning |
|--------|-----------|
| **Qwen 3.5 Family** | State-of-the-art open LLMs with strong reasoning, multilingual, and code capabilities. The 9B/4B split fits within 48GB VRAM while allowing concurrent reasoning + generation workloads. |
| **bfloat16 Precision** | Provides near-FP32 numerical stability with half the memory footprint, critical for fitting large models on consumer/prosumer GPUs. |
| **Local-Only Inference (`LOCAL_FILES_ONLY=true`)** | Eliminates latency, cost, and privacy concerns associated with API-dependent approaches. Enables air-gapped deployment for sensitive telecom environments. |
| **Retrieval-Augmented Generation** | Mitigates hallucination by anchoring generated intents in verified domain documents (TR290, specification seeds). Improves realism and compliance scores. |
| **Hybrid Serialization (JSON-LD/Turtle)** | Reflects TMF921's support for multiple expression formats, increasing dataset utility for diverse downstream consumers. |
| **Intent Taxonomy Coverage** | Generated intents span all major TMF921 layers (service/resource/business) and types (Intent/ProbeIntent) to maximize applicability. |

### 4. Validation & Quality Assurance

Each generated intent undergoes multiple validation checkpoints:

1. **Schema Validation**: 
   - Intent_FVO objects validated against `TMF921_Intent_Management_v5.0.0.oas.yaml`
   - Reflected in `"schema_validity": 1.0` metadata field

2. **TMF Open API (TIO) Compliance**:
   - Checked against canonical TMF921 schema
   - Indicated by `"tio_compliance": 1.0`

3. **Semantic Consistency**:
   - Automated checks ensure natural language description aligns with structured KPIs
   - Captured in `"semantic_score"` (typically >0.96)

4. **Realism Score**:
   - Measures grounding fidelity to source corpus via embedding similarity
   - Reflected in `"realism_score"` (typically >0.88)

5. **Manual Spot-Checking**:
   - Random samples reviewed for linguistic quality and intent coherence

## Dataset Contents

The dataset (`output/large_dataset/`) contains:

### `dataset.jsonl`
- Format: One JSON object per line (JSONL)
- Fields:
  - `nl_intent`: Natural language English description of the intent
  - `tmf921_intent`: Fully structured TMF921 Intent_FVO object (JSON-LD or Turtle expression)
  - `serialization`: Either `"json-ld"` or `"turtle"`
  - `metadata`: Rich annotation including:
    - `taxonomy_category`: Hierarchical classification (e.g., `service/embb/energy`)
    - `kpis`: Extracted key performance indicators with values
    - `quality_score`: Composite metric (0-1)
    - `tio_compliance`: Binary compliance flag
    - `schema_validity`: Binary schema validation flag
    - `realism_score`: Grounding to source documents (0-1)
    - `semantic_score`: NL-to-formal alignment score (0-1)
    - `validation_notes`: Any validation warnings (empty array if none)
    - `seed_id`: Identifier of the source seed document used for grounding
    - `generation_timestamp`: ISO 8601 timestamp of creation

### `manifest.json`
- Pipeline execution metadata:
  - Timestamps for each stage (preflight, build-corpus, generate, sample)
  - Model versions and hardware used
  - Configuration parameters (temperature, top-p, etc.)
  - Aggregate statistics (count, average scores, etc.)

## Contributions to the Field

This dataset and methodology offer several novel contributions:

### 1. First Large-Scale Synthetic TMF921 Intent Corpus
- To our knowledge, the largest publicly available dataset pairing NL intent descriptions with formal TMF921 representations
- Enables data-driven approaches to intent-based networking research

### 2. Fully Local, Reproducible Generation Pipeline
- Demonstrates that high-fidelity, domain-specific synthetic data generation is achievable without external APIs
- Provides a template for other organizations to generate compliant intent data in air-gapped or regulated environments
- All model choices, prompts, and hyperparameters are documented and replicable

### 3. Rich Multi-Modal Annotation
- Goes beyond simple intent-label pairs to include:
  - Formal semantic representations (JSON-LD/Turtle)
  - Extracted KPIs with units and constraint types
  - Quality and realism metrics for filtering/sorting
  - Provenance tracking to source seed materials

### 4. Foundation for Downstream Applications
The dataset supports immediate use in:
- **Intent Classification**: Train NLU models to map user/network goals to TMF921 intent types
- **Intent Translation**: Develop seq2seq or structured prediction models converting NL to Intent_FVO
- **Constraint Extraction**: Build systems that identify KPIs, thresholds, and relations from textual intents
- **Intent Validation**: Create verifiers that check generated intents against TMF921 schema and business rules
- **Retrieval-Augmented Intent Systems**: Use the embedded corpus to build RAG pipelines for intent drafting assistance

### 5. Open Science & Collaboration
- By releasing both the generation codebase (this repository) and the generated dataset under accessible terms, we promote:
  - Reproducibility of results
  - Comparative studies between different modeling approaches
  - Community-driven extension to other TMF specifications (e.g., TMF620, TMF638)
  - Benchmarking of intent management platforms

## Usage Example

```python
# Load the dataset
from datasets import load_dataset
dataset = load_dataset("nraptisss/TMF921-Intents", split="train")

# Access first sample
sample = dataset[0]
print("NL Intent:", sample["nl_intent"])
print("TMF921 Intent:", sample["tmf921_intent"])
print("KPIs:", sample["metadata"]["kpis"])
```

## Ethical Considerations

- **Synthetic Nature**: All intents are artificially generated; no private or proprietary information from live networks is included
- **TMF921 Compliance**: Generated intents adhere strictly to the publicly available TMF921 v5.0 specification
- **Bias Mitigation**: Generation prompts emphasize diversity across intent types, layers, and KPI combinations to avoid over-representation of specific scenarios
- **Transparency**: Full documentation of model choices, parameters, and validation metrics enables informed reuse

## Future Work

Potential extensions include:
- Expanding to other TMF specifications (e.g., Service Catalog, Trouble Ticket)
- Incorporating multi-lingual intent expressions
- Adding adversarial examples for robustness testing
- Linking generated intents to network topology or service orchestration models
- Creating evaluation benchmarks for intent translation tasks

---

*Generated as part of the TMF921 Dataset Generator project. For questions or contributions, please refer to the project repository.*