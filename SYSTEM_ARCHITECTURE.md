# TMF921 Dataset Generation System Architecture

## Overview
The TMF921 Dataset Generation system is a high-fidelity synthetic data pipeline designed to create a large-scale dataset of Natural Language (NL) intents and their corresponding TMF921 formal representations (JSON-LD/Turtle). The system focuses on **grounding**, **diversity**, and **strict technical compliance** with the TMF921/TIO standards.

## Core Pipeline: The Grounded Generation Graph
The system is implemented as a state-driven graph using `langgraph`, ensuring a repeatable and iterative process for every generated record.

### Workflow Sequence
The pipeline follows a **Retrieval-First** sequence to prevent hallucinated technical constraints:
`START` $\rightarrow$ `Retrieve` $\rightarrow$ `Diversify` $\rightarrow$ `Translate` $\rightarrow$ `Critique` $\rightarrow$ (`Refine` $\rightarrow$ `Critique`)* $\rightarrow$ `END`

### Component Detailed Breakdown

#### 1. BalancedRetriever (`rag/retriever.py`)
The retriever provides the domain evidence necessary for grounding.
- **Implementation**: Uses a `ChromaVectorStore` to index a normalized corpus of telecom documents (TR 290, seeds, IDAN references).
- **Balanced Strategy**: Instead of a simple top-k search, it implements a "balanced" retrieval that ensures a minimum number of chunks are pulled from each high-priority source (e.g., seeds, PDFs, docs) to avoid source bias.
- **Why**: This prevents the generator from over-relying on a single document and provides a broad evidence base for technical KPI validation.

#### 2. DiversityAgent (`agents/diversity.py`)
Responsible for generating a diverse set of natural language intents.
- **Grounding-First Logic**: 
    - It first attempts to extract actual KPI values (e.g., "500 Mbps", "99.999%") from the `retrieved_context`.
    - If `GROUNDING_MODE` is `grounded_corpus`, it **strictly prohibits** random sampling; if a value isn't in the evidence, it isn't generated.
- **Template-Based Diversity**: Uses multiple NL templates (conversational, administrative, business-focused, technical) to ensure the dataset reflects real-world operator requests.
- **LLM Rewrite**: An optional pass where an LLM polishes the template-generated intent into a natural, production-style request while strictly preserving the grounding KPIs.

#### 3. TranslatorAgent (`agents/translator.py`)
Maps the NL intent into the formal TMF921/TIO structure.
- **Formalization**: It doesn't just generate JSON; it builds a formal `IntentFrame` (symbolic representation) and then renders it into JSON-LD or Turtle.
- **Serialization**: Dynamically chooses between JSON-LD and Turtle based on a configurable ratio to ensure dataset variety.
- **Deterministic Mapping**: Uses `METRIC_SPECS` to map internal KPI keys to formal TMF921 payload keys (e.g., `latency_ms` $\rightarrow$ `met:latency`).

#### 4. CriticRefinementAgent (`agents/critic.py`)
The quality gate that decides if a record is accepted. It employs a **Multi-Layer Validation** strategy:
- **Layer 1: Schema Validation**: Ensures the output is a valid TMF921 JSON/Turtle structure using `jsonschema`.
- **Layer 2: TIO Compliance**: Checks for mandatory fields and structural rules defined by TIO.
- **Layer 3: Symbolic Alignment**: Uses the `SemanticFrame` to verify that the formal payload exactly matches the NL intent (no missing metrics, no value mismatches).
- **Layer 4: Evidence Grounding**: Verifies that every technical claim in the intent is supported by at least one row in the retrieved corpus.
- **Layer 5: KPI Plausibility**: Checks if the generated values are within realistic telecom ranges.
- **Refinement Loop**: If a record fails but is within the `max_refinement_loops`, the `TranslatorAgent` attempts a heuristic repair before the record is re-critiqued.

#### 5. Semantic Frame (`validation/semantic_frame.py`)
Acts as the symbolic "bridge" between NL and Formal representations.
- **Logic**: It extracts metrics and operators (e.g., `at_least`, `at_most`) from NL text and creates a canonical representation.
- **Why**: This allows the system to perform **symbolic verification**. Instead of asking an LLM "does this look right?", the system can mathematically prove that the payload's value matches the intent's value.

#### 6. Manifest & Audit (`export/manifest.py`)
Provides a research-grade evaluation of the final dataset.
- **Metrics**: Calculates diversity scores, bias reports, semantic pass rates, and supported claim ratios.
- **Release Gates**: Implements automatic gates (e.g., `release_semantic_pass_threshold`) to determine if the dataset is "Release Ready".

## Design Rationale

### Why a Graph Structure?
Traditional linear pipelines fail when a late-stage error (e.g., a critic rejection) requires an early-stage fix. The graph allows for conditional edges and refinement loops, enabling the system to "self-correct" without restarting the entire process.

### Why Symbolic Validation over LLM-only Review?
LLMs are prone to "yes-man" bias and can overlook small numeric mismatches. By extracting a `SemanticFrame`, the system can perform exact string and numeric comparisons, ensuring 100% faithfulness to the grounding sources.

### The "Retrieval-First" Shift
Early versions generated KPIs and then tried to find evidence for them. This led to high rejection rates in strict grounding modes. The current implementation retrieves evidence **first**, then derives KPIs **from** that evidence, fundamentally aligning the generator with the critic.

## Configuration & Backend
The system is decoupled from its inference engine via `LLMRouter`, allowing it to run in:
- **Mock Mode**: Uses heuristic templates for fast development and testing.
- **Local-Transformers Mode**: Uses local LLMs (e.g., Qwen) via HuggingFace for production-grade generation.
