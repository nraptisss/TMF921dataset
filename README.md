# TMF921 Dataset Generator

Synthetic, production-oriented pipeline for generating natural-language intents paired with TMF921 v5.0 `Intent_FVO` payloads.

## Links

- GitHub repository: https://github.com/nraptisss/TMF921dataset
- Hugging Face dataset: https://huggingface.co/datasets/nraptisss/TMF921-Intents

## Documentation

- Docs index: `docs/README.md`
- Methodology: `docs/methodology.md`
- Original plan baseline: `docs/plan.md`

## What This Project Does

This project builds grounded telecom intent datasets with:

- NL intents (`nl_intent`)
- Structured TMF921 payloads (`tmf921_intent`)
- Mixed serialization (`json-ld` and `turtle`)
- Rich metadata (taxonomy, KPIs, semantic/quality/TIO scoring, validation notes)

The core generation flow is:

1. Build normalized corpus from TMF/OAS/Postman/TR290/seeds/IDAN sources.
2. Retrieve grounding context (RAG).
3. Generate diverse NL intent candidates.
4. Translate into TMF921 payloads.
5. Critique/refine and accept/reject.
6. Export `dataset.jsonl` + `manifest.json`.

## Current Architecture

- Source package: `src/tmf921_dataset_gen/`
- CLI entrypoint: `python -m tmf921_dataset_gen.cli`
- Workflow orchestration: `src/tmf921_dataset_gen/graph/workflow.py`
- Agents:
  - Diversity: `src/tmf921_dataset_gen/agents/diversity.py`
  - Translator: `src/tmf921_dataset_gen/agents/translator.py`
  - Critic: `src/tmf921_dataset_gen/agents/critic.py`
- Ingestion/parsing: `src/tmf921_dataset_gen/ingestion/`
- RAG: `src/tmf921_dataset_gen/rag/`
- Export: `src/tmf921_dataset_gen/export/`
- Validation/scoring: `src/tmf921_dataset_gen/validation/`

## Repository Layout

- `src/`: implementation code
- `tests/`: unit + integration tests
- `docs/`: project documentation and methodology
- `scripts/`: operational helper scripts
- `artifacts/`: normalized assets, indexes, reports
- `seeds/`: synthetic seed records
- `tr290-docs/`: TR290 source files
- `idan-reference/`: IDAN reference submodule (recommended; required for large sample runs)

## Supported Backends

### 1. `mock` backend

- Fast deterministic generation path.
- Best for bulk test generation and CI-like workflows.

### 2. `local-transformers` backend

- Full local LLM path (no external model API required).
- Typical profile in this repo: Qwen 3.5 9B for both reasoning and bulk roles.

## Installation

### Base environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### GPU/local-transformers extras

Install a CUDA-compatible PyTorch wheel first, then GPU extras:

```bash
# Example for CUDA 12.1 wheels
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
python -m pip install -r requirements-local-gpu.txt
```

## Configuration

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Most important variables:

- `INFERENCE_BACKEND` (`mock` or `local-transformers`)
- `REASONING_MODEL`, `BULK_MODEL`, `EMBEDDING_MODEL`
- `LOCAL_REASONING_MODEL_PATH`, `LOCAL_BULK_MODEL_PATH`, `LOCAL_EMBEDDING_MODEL_PATH`
- `LOCAL_FILES_ONLY`, `LOCAL_DEVICE`, `LOCAL_DTYPE`
- `FAST_MODE`
- `ENABLE_LLM_REWRITE`, `ENABLE_LLM_TRANSLATION_HINTS`, `ENABLE_LLM_SEMANTIC_REVIEW`

Recommended local GPU profile:

```bash
INFERENCE_BACKEND=local-transformers
REASONING_MODEL=Qwen/Qwen3.5-9B
BULK_MODEL=Qwen/Qwen3.5-9B
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LOCAL_REASONING_MODEL_PATH=/home/user/work/codex-dataset/models/Qwen3.5-9B
LOCAL_BULK_MODEL_PATH=/home/user/work/codex-dataset/models/Qwen3.5-9B
LOCAL_EMBEDDING_MODEL_PATH=
LOCAL_FILES_ONLY=false
LOCAL_DEVICE=cuda:0
LOCAL_DTYPE=bfloat16
LOCAL_MAX_NEW_TOKENS=1024
LOCAL_PLANNING_MAX_NEW_TOKENS=160
LOCAL_TOP_P=0.9
LOCAL_USE_4BIT=false
FAST_MODE=true
ENABLE_LLM_REWRITE=false
ENABLE_LLM_TRANSLATION_HINTS=true
ENABLE_LLM_SEMANTIC_REVIEW=false
```

## Core Commands

Run from repository root:

```bash
source .venv/bin/activate
python -m tmf921_dataset_gen.cli preflight
python -m tmf921_dataset_gen.cli build-corpus
python -m tmf921_dataset_gen.cli generate --count 1000 --output output/production_generate
python -m tmf921_dataset_gen.cli sample --count 1000 --out output/production_sample
python -m tmf921_dataset_gen.cli dashboard
```

Command notes:

- `preflight`: validates required resources and local backend prerequisites.
- `build-corpus`: builds/refreshes normalized corpus artifacts.
- `generate`: general generation entrypoint.
- `sample`: grounded sample generation path.
- `dashboard`: launches Streamlit dashboard.

Large-sample guardrail:

- `sample --count >= 1000` is intentionally blocked when `idan-reference/` is missing.

## Batch Helpers

Location: `scripts/generation/`

Mock backend 10k:

```bash
INFERENCE_BACKEND=mock python scripts/generation/generate_10k.py
INFERENCE_BACKEND=mock python scripts/generation/run_daemon.py
```

Single batch helper:

```bash
python scripts/generation/generate_batch.py 20 output/gpu_test
```

GPU shell workflow:

```bash
bash scripts/generation/run_generate_10k.sh
```

## Output Artifacts

Typical run output directory contains:

- `dataset.jsonl`: one JSON object per line
- `manifest.json`: run metadata + aggregate quality metrics
- optional `hf_dataset/` (if `datasets` export path is available)

Global report file:

- `artifacts/reports/generation_report.json`

## Reproducibility Notes

- Canonical schema source: `resources/tmf921/TMF921_Intent_Management_v5.0.0.oas.yaml`
- Postman examples source: `resources/tmf921/intent_management.postman_collection.json`
- TR290 extraction source: `tr290-docs/`
- Seeds source: `seeds/seeds.jsonl`
- Optional high-value grounding source: `idan-reference/`

The manifest captures:

- configured embedding model
- effective embedding backend/model at runtime
- model/backend metadata
- aggregate quality metrics

## Docker

CPU/dev image:

```bash
docker build -t tmf921-dataset-gen .
```

GPU image:

```bash
docker build -f Dockerfile.gpu -t tmf921-dataset-gen-gpu .
docker run --gpus all --rm -it \
  -v /models:/models \
  --env-file .env \
  tmf921-dataset-gen-gpu \
  python -m tmf921_dataset_gen.cli generate --count 20
```

## Testing

```bash
./.venv/bin/pytest -q tests/unit
./.venv/bin/pytest -q tests/integration
```

## Quality Controls

- schema validation
- semantic alignment scoring
- realism scoring
- TIO compliance scoring
- diversity and bias reporting

## Troubleshooting

- Check `preflight` first for missing resources and backend prerequisites.
- For local-transformers, verify model paths + CUDA availability.
- For long runs, use detached execution (`nohup`) and monitor logs in output directory.
- Historical issue log: `CHANGELOG.md`
