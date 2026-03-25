# TMF921 Dataset Generator

Synthetic, production-oriented pipeline for generating natural-language intents paired with TMF921 v5.0 `Intent_FVO` payloads.

## What It Builds

- TMF921/OpenAPI schema registry and normalized Postman operation corpus
- TR290 document extraction to Markdown for semantic grounding
- Seed and optional IDAN reference normalization
- Chroma-backed RAG with embedding fallback for offline/local runs
- LangGraph workflow with `DiversityAgent`, `TranslatorAgent`, and `CriticRefinementAgent`
- Export to JSONL plus Hugging Face `Dataset.save_to_disk()` when the `datasets` package is installed
- Optional Streamlit dashboard for browsing generated pairs

## Repository Layout

- `src/tmf921_dataset_gen/`: application code
- `tests/`: unit and integration tests
- `artifacts/`: normalized schemas, corpora, indexes, and reports
- `output/test_dataset/`: sample-run output

## Installation

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and adjust these fields as needed:

- `INFERENCE_BACKEND=mock|openai|anthropic|together|vllm`
- `REASONING_MODEL`
- `BULK_MODEL`
- `EMBEDDING_MODEL`
- `TARGET_PAIR_COUNT`
- `BATCH_SIZE`
- `JSONLD_RATIO`
- `MAX_REFINEMENT_LOOPS`

`mock` is the safest default for offline/local verification. In mock mode the pipeline uses deterministic heuristic generation and hashing embeddings to avoid remote model downloads.

## Commands

```bash
python -m tmf921_dataset_gen.cli preflight
python -m tmf921_dataset_gen.cli build-corpus
python -m tmf921_dataset_gen.cli generate --count 100
python -m tmf921_dataset_gen.cli sample --count 1000 --out output/test_dataset
python -m tmf921_dataset_gen.cli dashboard
```

## Reproducibility Notes

- Canonical schema validation is sourced from `TMF921_Intent_Management_v5.0.0.oas.yaml`.
- Postman is used for endpoint/example alignment and hub payload coverage.
- If `tr290-docs/*.md` is absent, Markdown is generated from the bundled DOCX/PDF into `artifacts/normalized/tr290/`.
- If `idan-reference/` is absent, `sample --count 1000` is intentionally blocked and writes `output/test_dataset/blocked_run_manifest.json` instead of producing a weakly grounded large sample.
- If the `datasets` package is unavailable, exports still write `dataset.jsonl` and `manifest.json`; Hugging Face dataset export is skipped gracefully.

## Testing

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```

## Docker

```bash
docker build -t tmf921-dataset-gen .
docker run --rm -it tmf921-dataset-gen python -m tmf921_dataset_gen.cli generate --count 20
```

## Dashboard

The Streamlit dashboard reads `output/test_dataset/manifest.json` and `output/test_dataset/dataset.jsonl` by default.

```bash
python -m tmf921_dataset_gen.cli dashboard
```
