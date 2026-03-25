# TMF921 Dataset Generator

Production-oriented synthetic dataset generator for natural-language intents paired with TMF921 v5.0 `Intent_FVO` payloads.

## Status

This repository contains:

- TMF921/OpenAPI ingestion and schema validation
- TR290/seed/official-example corpus normalization
- Chroma-backed RAG retrieval
- LangGraph orchestration for diversity, translation, and critique
- Hugging Face dataset export with JSONL mirrors
- Optional Streamlit dashboard

## Quickstart

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and set your model credentials.
4. Run:

```bash
python -m tmf921_dataset_gen.cli build-corpus
python -m tmf921_dataset_gen.cli generate --count 100
```

`sample --count 1000` will refuse to run while `idan-reference/` is missing, because the implementation treats that repository as a grounding requirement for the larger validation run.

