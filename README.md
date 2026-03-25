# TMF921 Dataset Generator

Synthetic, production-oriented pipeline for generating natural-language intents paired with TMF921 v5.0 `Intent_FVO` payloads.

## What Changed For Local GPU Servers

This project can now run fully locally without OpenAI, Anthropic, Together, or any other external model API.

- `INFERENCE_BACKEND=local-transformers` runs reasoning and bulk generation with local Hugging Face models loaded directly on the server.
- `LOCAL_REASONING_MODEL_PATH`, `LOCAL_BULK_MODEL_PATH`, and `LOCAL_EMBEDDING_MODEL_PATH` let you pin the exact on-disk model artifacts.
- Preflight now validates local model paths and CUDA/runtime prerequisites.
- The agents will use local models when configured, but still retain deterministic heuristic fallbacks so the pipeline remains testable offline.

## Recommended RTX 6000 Ada 50GB Profile

Recommended local model split:

- Reasoning model: `Qwen/Qwen2.5-14B-Instruct`
- Bulk generation model: `Qwen/Qwen2.5-7B-Instruct`
- Embeddings: `BAAI/bge-large-en-v1.5`
- Precision: `bfloat16`
- Quantization: optional 4-bit if you want to reduce VRAM pressure further on Linux

This fits well on a 50 GB Ada card when you run one generation model at a time.

## Installation

### Base environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Local GPU extras

Install PyTorch with the CUDA wheel that matches your server first, then install the GPU extras:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install -r requirements-local-gpu.txt
```

## Local Server Configuration

Copy `.env.example` to `.env` and point it at local model directories. Example:

```bash
INFERENCE_BACKEND=local-transformers
REASONING_MODEL=Qwen/Qwen2.5-14B-Instruct
BULK_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LOCAL_REASONING_MODEL_PATH=/models/Qwen2.5-14B-Instruct
LOCAL_BULK_MODEL_PATH=/models/Qwen2.5-7B-Instruct
LOCAL_EMBEDDING_MODEL_PATH=/models/bge-large-en-v1.5
LOCAL_FILES_ONLY=true
LOCAL_DEVICE=cuda:0
LOCAL_DTYPE=bfloat16
LOCAL_MAX_NEW_TOKENS=1024
LOCAL_TOP_P=0.9
LOCAL_USE_4BIT=false
```

`LOCAL_FILES_ONLY=true` ensures the runtime never tries to fetch model weights from the network. Pre-download the models to the mounted paths above.

## Commands

Run directly from the repo checkout:

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

## Docker

General CPU/dev image:

```bash
docker build -t tmf921-dataset-gen .
```

GPU server image:

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
python -m pytest tests/unit -q
python -m pytest tests/integration -q
```
