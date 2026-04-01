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

- Reasoning model: `Qwen/Qwen3.5-9B-Instruct`
- Bulk generation model: `Qwen/Qwen3.5-4B-Instruct`
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

Install PyTorch with the CUDA wheel that matches your server driver first, then install the GPU extras:

```bash
# Check your driver's max CUDA version: nvidia-smi | grep "CUDA Version"
# Driver 535.x supports up to CUDA 12.6
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
python -m pip install -r requirements-local-gpu.txt
```

## Local Server Configuration

Copy `.env.example` to `.env` and point it at local model directories. Example:

```bash
INFERENCE_BACKEND=local-transformers
REASONING_MODEL=Qwen/Qwen3.5-9B
BULK_MODEL=Qwen/Qwen3.5-9B
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LOCAL_REASONING_MODEL_PATH=/home/user/work/codex-dataset/models/Qwen3.5-9B
LOCAL_BULK_MODEL_PATH=/home/user/work/codex-dataset/models/Qwen3.5-9B
LOCAL_EMBEDDING_MODEL_PATH=
LOCAL_FILES_ONLY=true
LOCAL_DEVICE=cuda:0
LOCAL_DTYPE=bfloat16
LOCAL_MAX_NEW_TOKENS=1024
LOCAL_TOP_P=0.9
LOCAL_USE_4BIT=false
```

> **Note**: The Qwen3.5-4B model weights are not bundled in this repo. Use the 9B model for both roles — it fits comfortably on an RTX 6000 Ada (51 GB VRAM). Clear `LOCAL_EMBEDDING_MODEL_PATH` to fall back to `HashingVectorizer` embeddings when `BAAI/bge-large-en-v1.5` is not available locally.

`LOCAL_FILES_ONLY=true` ensures the runtime never tries to fetch model weights from the network. Pre-download the models to the mounted paths above.

## Commands

Run directly from the repo checkout:

```bash
source .venv/bin/activate
python -m tmf921_dataset_gen.cli preflight
python -m tmf921_dataset_gen.cli build-corpus
python -m tmf921_dataset_gen.cli generate --count 1000
python -m tmf921_dataset_gen.cli sample --count 1000 --out output/production_dataset
python -m tmf921_dataset_gen.cli dashboard
```

### Batch generation (10k samples)

For large-scale generation, use the daemon script which runs 10 batches of 1000 samples:

```bash
source .venv/bin/activate
# Mock backend (fast, ~12 min for 10k):
INFERENCE_BACKEND=mock python run_daemon.py

# Or use the non-daemon script for foreground runs:
INFERENCE_BACKEND=mock python generate_10k.py
```

Output goes to `output/10k_generated/` with individual batches in `batch_1/` through `batch_10/`.

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

## Troubleshooting

See [CHANGELOG.md](CHANGELOG.md) for a detailed log of problems encountered and fixes applied during setup and generation.
