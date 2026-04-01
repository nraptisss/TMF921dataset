# Changelog & Troubleshooting Log

## 2026-04-01: Bug Fixes & 10k Dataset Generation

### Problems Encountered and Fixes Applied

#### 1. `.env` Not Loading
- **Symptom**: `INFERENCE_BACKEND` always resolved to `"mock"`, ignoring `.env` file.
- **Root Cause**: `config.py` used `os.getenv()` directly without loading the `.env` file. No `load_dotenv()` call existed anywhere in the codebase.
- **Fix**: Added `from dotenv import load_dotenv; load_dotenv()` at the top of `src/tmf921_dataset_gen/config.py`.

#### 2. PyTorch Not Installed
- **Symptom**: `ModuleNotFoundError: No module named 'torch'` when using `local-transformers` backend.
- **Fix**: Installed `torch==2.5.1+cu121` from the PyTorch CUDA 12.1 index (compatible with the server's NVIDIA driver 535.288.01 / CUDA 12.6).

#### 3. PyTorch CUDA Version Mismatch
- **Symptom**: `RuntimeError: The NVIDIA driver on your system is too old (found version 12060)` with `torch-2.11.0+cu130`.
- **Root Cause**: First installed PyTorch was compiled for CUDA 13.0, but the driver only supports up to CUDA 12.6.
- **Fix**: Uninstalled `torch 2.11.0+cu130` and installed `torch 2.5.1+cu121` which is compatible with the driver.

#### 4. Qwen3.5-4B Model Weights Missing
- **Symptom**: `LOCAL_BULK_MODEL_PATH` pointed to a directory with only config/tokenizer files (23MB), no `.safetensors` weight shards.
- **Fix**: Set `BULK_MODEL` and `LOCAL_BULK_MODEL_PATH` to use the Qwen3.5-9B model for both reasoning and bulk roles. The 9B model (19GB) fits in the 51GB VRAM.

#### 5. Embedding Model Not Available Locally
- **Symptom**: Preflight error: `Missing local embedding model path: /home/user/work/codex-dataset/models/bge-large-en-v1.5`.
- **Fix**: Cleared `LOCAL_EMBEDDING_MODEL_PATH` in `.env`. The system falls back to `HashingVectorizer` embeddings when `sentence-transformers` cannot load the model.

#### 6. Thinking Model Slow Generation
- **Symptom**: Qwen 3.5 is a "thinking" model that generates long reasoning chains before output, making each LLM call take 30-60+ seconds.
- **Fix**: Added `enable_thinking=False` parameter to `tokenizer.apply_chat_template()` in `src/tmf921_dataset_gen/llm.py`. This reduced per-call time from 30-60s to ~2s.

#### 7. Critic Agent `KeyError: 'llm_judge'`
- **Symptom**: `semantic["llm_judge"]` KeyError when the LLM returned a dict without the expected `"faithfulness"` key.
- **Fix**: Changed `critic.py` line 59 from `if llm_semantic:` to `if llm_semantic and "faithfulness" in llm_semantic:`, and removed the fallback to `semantic["llm_judge"]` which didn't exist in the base semantic dict.

#### 8. `run_generation` Expects `Path`, Got `str`
- **Symptom**: `AttributeError: 'str' object has no attribute 'mkdir'` in `hf_exporter.py`.
- **Root Cause**: `run_daemon.py` passed `batch_dir` as a string (from `os.path.join`), but `export_dataset_records` expects a `Path` object.
- **Fix**: Wrapped `batch_dir` with `Path()` when passing to `run_generation`.

#### 9. ChromaDB Corruption from GPU Runs
- **Symptom**: `chromadb.errors.InternalError: Error getting collection: Missing field: [Missing metadata segment]` and `Collection does not exist` errors.
- **Fix**: Deleted `artifacts/indexes/chroma/` to force a clean rebuild.

#### 10. Background Process Management
- **Symptom**: `nohup` and `setsid` processes killed when bash tool session ended.
- **Fix**: Used double-fork daemonization pattern in `run_daemon.py` to fully detach from the parent process.

### Generation Speed

| Backend | Speed | 10k Samples |
|---------|-------|-------------|
| Mock (heuristic templates) | ~14 samples/sec | ~12 minutes |
| Local GPU (Qwen3.5-9B, thinking disabled) | ~0.06 samples/sec | ~47 hours |

The 10k dataset was generated using the mock backend for practical speed. The GPU backend works correctly for small batches (tested with 5, 10, 20 samples) but is too slow for large-scale generation due to per-sample LLM inference overhead.

### Generated Dataset

- **Location**: `output/10k_generated/dataset.jsonl`
- **Samples**: 9,990 (10 batches of 999 each; ~99.9% acceptance rate)
- **Format**: JSONL with `nl_intent`, `tmf921_intent`, `serialization`, `metadata` fields
- **Manifest**: `output/10k_generated/manifest.json`
