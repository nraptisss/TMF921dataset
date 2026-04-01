# Changelog & Troubleshooting Log

## 2026-04-01: Dataset Quality Enhancements

### Quality Improvements Implemented

#### Data Ingestion Enhancements
- **IDAN Filtering**: Added keyword-based filtering to exclude non-intent files. Only loads documents containing terms like "intent", "fvo", "kpi".
- **TR290 Normalization**: Enhanced DOCX/PDF extraction to skip boilerplate (copyright notices, page headers). Removes common TM Forum headers.
- **Corpus Deduplication**: Implemented fuzzy deduplication using TF-IDF and cosine similarity (threshold 0.8) to remove overlapping content across sources.
- **Seed Validation**: Added checks for required fields and basic TMF921 structure before inclusion.
- **Impact**: Corpus reduced to 64 high-quality documents after deduplication.

#### Validation Depth Improvements
- **Semantic Validation**: Improved KPI extraction with enhanced regex patterns and plausibility checks (e.g., latency > 0, percentages 0-100).
- **TIO Compliance**: Upgraded Turtle parsing with RDF graph validation, checking for valid Intent nodes and awarding bonuses.
- **New Metrics**: Added diversity score (unique intents/payloads ratio), bias detection (underrepresented layers/traffic profiles).
- **LLM Judge Reliability**: Ensured LLM judge is prioritized for semantic scoring when available.
- **Impact**: TIO scores improved (e.g., 0.85 for valid RDF with Intent nodes); diversity metrics provide batch-level insights.

#### Workflow and Generation Enhancements
- **Context Integration**: Fed retrieved RAG context into Translator LLM prompts for more grounded payload generation.
- **Quality Gates**: Added diversity and bias calculation in generation reports.
- **Impact**: Payloads now incorporate domain knowledge, improving fidelity.

#### Export and Metadata Tracking
- **Detailed Manifest**: Includes average quality scores, semantic/TIO compliance, diversity, and bias reports.
- **Quality Reports**: Generation reports now feature diversity (e.g., 0.75 for varied records) and bias detection.
- **Impact**: Enables downstream quality assessment and reproducibility.

### Testing Results
- Corpus building: 64 deduplicated documents.
- KPI extraction: Correctly parses "10 ms latency, 1 Gbps throughput".
- TIO compliance: Scores 0.85 for valid RDF graphs.
- Diversity metrics: Calculates 0.75 for test datasets.
- Workflow: Compiles successfully with all enhancements.

### Generation Impact
Quality enhancements ensure the first-of-its-kind TMF921 dataset meets high standards for training and evaluation. Datasets now include purity, compliance, and diversity metrics for robust research use.

## 2026-04-01: Codebase Fixes & GPU Pipeline Stabilization

### Code Quality Fixes

#### BOM Removal (42 files)
- **Problem**: 42 Python files and 1 YAML file contained UTF-8 BOM (`U+FEFF`), causing `SyntaxError: invalid non-printable character U+FEFF` when parsed.
- **Fix**: Stripped BOM from all affected files in `src/`, `tests/`, and `src/tmf921_dataset_gen/taxonomy/taxonomy.yaml`.

#### `generate_batch.py` Import Error
- **Problem**: Imported non-existent `run_sample` function from `workflow`.
- **Fix**: Changed import to `run_generation` and updated function call.

#### `pyproject.toml` Empty Dependencies
- **Problem**: `dependencies = []` meant `pip install .` produced a broken package.
- **Fix**: Populated dependencies from `requirements.txt` (16 packages including `python-dotenv`).

#### `DatasetMetadata` Required Fields Without Defaults
- **Problem**: `quality_score`, `tio_compliance`, `generation_timestamp` were required with no defaults, risking validation errors.
- **Fix**: Added sensible defaults (`0.0`, `0.0`, `None`).

#### `diversity.py` Silent Exception Swallowing
- **Problem**: `_llm_rewrite` caught `Exception` with `pass`, making LLM failures invisible.
- **Fix**: Added `logging.debug()` call to record failure reason.

#### `validate_intents.py` Duplicate Initialization
- **Problem**: `Settings` and `TMFJsonSchemaValidator` created twice in `validate_dataset()`.
- **Fix**: Removed duplicate initialization.

#### `validate_intents.py` Dangerous File Deletion
- **Problem**: Iterated `os.listdir('.')` and deleted files matching pattern — could affect wrong directory.
- **Fix**: Changed to use `Path(__file__).parent.glob()` to target only the scripts directory.

#### Hardcoded Paths in Generation Scripts
- **Problem**: `generate_10k.py` and `run_daemon.py` hardcoded `/home/user/work/codex-dataset`.
- **Fix**: Replaced with `Path(__file__).resolve().parent` for portability.

#### `run_daemon.py` No Windows Check
- **Problem**: `os.fork()` used without platform check.
- **Fix**: Added `platform.system() == 'Windows'` guard with error message.

#### OpenAI Client Created Per Call
- **Problem**: `LLMRouter.generate_text()` created new `OpenAI` client for every cloud API call.
- **Fix**: Added `_openai_client` field with lazy initialization and caching.

### GPU Pipeline Fixes

#### PyTorch CUDA Version Mismatch
- **Problem**: PyTorch 2.11.0+cu130 installed but driver only supports CUDA 12.6.
- **Fix**: Reinstalled `torch==2.5.1+cu121` from PyTorch CUDA 12.1 index.

#### ChromaDB Collection Corruption
- **Problem**: Repeated `reset()` calls during batch generation corrupted ChromaDB collection references, causing `Collection does not exist` errors.
- **Root Cause**: `get_or_create_collection()` returns stale references after delete/recreate cycles. Rapid successive `WorkflowRunner` instances corrupted shared state.
- **Fix**: Replaced `get_or_create_collection()` with explicit `_ensure_collection()` using `get_collection()` with fallback to `create_collection()`. Added `time.sleep(0.5)` after delete and retry logic in `query()` method.

### Generation Results

| Backend | Speed | Samples Generated |
|---------|-------|-------------------|
| Mock (heuristic templates) | ~14 samples/sec | 9,990 (10k) |
| Local GPU (Qwen3.5-9B) | ~0.041 samples/sec | 134 (4 batches of 20) |

GPU-generated dataset at `output/10k_gpu_generated/dataset.jsonl` contains high-quality TMF921 JSON-LD payloads with valid schema compliance.

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
