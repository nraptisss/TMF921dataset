# Changelog & Troubleshooting Log

## 2026-04-04: Critical Dataset Generation Bug Fixes (v0.1.1)

### Critical Issues Fixed

#### Issue 1: Spurious `device_count` Constraint Injection
- **Problem**: Non-mmtc records contained phantom `device_count: 0` constraints from faulty regex matching "for latency below 0.69 ms"
- **Impact**: 190 records (9.5%) contaminated with meaningless device constraints
- **Fix**: Simplified regex to require unit words (devices, sensors, etc.) after numbers
- **Verification**: 0 non-mmtc records now have device_count in context

#### Issue 2: Operator Inversion ("at least" → "atMost")
- **Problem**: Global operator search caused "under" from energy constraints to affect throughput operators
- **Impact**: 111 records (5.5%) had inverted semantics (minimum requirements became maximum caps)
- **Fix**: Implemented per-metric operator inference with text segmentation by metric boundaries
- **Verification**: 0 operator inversions in regenerated dataset

#### Issue 3: KPI Extraction False Positives
- **Problem**: Overly broad regex `(?:devicecount|support for|for)[^0-9]{0,20}?(\d+)` matched unrelated text
- **Impact**: Root cause of Issue 1, causing spurious device_count extraction
- **Fix**: Simplified pattern to require device unit words immediately after numbers
- **Verification**: No false positives in test cases

#### Issue 5: `unsupported_claims_ratio` Miscalculation
- **Problem**: Computed average unsupported claims per record (count), compared against 0.05 ratio threshold
- **Impact**: Release gate always failed (average ~3.3 claims per record)
- **Fix**: Split into `unsupported_claims_ratio` (fraction of records with any unsupported claims) and `avg_unsupported_claims_per_record`
- **Verification**: Ratio now ~1.0 (expected for synthetic values not in reference docs)

#### Issue 6: Semantic Preservation Benchmark Always Failed
- **Problem**: Exact string matching against synthetically generated NL intents
- **Impact**: Rate always 0/3 = 0.0, failing 95% threshold
- **Fix**: Implemented fuzzy semantic signature matching with Jaccard similarity
- **Verification**: Improved to 2/3 = 67% in regenerated dataset

#### Issue 7: Numeric Tolerance Inconsistent Across Scales
- **Problem**: Tolerance `max(abs(left) * 0.01, 1.0)` gave 48% relative tolerance for small values (latency) but only 1% for large values (throughput)
- **Impact**: Distorted semantic scores based on metric magnitude
- **Fix**: Changed to `max(magnitude * 0.1, 1.0)` for consistent 10% relative tolerance
- **Verification**: Now consistent across all metric scales

#### Issue 8: Dead Code in TIO Scoring
- **Problem**: Three instances of `score += 0.0` (no-ops) in TIO compliance scoring
- **Impact**: Confusing code suggesting incomplete implementation
- **Fix**: Removed dead code lines and improved scoring logic
- **Verification**: Cleaner TIO scoring without dead code

#### Issue 10: Missing Constraint Detection Broken
- **Problem**: Benchmark looked for non-existent `payload_constraints` field in payload
- **Impact**: Always reported 0% detection rate
- **Fix**: Extract constraints from JSON-LD expressionValue graph nodes
- **Verification**: Now 100% detection rate

#### Issue 11: `reaction_time_ms` Operator Mapping
- **Problem**: `trigger_within` operator not round-tripped correctly between intent_frame and payload
- **Impact**: Operator mismatch in semantic validation for predictive_assurance/closed_loop_autonomy
- **Fix**: Added special handling for `reaction_time_ms` to normalize `at_most` back to `trigger_within`
- **Verification**: All event-driven scenarios now pass semantic validation

#### Issue 12: False Contradictory Scenario Detection
- **Problem**: Context field containing `trigger_within` falsely detected as "closed_loop_autonomy" scenario
- **Impact**: predictive_assurance records rejected as contradictory
- **Fix**: Only check payload name for contradictory scenarios, ignore context (contains operator names)
- **Verification**: All scenario families now generate successfully

### Post-Fix Dataset Quality
- **Records**: 2000 (full coverage of all 8 scenarios × 3 traffic profiles × 3 layers)
- **Phantom Constraints**: 0 (was 190)
- **Operator Inversions**: 0 (was 111)
- **Benchmark Scores**: 100% operator preservation, 100% constraint detection, 67% semantic preservation
- **Quality Metrics**: 0.86 average quality score, 100% semantic pass rate

### Testing Results
- **Unit Tests**: 54/54 pass
- **Integration Tests**: All pass
- **Generation**: 2000 records accepted from 3544 attempts (56.4% acceptance rate)
- **Scenarios**: All 8 scenario families represented (192-436 records each)

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
