from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, root: Path) -> Path | None:
    raw = os.getenv(name)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


@dataclass(slots=True)
class RepoPaths:
    root: Path
    postman_collection: Path
    oas_spec: Path
    tr290_dir: Path
    tr290_pdf: Path
    tr290_docx: Path
    seeds_path: Path
    idan_reference_dir: Path
    artifacts_dir: Path
    normalized_dir: Path
    indexes_dir: Path
    reports_dir: Path
    output_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "RepoPaths":
        artifacts_dir = root / "artifacts"
        return cls(
            root=root,
            postman_collection=root / "Intent Management.postman_collection.json",
            oas_spec=root / "TMF921_Intent_Management_v5.0.0.oas.yaml",
            tr290_dir=root / "tr290-docs",
            tr290_pdf=root / "tr290-docs" / "TR290_Intent_Common_Model_v3.0.0.pdf",
            tr290_docx=root / "tr290-docs" / "TR290_Intent_Common_Model_v3.0.0.docx",
            seeds_path=root / "seeds" / "seeds.jsonl",
            idan_reference_dir=root / "idan-reference",
            artifacts_dir=artifacts_dir,
            normalized_dir=artifacts_dir / "normalized",
            indexes_dir=artifacts_dir / "indexes",
            reports_dir=artifacts_dir / "reports",
            output_dir=root / "output",
        )


@dataclass(slots=True)
class Settings:
    repo: RepoPaths
    reasoning_model: str
    bulk_model: str
    embedding_model: str
    vector_db: str
    inference_backend: str
    target_pair_count: int
    batch_size: int
    jsonld_ratio: float
    max_refinement_loops: int
    random_seed: int
    vector_index_dir: Path
    openai_api_key: str | None
    anthropic_api_key: str | None
    together_api_key: str | None
    openai_base_url: str | None
    local_reasoning_model_path: Path | None
    local_bulk_model_path: Path | None
    local_embedding_model_path: Path | None
    local_files_only: bool
    local_device: str
    local_dtype: str
    local_max_new_tokens: int
    local_top_p: float
    local_use_4bit: bool
    local_trust_remote_code: bool
    fast_mode: bool
    enable_llm_rewrite: bool
    enable_llm_translation_hints: bool
    enable_llm_semantic_review: bool
    local_planning_max_new_tokens: int
    effective_embedding_model: str | None
    effective_embedding_backend: str | None

    @property
    def is_local_model_backend(self) -> bool:
        return self.inference_backend == "local-transformers"

    def resolve_generation_model(self, model: str) -> str:
        if model == self.reasoning_model and self.local_reasoning_model_path:
            return str(self.local_reasoning_model_path)
        if model == self.bulk_model and self.local_bulk_model_path:
            return str(self.local_bulk_model_path)
        return model

    def resolve_embedding_model(self) -> str:
        if self.local_embedding_model_path:
            return str(self.local_embedding_model_path)
        return self.embedding_model

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        repo_root = Path(root or Path.cwd()).resolve()
        repo = RepoPaths.from_root(repo_root)
        vector_index = Path(_env("VECTOR_INDEX_DIR", str(repo.indexes_dir / "chroma")))
        if not vector_index.is_absolute():
            vector_index = repo_root / vector_index
        inference_backend = _env("INFERENCE_BACKEND", "mock")
        fast_mode = _env_bool("FAST_MODE", inference_backend == "local-transformers")
        return cls(
            repo=repo,
            reasoning_model=_env("REASONING_MODEL", "claude-4-sonnet"),
            bulk_model=_env("BULK_MODEL", "llama-4-70b"),
            embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            vector_db=_env("VECTOR_DB", "chroma"),
            inference_backend=inference_backend,
            target_pair_count=_env_int("TARGET_PAIR_COUNT", 10000),
            batch_size=_env_int("BATCH_SIZE", 25),
            jsonld_ratio=_env_float("JSONLD_RATIO", 0.7),
            max_refinement_loops=_env_int("MAX_REFINEMENT_LOOPS", 2),
            random_seed=_env_int("RANDOM_SEED", 42),
            vector_index_dir=vector_index,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            local_reasoning_model_path=_env_path("LOCAL_REASONING_MODEL_PATH", repo_root),
            local_bulk_model_path=_env_path("LOCAL_BULK_MODEL_PATH", repo_root),
            local_embedding_model_path=_env_path("LOCAL_EMBEDDING_MODEL_PATH", repo_root),
            local_files_only=_env_bool("LOCAL_FILES_ONLY", True),
            local_device=_env("LOCAL_DEVICE", "cuda:0"),
            local_dtype=_env("LOCAL_DTYPE", "bfloat16"),
            local_max_new_tokens=_env_int("LOCAL_MAX_NEW_TOKENS", 1024),
            local_top_p=_env_float("LOCAL_TOP_P", 0.9),
            local_use_4bit=_env_bool("LOCAL_USE_4BIT", False),
            local_trust_remote_code=_env_bool("LOCAL_TRUST_REMOTE_CODE", False),
            fast_mode=fast_mode,
            enable_llm_rewrite=_env_bool("ENABLE_LLM_REWRITE", not fast_mode),
            enable_llm_translation_hints=_env_bool("ENABLE_LLM_TRANSLATION_HINTS", True),
            enable_llm_semantic_review=_env_bool("ENABLE_LLM_SEMANTIC_REVIEW", not fast_mode),
            local_planning_max_new_tokens=_env_int("LOCAL_PLANNING_MAX_NEW_TOKENS", 160),
            effective_embedding_model=None,
            effective_embedding_backend=None,
        )
