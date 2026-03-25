from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


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

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        repo_root = Path(root or Path.cwd()).resolve()
        repo = RepoPaths.from_root(repo_root)
        vector_index = Path(_env("VECTOR_INDEX_DIR", str(repo.indexes_dir / "chroma")))
        if not vector_index.is_absolute():
            vector_index = repo_root / vector_index
        return cls(
            repo=repo,
            reasoning_model=_env("REASONING_MODEL", "claude-4-sonnet"),
            bulk_model=_env("BULK_MODEL", "llama-4-70b"),
            embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            vector_db=_env("VECTOR_DB", "chroma"),
            inference_backend=_env("INFERENCE_BACKEND", "mock"),
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
        )
