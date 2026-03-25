from __future__ import annotations

from typing import Any

from ..config import Settings
from .idan_loader import load_idan_documents
from .oas_parser import extract_oas_assets
from .postman_parser import extract_postman_assets
from .seed_loader import load_seed_records
from .tr290_extractor import extract_tr290_documents


def build_corpus(settings: Settings) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []
    corpus.extend(extract_oas_assets(settings))
    corpus.extend(extract_postman_assets(settings))
    corpus.extend(load_seed_records(settings))
    corpus.extend(extract_tr290_documents(settings))
    corpus.extend(load_idan_documents(settings))
    return corpus
