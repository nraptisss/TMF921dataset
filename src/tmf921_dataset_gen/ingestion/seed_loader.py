from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from ..config import Settings


TOP_LEVEL_KEYS = {"id", "nl_intent", "tmf921_intent", "notes"}


def _parse_seed_line(line: str) -> dict[str, Any]:
    try:
        return json.loads(line)
    except JSONDecodeError as exc:
        if exc.msg != "Extra data":
            raise

    prefix, index = json.JSONDecoder().raw_decode(line)
    tail = line[index:].strip()
    if tail.startswith(","):
        tail_payload = json.loads("{" + tail.lstrip(", "))
        prefix.update(tail_payload)

    intent_payload = dict(prefix.get("tmf921_intent", {}))
    leaked_keys = [key for key in prefix.keys() if key not in TOP_LEVEL_KEYS]
    for key in leaked_keys:
        intent_payload[key] = prefix.pop(key)
    prefix["tmf921_intent"] = intent_payload
    return prefix


def load_seed_records(settings: Settings) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    raw_lines = settings.repo.seeds_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        payload = _parse_seed_line(line)
        serialization = "json-ld"
        expression_type = payload.get("tmf921_intent", {}).get("expression", {}).get("@type")
        if expression_type == "TurtleExpression":
            serialization = "turtle"
        seed_id = payload.get("id") or f"seed-{index:03d}"
        rows.append(
            {
                "id": seed_id,
                "source_type": "seed",
                "title": payload.get("nl_intent", f"Seed {index}"),
                "text": json.dumps(payload, indent=2),
                "metadata": {
                    "seed_id": seed_id,
                    "serialization": serialization,
                    "notes": payload.get("notes"),
                },
                "nl_intent": payload.get("nl_intent"),
                "tmf921_intent": payload.get("tmf921_intent"),
            }
        )
    normalized_dir = settings.repo.normalized_dir / "seeds"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    (normalized_dir / "seeds.normalized.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows
