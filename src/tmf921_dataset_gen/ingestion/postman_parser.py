from __future__ import annotations

import json
from typing import Any

from ..config import Settings


def load_postman_collection(settings: Settings) -> dict[str, Any]:
    return json.loads(settings.repo.postman_collection.read_text(encoding="utf-8"))


def _walk_items(items: list[dict[str, Any]], path: list[str], results: list[dict[str, Any]]) -> None:
    for item in items:
        current_path = [*path, item.get("name", "unnamed")]
        if "request" in item:
            request = item["request"]
            body = request.get("body", {}) if isinstance(request.get("body"), dict) else {}
            results.append(
                {
                    "name": item.get("name"),
                    "path": current_path,
                    "method": request.get("method"),
                    "url": request.get("url", {}).get("raw"),
                    "body_mode": body.get("mode"),
                    "body_raw": body.get("raw"),
                    "response_count": len(item.get("response", [])),
                    "responses": [
                        {
                            "name": response.get("name"),
                            "code": response.get("code"),
                            "body": response.get("body"),
                        }
                        for response in item.get("response", [])
                    ],
                }
            )
        if "item" in item:
            _walk_items(item["item"], current_path, results)


def extract_postman_assets(settings: Settings) -> list[dict[str, Any]]:
    collection = load_postman_collection(settings)
    operations: list[dict[str, Any]] = []
    _walk_items(collection.get("item", []), [], operations)
    normalized_dir = settings.repo.normalized_dir / "postman"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    (normalized_dir / "operations.json").write_text(json.dumps(operations, indent=2), encoding="utf-8")

    corpus: list[dict[str, Any]] = []
    for operation in operations:
        text_parts = [
            f"Method: {operation['method']}",
            f"URL: {operation['url']}",
            f"Path: {' / '.join(operation['path'])}",
        ]
        if operation.get("body_raw"):
            text_parts.append(operation["body_raw"])
        for response in operation["responses"][:2]:
            if response.get("body"):
                text_parts.append(response["body"])
        corpus.append(
            {
                "id": f"postman:{operation['method']}:{'/'.join(operation['path'])}",
                "source_type": "postman_operation",
                "title": operation["name"],
                "text": "\n".join(text_parts),
                "metadata": {
                    "method": operation["method"],
                    "url": operation["url"],
                },
            }
        )
    return corpus
