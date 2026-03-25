from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

from .config import Settings


@dataclass(slots=True)
class LLMRouter:
    settings: Settings

    def generate_text(self, prompt: str, model: str, temperature: float = 0.4) -> str:
        if self.settings.inference_backend == "mock":
            return ""
        if self.settings.inference_backend in {"openai", "together", "vllm"}:
            client = OpenAI(
                api_key=self.settings.openai_api_key or self.settings.together_api_key or "unused",
                base_url=self.settings.openai_base_url,
            )
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        if self.settings.inference_backend == "anthropic":
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.settings.anthropic_api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 2048,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            parts = payload.get("content", [])
            return "".join(part.get("text", "") for part in parts if part.get("type") == "text")
        raise ValueError(f"Unsupported inference backend: {self.settings.inference_backend}")

    def generate_json(self, prompt: str, model: str, temperature: float = 0.2) -> dict[str, Any]:
        if self.settings.inference_backend == "mock":
            return {}
        raw = self.generate_text(prompt, model=model, temperature=temperature)
        return json.loads(raw)
