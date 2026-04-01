from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

import requests
from openai import OpenAI

from .config import Settings


JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    fence_match = JSON_FENCE_PATTERN.search(raw)
    if fence_match:
        return json.loads(fence_match.group(1))

    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    for index, character in enumerate(raw[start:], start=start):
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : index + 1])
    raise ValueError("Unbalanced JSON object in model response.")


@dataclass(slots=True)
class LocalTransformersEngine:
    settings: Settings
    _model_cache: ClassVar[dict[str, Any]] = {}
    _tokenizer_cache: ClassVar[dict[str, Any]] = {}

    def _dtype(self):
        import torch

        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": None,
        }
        return mapping.get(self.settings.local_dtype.lower(), torch.bfloat16)

    def _load_components(self, model_ref: str):
        if model_ref in self._model_cache and model_ref in self._tokenizer_cache:
            return self._tokenizer_cache[model_ref], self._model_cache[model_ref]

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.local_trust_remote_code,
            "local_files_only": self.settings.local_files_only,
        }
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.local_trust_remote_code,
            "local_files_only": self.settings.local_files_only,
        }
        torch_dtype = self._dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if self.settings.local_use_4bit:
            if not importlib.util.find_spec("bitsandbytes"):
                raise RuntimeError("LOCAL_USE_4BIT=true but bitsandbytes is not installed.")
            from transformers import BitsAndBytesConfig
            import torch

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["device_map"] = "auto"
        elif importlib.util.find_spec("accelerate") and self.settings.local_device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(model_ref, **tokenizer_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)

        if "device_map" not in model_kwargs:
            model.to(self.settings.local_device)
        model.eval()
        self._tokenizer_cache[model_ref] = tokenizer
        self._model_cache[model_ref] = model
        return tokenizer, model

    def generate_text(self, prompt: str, model: str, temperature: float = 0.4) -> str:
        import torch

        model_ref = self.settings.resolve_generation_model(model)
        tokenizer, model_obj = self._load_components(model_ref)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered_prompt = prompt
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        target_device = model_obj.device if hasattr(model_obj, "device") else self.settings.local_device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model_obj.generate(
                **inputs,
                max_new_tokens=self.settings.local_max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=self.settings.local_top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@dataclass(slots=True)
class LLMRouter:
    settings: Settings
    _local_engine: LocalTransformersEngine = field(init=False, repr=False)
    _openai_client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._local_engine = LocalTransformersEngine(self.settings)

    def supports_generation(self) -> bool:
        return self.settings.inference_backend != "mock"

    def generate_text(self, prompt: str, model: str, temperature: float = 0.4) -> str:
        if self.settings.inference_backend == "mock":
            return ""
        if self.settings.inference_backend == "local-transformers":
            return self._local_engine.generate_text(prompt=prompt, model=model, temperature=temperature)
        if self.settings.inference_backend in {"openai", "together", "vllm"}:
            if self._openai_client is None:
                self._openai_client = OpenAI(
                    api_key=self.settings.openai_api_key or self.settings.together_api_key or "unused",
                    base_url=self.settings.openai_base_url,
                )
            response = self._openai_client.chat.completions.create(
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
        return _extract_json_object(raw)
