from pathlib import Path

from tmf921_dataset_gen.agents.diversity import DiversityAgent
from tmf921_dataset_gen.agents.translator import TranslatorAgent
from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.llm import _extract_json_object
from tmf921_dataset_gen.preflight import run_preflight


def test_preflight_flags_missing_local_model_paths(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "local-transformers")
    monkeypatch.setenv("LOCAL_REASONING_MODEL_PATH", "missing/reasoning")
    monkeypatch.setenv("LOCAL_BULK_MODEL_PATH", "missing/bulk")
    settings = Settings.from_env(Path.cwd())
    report = run_preflight(settings)
    assert any("Missing local reasoning model path" in error for error in report.errors)
    assert any("Missing local bulk model path" in error for error in report.errors)


def test_extract_json_object_handles_fenced_json() -> None:
    payload = _extract_json_object("```json\n{\"faithfulness\": 0.9}\n```")
    assert payload["faithfulness"] == 0.9


def test_diversity_agent_uses_llm_rewrite_when_available(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LLM_REWRITE", "true")
    settings = Settings.from_env(Path.cwd())
    agent = DiversityAgent(settings)
    monkeypatch.setattr(type(agent.router), "supports_generation", lambda self: True)
    monkeypatch.setattr(type(agent.router), "generate_json", lambda self, *args, **kwargs: {"nl_intent": "LLM rewritten intent"})
    result = agent.generate({"taxonomy_category": "service/urllc/energy", "layer": "service", "traffic_profile": "urllc", "scenario_family": "energy", "domain_context": "private campus"}, 0)
    assert result["nl_intent"] == "LLM rewritten intent"


def test_translator_applies_llm_translation_options(monkeypatch) -> None:
    settings = Settings.from_env(Path.cwd())
    translator = TranslatorAgent(settings)
    monkeypatch.setattr(type(translator.router), "supports_generation", lambda self: True)
    monkeypatch.setattr(
        type(translator.router),
        "generate_json",
        lambda self, *args, **kwargs: {"name": "Local Slice Intent", "context": "server-local", "serialization": "turtle", "priority": "medium"},
    )
    result = translator.translate(
        nl_intent="Ensure URLLC latency below 1 ms with reliability 99.999%.",
        taxonomy_target={"taxonomy_category": "service/urllc/predictive_assurance", "layer": "service", "traffic_profile": "urllc", "scenario_family": "predictive_assurance"},
        retrieved_context=[],
        seed_ids=["seed-001"],
        sample_index=1,
        source_kpis={"latency_ms": 1.0, "reliability_percent": 99.999},
    )
    assert result["serialization"] == "turtle"
    assert result["tmf921_intent"]["name"] == "Local Slice Intent"
    assert result["tmf921_intent"]["context"] == "server-local"
