from pathlib import Path

from tmf921_dataset_gen.agents.critic import CriticRefinementAgent
from tmf921_dataset_gen.agents.diversity import DiversityAgent
from tmf921_dataset_gen.agents.translator import TranslatorAgent
from tmf921_dataset_gen.config import Settings
from tmf921_dataset_gen.rag.embeddings import HashingEmbeddingModel
from tmf921_dataset_gen.rag.retriever import BalancedRetriever
from tmf921_dataset_gen.taxonomy.quota_planner import QuotaPlanner
from tmf921_dataset_gen.validation.diversity_metrics import detect_bias


def test_quota_planner_covers_taxonomy_space() -> None:
    settings = Settings.from_env(Path.cwd())
    planner = QuotaPlanner(settings)
    combos = planner.combinations()
    assert combos
    assert any(combo.taxonomy_category.startswith("business/urllc") for combo in combos)
    targets = planner.plan_targets(25)
    assert len(targets) == 25


def test_diversity_agent_generates_distinct_intents() -> None:
    settings = Settings.from_env(Path.cwd())
    agent = DiversityAgent(settings)
    planner = QuotaPlanner(settings)
    targets = planner.plan_targets(2)
    first = agent.generate(targets[0], 0)
    second = agent.generate(targets[1], 1)
    assert first["nl_intent"] != second["nl_intent"]
    assert first["metadata"]["seed_id"]
    assert first["metadata"]["taxonomy_target"]["taxonomy_category"] == targets[0]["taxonomy_category"]


def test_translator_generates_schema_valid_intent(tmp_path: Path) -> None:
    settings = Settings.from_env(Path.cwd())
    settings.vector_index_dir = tmp_path / "translator-index"
    retriever = BalancedRetriever(settings, embedder=HashingEmbeddingModel())
    retriever.build()
    planner = QuotaPlanner(settings)
    target = planner.plan_targets(1)[0]
    diversity = DiversityAgent(settings)
    candidate = diversity.generate(target, 0)
    context = retriever.retrieve(candidate["nl_intent"], top_k=6, per_source=1)
    translator = TranslatorAgent(settings)
    translated = translator.translate(
        candidate["nl_intent"],
        target,
        context,
        candidate["seed_ids"],
        0,
        source_kpis=candidate["metadata"]["kpis"],
    )
    assert translated["serialization"] in {"json-ld", "turtle"}
    assert translated["tmf921_intent"]["expression"]["@type"] in {"JsonLdExpression", "TurtleExpression"}
    assert translated["metadata"]["taxonomy_target"]["layer"] == target["layer"]


def test_detect_bias_uses_taxonomy_targets_when_metadata_is_missing() -> None:
    records = [{"metadata": {"taxonomy_category": "service/urllc/energy"}}]
    taxonomy_targets = [{"layer": "service", "traffic_profile": "urllc", "scenario_family": "energy"}]
    report = detect_bias(records, taxonomy_targets)
    assert report["bias_score"] == 0.0


def test_critic_repairs_invalid_payload() -> None:
    settings = Settings.from_env(Path.cwd())
    planner = QuotaPlanner(settings)
    target = planner.plan_targets(1)[0]
    critic = CriticRefinementAgent(settings)
    report = critic.review(
        nl_intent="Ensure URLLC latency below 1 ms with reliability of 99.999%.",
        payload={"name": "broken"},
        serialization="json-ld",
        taxonomy_target=target,
        retrieved_context=[],
        refinement_count=0,
    )
    assert report["accepted"] is False
    assert "expression" in report["repaired_payload"]
