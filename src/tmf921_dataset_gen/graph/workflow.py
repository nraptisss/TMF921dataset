from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..agents.critic import CriticRefinementAgent
from ..agents.diversity import DiversityAgent
from ..agents.translator import TranslatorAgent
from ..config import Settings
from ..export.hf_exporter import export_dataset_records
from ..models.dataset import DatasetMetadata, DatasetRecord
from ..models.state import GraphState
from ..rag.retriever import BalancedRetriever
from ..taxonomy.quota_planner import QuotaPlanner
from ..validation.diversity_metrics import calculate_diversity_score, detect_bias


class WorkflowRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever = BalancedRetriever(settings)
        self.retriever.build()
        self.diversity = DiversityAgent(settings)
        self.translator = TranslatorAgent(settings)
        self.critic = CriticRefinementAgent(settings, self.translator)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        def diversify_node(state: GraphState) -> GraphState:
            sample_index = state["taxonomy_target"].get("sample_index", 0)
            candidate = self.diversity.generate(state["taxonomy_target"], sample_index)
            return {
                **state,
                "nl_intent": candidate["nl_intent"],
                "seed_ids": candidate["seed_ids"],
                "metadata": candidate["metadata"],
                "refinement_count": state.get("refinement_count", 0),
            }

        def retrieve_node(state: GraphState) -> GraphState:
            context = self.retriever.retrieve(state["nl_intent"], top_k=8, per_source=1)
            return {**state, "retrieved_context": context}

        def translate_node(state: GraphState) -> GraphState:
            sample_index = state["taxonomy_target"].get("sample_index", 0)
            translated = self.translator.translate(
                state["nl_intent"],
                state["taxonomy_target"],
                state.get("retrieved_context", []),
                state.get("seed_ids", []),
                sample_index,
            )
            metadata = {**state.get("metadata", {}), **translated["metadata"]}
            return {
                **state,
                "tmf921_intent": translated["tmf921_intent"],
                "serialization": translated["serialization"],
                "metadata": metadata,
                "generation_timestamp": metadata.get("generation_timestamp", datetime.now(timezone.utc)),
            }

        def critique_node(state: GraphState) -> GraphState:
            report = self.critic.review(
                state["nl_intent"],
                state["tmf921_intent"],
                state["serialization"],
                state["taxonomy_target"],
                state.get("retrieved_context", []),
                state.get("refinement_count", 0),
            )
            metadata = {
                **state.get("metadata", {}),
                "quality_score": report["quality_score"],
                "tio_compliance": report["tio"]["score"],
                "schema_validity": report["schema_validity"],
                "realism_score": report["realism"]["score"],
                "semantic_score": report["semantic"]["score"],
                "validation_notes": report["notes"],
            }
            return {
                **state,
                "critic_report": report,
                "accepted": report["accepted"],
                "metadata": metadata,
            }

        def refine_node(state: GraphState) -> GraphState:
            return {
                **state,
                "tmf921_intent": state["critic_report"]["repaired_payload"],
                "refinement_count": state.get("refinement_count", 0) + 1,
            }

        workflow.add_node("diversify", diversify_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("translate", translate_node)
        workflow.add_node("critique", critique_node)
        workflow.add_node("refine", refine_node)
        workflow.add_edge(START, "diversify")
        workflow.add_edge("diversify", "retrieve")
        workflow.add_edge("retrieve", "translate")
        workflow.add_edge("translate", "critique")
        workflow.add_conditional_edges(
            "critique",
            lambda state: END if state.get("accepted") or state.get("refinement_count", 0) >= self.settings.max_refinement_loops else "refine",
            {END: END, "refine": "refine"},
        )
        workflow.add_edge("refine", "critique")
        return workflow.compile()

    def invoke(self, taxonomy_target: dict[str, Any]) -> GraphState:
        initial_state: GraphState = {
            "taxonomy_target": taxonomy_target,
            "refinement_count": 0,
        }
        return self.graph.invoke(initial_state)


def run_generation(settings: Settings, count: int, output_dir: Path | None) -> list[dict[str, Any]]:
    planner = QuotaPlanner(settings)
    runner = WorkflowRunner(settings)
    accepted_records: list[DatasetRecord] = []
    attempts = 0
    max_attempts = max(count * 3, count)
    taxonomy_targets = list(planner.plan_targets(max_attempts))
    for taxonomy_target in taxonomy_targets:
        state = runner.invoke(taxonomy_target)
        attempts += 1
        if not state.get("accepted"):
            continue
        metadata = DatasetMetadata.model_validate(state["metadata"])
        accepted_records.append(
            DatasetRecord(
                nl_intent=state["nl_intent"],
                tmf921_intent=state["tmf921_intent"],
                serialization=state["serialization"],
                metadata=metadata,
            )
        )
        if len(accepted_records) >= count:
            break
    payloads = [record.model_dump(mode="json") for record in accepted_records]

    # Calculate diversity and bias
    diversity_score = calculate_diversity_score(payloads)
    bias_report = detect_bias(payloads, taxonomy_targets[:len(payloads)])

    if output_dir is not None:
        export_dataset_records(settings, accepted_records, output_dir)
    report_path = settings.repo.reports_dir / "generation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "requested_count": count,
                "accepted_count": len(accepted_records),
                "attempts": attempts,
                "diversity_score": diversity_score,
                "bias_report": bias_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return payloads
