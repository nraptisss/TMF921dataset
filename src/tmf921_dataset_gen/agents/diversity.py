from __future__ import annotations

import random
from typing import Any

from ..config import Settings
from ..ingestion.seed_loader import load_seed_records


SCENARIO_HINTS = {
    "energy": ["minimize", "reduce", "optimize"],
    "slicing": ["create", "deploy", "instantiate"],
    "predictive_assurance": ["ensure", "predict", "proactively maintain"],
    "multi_domain": ["guarantee", "coordinate", "assure"],
    "conflict_negotiation": ["negotiate", "resolve", "balance"],
    "resilience": ["preserve", "maintain", "protect"],
    "reporting": ["report", "notify", "monitor"],
    "closed_loop_autonomy": ["automate", "heal", "reconfigure"],
}


class DiversityAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.seeds = load_seed_records(settings)

    def _matching_seed(self, scenario_family: str) -> dict[str, Any]:
        keywords = scenario_family.replace("_", " ")
        for seed in self.seeds:
            haystack = f"{seed['title']} {seed['metadata'].get('notes', '')}".lower()
            if any(token in haystack for token in keywords.split()):
                return seed
        return self.seeds[0]

    def _sample_kpis(self, taxonomy_target: dict[str, Any], rng: random.Random) -> dict[str, Any]:
        traffic = taxonomy_target["traffic_profile"]
        scenario = taxonomy_target["scenario_family"]
        if traffic == "urllc":
            base = {"latency_ms": round(rng.uniform(0.5, 5.0), 2), "reliability_percent": round(rng.uniform(99.99, 99.9999), 4)}
        elif traffic == "embb":
            base = {"throughput_mbps": rng.randint(200, 1500), "reliability_percent": round(rng.uniform(99.9, 99.999), 3)}
        else:
            base = {"device_count": rng.randint(5000, 50000), "delivery_ratio_percent": round(rng.uniform(99.0, 99.99), 3)}
        if scenario == "energy":
            base["energy_kwh"] = rng.randint(200, 1200)
        if scenario == "reporting":
            base["reporting_interval_seconds"] = rng.choice([60, 120, 300])
        if scenario == "predictive_assurance":
            base["reaction_time_ms"] = rng.randint(50, 300)
        return base

    def _render_nl(self, taxonomy_target: dict[str, Any], kpis: dict[str, Any], rng: random.Random) -> str:
        layer = taxonomy_target["layer"]
        traffic = taxonomy_target["traffic_profile"].upper()
        scenario = taxonomy_target["scenario_family"].replace("_", " ")
        context = taxonomy_target["domain_context"]
        lead = rng.choice(SCENARIO_HINTS[taxonomy_target["scenario_family"]])
        fragments = [f"For the {context} 6G {layer} layer, {lead}"]
        if "latency_ms" in kpis:
            fragments.append(f"{traffic} performance with latency below {kpis['latency_ms']} ms")
        if "throughput_mbps" in kpis:
            fragments.append(f"per-user throughput of at least {kpis['throughput_mbps']} Mbps")
        if "device_count" in kpis:
            fragments.append(f"support for {kpis['device_count']} devices")
        if "reliability_percent" in kpis:
            fragments.append(f"reliability of {kpis['reliability_percent']}%")
        if "delivery_ratio_percent" in kpis:
            fragments.append(f"packet delivery ratio of {kpis['delivery_ratio_percent']}%")
        if "energy_kwh" in kpis:
            fragments.append(f"while keeping energy consumption under {kpis['energy_kwh']} kWh per day")
        if "reaction_time_ms" in kpis:
            fragments.append(f"and trigger corrective action within {kpis['reaction_time_ms']} ms")
        if "reporting_interval_seconds" in kpis:
            fragments.append(f"with fulfillment reports every {kpis['reporting_interval_seconds']} seconds")
        fragments.append(f"for the {scenario} scenario.")
        return ", ".join(fragments[:-1]) + " " + fragments[-1]

    def generate(self, taxonomy_target: dict[str, Any], sample_index: int) -> dict[str, Any]:
        rng = random.Random(self.settings.random_seed + sample_index)
        seed = self._matching_seed(taxonomy_target["scenario_family"])
        kpis = self._sample_kpis(taxonomy_target, rng)
        nl_intent = self._render_nl(taxonomy_target, kpis, rng)
        return {
            "nl_intent": nl_intent,
            "seed_ids": [seed["id"]],
            "metadata": {
                "taxonomy_category": taxonomy_target["taxonomy_category"],
                "kpis": kpis,
                "seed_id": seed["id"],
            },
        }
