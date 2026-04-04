from __future__ import annotations

import json
import logging
import random
from typing import Any

from ..config import Settings
from ..ingestion.seed_loader import load_seed_records
from ..llm import LLMRouter

LOGGER = logging.getLogger(__name__)


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

# Natural language template variations for more human-like expressions
NL_TEMPLATES = {
    "conversational": [
        "Hey, can we {action} a {traffic} slice for the {context}? I need {kpis} and make sure it's {reliability_desc}.",
        "Could you please {action} this {layer} slice? We're looking for {kpis} with {energy_constraint}.",
        "I'd like to request a {action} {layer} connection for {context}. The key requirements are: {kpis}.",
        "Is it possible to {action} the {traffic} service in {context}? We need {kpis}.",
    ],
    "administrative": [
        "{action} {traffic} slice. {kpis}. Target: {context}.",
        "Deploy: {layer} slice. Requirements: {kpis}.",
        "Action required: {action} {layer} for {context}. Specs: {kpis}.",
        "Implement {action} {traffic} functionality. Constraints: {kpis}.",
    ],
    "business_focused": [
        "We need to {action} the {layer} for {business_goal}. Keep {energy_constraint} but don't let {reliability_constraint} drop below {threshold}.",
        "The business requires {action} of {layer} services. Critical parameters: {kpis}.",
        "To support {business_goal}, please {action} {traffic} with {kpis}.",
        "Optimize for {business_goal}: {action} {layer} must maintain {kpis}.",
    ],
    "technical_precise": [
        "{action} {layer} interface: {kpis}. Context: {context}.",
        "Technical specification: {traffic} {action} in {domain} with {kpis}.",
        "Configure {layer} {action}. Parameters: {kpis}. Environment: {context}.",
        "Engineering requirement: {traffic} slice must achieve {kpis} in {setting}.",
    ]
}


class DiversityAgent:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.seeds = load_seed_records(settings)
        self.router = LLMRouter(settings)

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
        if scenario == "closed_loop_autonomy":
            base["reaction_time_ms"] = rng.randint(30, 120)

        return base

    def _render_nl(self, taxonomy_target: dict[str, Any], kpis: dict[str, Any], rng: random.Random) -> str:
        layer = taxonomy_target["layer"]
        traffic = taxonomy_target["traffic_profile"]
        scenario = taxonomy_target["scenario_family"]
        context = taxonomy_target["domain_context"]
        
        # Select template style randomly
        template_style = rng.choice(list(NL_TEMPLATES.keys()))
        template = rng.choice(NL_TEMPLATES[template_style])
        
        # Prepare template variables
        action_map = {
            "minimize": "minimize",
            "reduce": "reduce", 
            "optimize": "optimize",
            "create": "create",
            "deploy": "deploy",
            "instantiate": "instantiate",
            "ensure": "ensure",
            "predict": "predict",
            "proactively maintain": "proactively maintain",
            "guarantee": "guarantee",
            "coordinate": "coordinate",
            "assure": "assure",
            "negotiate": "negotiate",
            "resolve": "resolve",
            "balance": "balance",
            "preserve": "preserve",
            "maintain": "maintain",
            "protect": "protect",
            "report": "report",
            "notify": "notify",
            "monitor": "monitor",
            "automate": "automate",
            "heal": "heal",
            "reconfigure": "reconfigure"
        }
        
        action = action_map.get(rng.choice(SCENARIO_HINTS[scenario]), "configure")
        
        # Format KPIs for natural language
        kpi_parts = []
        if "latency_ms" in kpis:
            kpi_parts.append(f"latency below {kpis['latency_ms']} ms")
        if "throughput_mbps" in kpis:
            kpi_parts.append(f"throughput of at least {kpis['throughput_mbps']} Mbps")
        if "device_count" in kpis:
            kpi_parts.append(f"support for {kpis['device_count']} devices")
        if "reliability_percent" in kpis:
            kpi_parts.append(f"reliability of {kpis['reliability_percent']}%")
        if "delivery_ratio_percent" in kpis:
            kpi_parts.append(f"packet delivery ratio of {kpis['delivery_ratio_percent']}%")
        if "energy_kwh" in kpis:
            kpi_parts.append(f"energy consumption under {kpis['energy_kwh']} kWh per day")
        if "reaction_time_ms" in kpis:
            kpi_parts.append(f"corrective action within {kpis['reaction_time_ms']} ms")
        if "reporting_interval_seconds" in kpis:
            kpi_parts.append(f"fulfillment reports every {kpis['reporting_interval_seconds']} seconds")
        if scenario == "closed_loop_autonomy" and "reaction_time_ms" in kpis:
            kpi_parts.append(f"if degradation is detected, trigger failover within {kpis['reaction_time_ms']} ms")
        
        kpis_str = ", ".join(kpi_parts) if kpi_parts else "standard performance"
        
        # Format additional context variables
        reliability_desc = "highly reliable" if "reliability_percent" in kpis and kpis["reliability_percent"] >= 99.9 else "reliable"
        energy_constraint = f"energy under {kpis.get('energy_kwh', 'N/A')} kWh" if "energy_kwh" in kpis else "standard energy usage"
        business_goal = {
            "energy": "energy efficiency",
            "reporting": "better monitoring",
            "predictive_assurance": "proactive maintenance",
            "multi_domain": "seamless coordination"
        }.get(scenario, "optimal performance")
        threshold = f"{kpis.get('reliability_percent', '99.9')}%" if "reliability_percent" in kpis else "99.9%"
        
        # Format the template
        try:
            nl_intent = template.format(
                action=action,
                traffic=traffic.upper(),
                layer=layer,
                context=context,
                kpis=kpis_str,
                reliability_desc=reliability_desc,
                energy_constraint=energy_constraint,
                business_goal=business_goal,
                threshold=threshold
            )
        except KeyError:
            # Fallback to original method if template formatting fails
            lead = rng.choice(SCENARIO_HINTS[scenario])
            fragments = [f"For the {context} 6G {layer} layer, {lead}"]
            if "latency_ms" in kpis:
                fragments.append(f"{traffic.upper()} performance with latency below {kpis['latency_ms']} ms")
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
            fragments.append(f"for the {scenario.replace('_', ' ')} scenario.")
            nl_intent = ", ".join(fragments[:-1]) + " " + fragments[-1]

        if scenario == "closed_loop_autonomy" and "reaction_time_ms" in kpis and "failover" not in nl_intent.lower():
            nl_intent = f"{nl_intent.rstrip('.')} If degradation occurs, trigger failover within {kpis['reaction_time_ms']} ms."

        return nl_intent

    def _llm_rewrite(self, baseline_intent: str, taxonomy_target: dict[str, Any], seed: dict[str, Any], kpis: dict[str, Any]) -> str:
        if not self.settings.enable_llm_rewrite or not self.router.supports_generation():
            return baseline_intent
        prompt = f"""
You are generating a realistic 6G telecom natural-language intent for synthetic data.
Preserve every numeric KPI, unit, and hard constraint exactly.
Rewrite the baseline request into a more natural production-style operator request.
Return JSON only with this shape:
{{"nl_intent": "..."}}

Taxonomy target:
{json.dumps(taxonomy_target, indent=2)}

Seed example:
{seed['title']}

Seed notes:
{seed['metadata'].get('notes')}

KPI payload:
{json.dumps(kpis, indent=2)}

Baseline intent:
{baseline_intent}
""".strip()
        try:
            response = self.router.generate_json(
                prompt,
                model=self.settings.reasoning_model,
                temperature=0.4,
                max_new_tokens=self.settings.local_planning_max_new_tokens,
            )
            candidate = str(response.get("nl_intent", "")).strip()
            if candidate:
                return candidate
        except Exception as exc:
            LOGGER.debug("LLM rewrite failed, using baseline: %s", exc)
        return baseline_intent

    def generate(self, taxonomy_target: dict[str, Any], sample_index: int) -> dict[str, Any]:
        rng = random.Random(self.settings.random_seed + sample_index)
        seed = self._matching_seed(taxonomy_target["scenario_family"])
        kpis = self._sample_kpis(taxonomy_target, rng)
        baseline_intent = self._render_nl(taxonomy_target, kpis, rng)
        nl_intent = self._llm_rewrite(baseline_intent, taxonomy_target, seed, kpis)
        return {
            "nl_intent": nl_intent,
            "seed_ids": [seed["id"]],
            "metadata": {
                "taxonomy_category": taxonomy_target["taxonomy_category"],
                "taxonomy_target": dict(taxonomy_target),
                "kpis": kpis,
                "seed_id": seed["id"],
                "generation_backend": self.settings.inference_backend,
            },
        }
