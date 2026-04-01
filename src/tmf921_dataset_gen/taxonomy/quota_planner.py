from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

import yaml

from ..config import Settings


@dataclass(slots=True)
class TaxonomyTarget:
    taxonomy_category: str
    layer: str
    traffic_profile: str
    scenario_family: str
    domain_context: str
    ordinal: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "taxonomy_category": self.taxonomy_category,
            "layer": self.layer,
            "traffic_profile": self.traffic_profile,
            "scenario_family": self.scenario_family,
            "domain_context": self.domain_context,
            "ordinal": self.ordinal,
        }


class QuotaPlanner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        taxonomy_path = settings.repo.root / "src" / "tmf921_dataset_gen" / "taxonomy" / "taxonomy.yaml"
        self.taxonomy = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))

    def combinations(self) -> list[TaxonomyTarget]:
        combos = [
            TaxonomyTarget(
                taxonomy_category=f"{layer}/{traffic_profile}/{scenario_family}",
                layer=layer,
                traffic_profile=traffic_profile,
                scenario_family=scenario_family,
                domain_context=domain_context,
                ordinal=index,
            )
            for index, (layer, traffic_profile, scenario_family, domain_context) in enumerate(
                product(
                    self.taxonomy["layers"],
                    self.taxonomy["traffic_profiles"],
                    self.taxonomy["scenario_families"],
                    self.taxonomy["domain_contexts"],
                ),
                start=1,
            )
        ]
        return combos

    def plan_targets(self, total_count: int) -> list[dict[str, Any]]:
        combos = self.combinations()
        rng = Random(self.settings.random_seed)
        rng.shuffle(combos)
        targets: list[dict[str, Any]] = []
        for index in range(total_count):
            target = combos[index % len(combos)]
            targets.append({**target.as_dict(), "sample_index": index})
        return targets
