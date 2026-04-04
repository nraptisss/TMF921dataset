# Research-Grade Upgrade Plan for TMF921 NL Intent <-> TMF921 Intent Dataset Generation

## Summary

Upgrade the pipeline from a schema-valid synthetic generator into a research-grade dataset production system with stronger semantic guarantees, evidence-grounding, and release-quality evaluation. The main changes are: tighten how payloads are generated, replace weak heuristic acceptance with layered symbolic + model-based verification, add dataset audit gates, and align the methodology/docs with what is actually measured.

Success criteria:
- Accepted records must preserve all required constraints from the NL intent in the formal payload.
- Acceptance must fail on omitted constraints, contradictory names/context, wrong expectation type, and operator mismatches.
- Dataset release must be blocked unless audit thresholds pass on automatic checks plus manual review.
- Methodology claims must be limited to validated behavior.

## Key Changes

### 1. Generation architecture
- Keep the current deterministic TMF921 payload builder as the structural backbone, but treat it as one candidate generator, not the final truth.
- Split generation into explicit intermediate representations:
  1. `intent_frame`: normalized semantic frame from NL intent or sampled target
  2. `constraint_set`: typed KPI and operator structure
  3. `tmf921_payload`: formal payload rendered from the frame
- Require every field in payload `name`, `context`, expectation type, and metric params to be derived from the intermediate frame, not directly from free-form LLM hints.
- Restrict LLM use to bounded roles:
  - NL rewriting
  - candidate naming/context suggestions
  - semantic judging only as a secondary signal
- Add a canonical operator model for constraints:
  - `at_most`, `at_least`, `exactly`, `within`, `between`, `periodic_every`, `trigger_within`
- Map scenario families explicitly to allowed intent semantics.
  - `reporting` must require reporting semantics.
  - `predictive_assurance` and `closed_loop_autonomy` must support event/trigger/remediation semantics instead of collapsing to plain throughput/reliability templates.
- Remove acceptance of payload fields whose content is semantically unrelated to the final NL intent, even if schema-valid.

### 2. Verifier stack
- Replace the current heuristic acceptance rule with layered validation:
  1. Schema validation against TMF921 OAS and Pydantic model
  2. TIO structural validation
  3. Symbolic semantic equivalence checks
  4. Evidence-grounding checks
  5. Optional LLM judge as tie-breaker or secondary reviewer only
- Add a symbolic semantic verifier that compares:
  - extracted constraints from `nl_intent`
  - normalized constraint frame used to build payload
  - constraints re-parsed from rendered payload
- The symbolic verifier must reject:
  - missing numeric constraints
  - changed operators such as `strict/exactly` becoming `atLeast`
  - omitted event timing constraints like failover within `50 ms`
  - mismatched scenario semantics
  - contradictory payload `name` or `context`
- Replace the current realism score with evidence attribution:
  - for each accepted record, store which retrieved chunks support which constraint or context claim
  - reject records with unsupported domain/context claims if grounding is required for that mode
- Downgrade LLM semantic review from score-maximizing behavior to advisory behavior.
  - It may add notes or confirm borderline cases.
  - It must not rescue a record that fails symbolic checks.

### 3. Retrieval and grounding
- Make RAG contribute to generation in a traceable way:
  - retrieve candidate evidence before generation
  - attach evidence IDs to each semantic frame field
  - require `name` and `context` suggestions to cite top supporting chunks
- Add retrieval quality checks:
  - source diversity
  - minimum similarity threshold
  - source whitelist for factual grounding claims
- Separate two operating modes:
  - `synthetic_semantic`: internally coherent generation, weaker external grounding claims
  - `grounded_corpus`: accepted only if important claims are attributable to retrieved evidence
- Prevent noisy sources like irrelevant Postman operations from dominating accepted-context evidence for semantic claims.

### 4. Evaluation and release criteria
- Create an evaluation set of hand-audited examples with:
  - gold NL intents
  - gold normalized constraint frames
  - expected TMF921 payload properties
  - hard negative cases
- Add benchmark suites for:
  - operator preservation
  - missing-constraint detection
  - wrong-expectation detection
  - context/name contradiction detection
  - retrieval support detection
- Add dataset-level release gates:
  - semantic preservation rate threshold
  - unsupported-claim rate ceiling
  - duplicate/near-duplicate ceiling
  - manual audit pass rate threshold
- Require manual review on a statistically meaningful sample per release with a fixed rubric:
  - semantic faithfulness
  - TMF/TIO correctness
  - grounding support
  - linguistic naturalness
- Record inter-rater agreement if multiple reviewers are used.
- Reframe manifest metrics:
  - keep heuristic scores if useful internally
  - clearly label them as diagnostics, not proof of correctness
- Update docs so claims such as “strict adherence,” “ensures grounding,” or “largest public dataset” are only made if backed by explicit evidence.

## Public Interfaces and Data Changes

- Add normalized intermediate artifact per record, for example:
  - `metadata.intent_frame`
  - `metadata.constraint_set`
  - `metadata.constraint_alignment`
  - `metadata.evidence_map`
- Extend validation metadata with explicit booleans and counts:
  - `semantic_pass`
  - `operator_pass`
  - `constraint_coverage`
  - `unsupported_claim_count`
  - `contradiction_count`
- Replace or supplement scalar `realism_score` with:
  - `grounding_mode`
  - `grounding_pass`
  - `supported_claim_ratio`
- Keep existing `dataset.jsonl` stable where possible, but add new metadata keys rather than overloading old ones.
- Add a release audit artifact, such as `artifacts/reports/release_audit.json`, summarizing pass/fail gates and sampled review outcomes.

## Test Plan

- Unit tests:
  - symbolic extraction of constraints from NL
  - round-trip extraction from payload
  - operator-preservation logic
  - scenario-family to expectation mapping
  - contradiction detection for `name` and `context`
- Integration tests:
  - records with omitted timing constraints must be rejected
  - records with mismatched payload names must be rejected
  - reporting intents missing reporting semantics must be rejected
  - grounded mode must reject unsupported contextual claims
- Regression tests:
  - include examples similar to the current accepted failures in `output/1k_qwen_gpu_fullpower/dataset.jsonl`
  - include adversarial phrasing such as `strictly`, `exactly`, `at least`, `under`, `within`, `unless`, `if degradation occurs`
- Evaluation tests:
  - run release benchmark suite on every candidate pipeline revision
  - compare against prior baseline metrics and human audit sample results
- Acceptance criteria for implementation completion:
  - previously identified false positives are rejected
  - benchmark suite passes
  - docs and manifests reflect the new measurement model

## Assumptions and Defaults

- Target mode is `research-grade` and scope is `full upgrade`.
- Backward compatibility for old output files is desirable but secondary to correctness.
- The current deterministic TMF921 builder remains useful and should be preserved as a controlled renderer.
- LLMs are treated as helpful but untrusted components.
- “SOTA-like” here means stronger semantic guarantees, stronger evaluation, and more defensible dataset release practice, not merely using a larger model.
