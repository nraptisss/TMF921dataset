# plan.md — TMF921 v5.0 Synthetic NL Intent Dataset Generator

## Summary

Build a Python 3.11, LangGraph-based pipeline that generates `Intent_FVO` payloads for TMF921 `/intent` POST requests, grounded by TMF schemas, TR290 semantics, official TMF examples, seeds, and optional IDAN reference assets.

```text
[Postman collection]   [TMF921 OAS]   [TR290 docs]   [IDAN refs]   [Seeds]
         |                    |              |             |           |
         +-------> Source Normalizer / Example Extractor / Doc Parser--+
                                      |
                         [Canonical schema registry + corpora]
                                      |
                           [Embeddings + Chroma index]
                                      |
                              [LangGraph orchestration]
                                      |
      [DiversityAgent] -> [Retriever] -> [TranslatorAgent] -> [Critic/Refiner]
             ^                                                       |
             +---------------- quota / retry / accept ----------------+
                                      |
                         [HF Dataset export + manifests + dashboard]
                                      |
                          ./output/test_dataset/ and final corpus
```

## Implementation Order

1. Bootstrap project skeleton, config, CLI, logging, pytest setup, and preflight checks.
2. Extract canonical schemas and examples from the TMF921 OpenAPI and Postman collection.
3. Normalize TR290, seeds, and optional IDAN material into a shared corpus.
4. Build embedding and retrieval layers.
5. Implement taxonomy-driven diversity, translation, and critic/refinement agents.
6. Orchestrate them with LangGraph and export the accepted dataset.
7. Add docs, Docker support, dashboard, tests, and a guarded sample generation entrypoint.

