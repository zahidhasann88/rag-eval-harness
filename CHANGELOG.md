# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-17

### Added

- JSONL dataset and corpus loading with line-numbered validation, duplicate id detection,
  unknown-field rejection and warnings for missing ground truth.
- `RAGPipeline` adapter protocol with optional `run_batch`, a `FunctionPipeline` wrapper, and
  loading of custom pipelines from `module:attribute`.
- Built-in `mock` (keyword overlap) and `tfidf` (TF-IDF retrieval with an extractive answer)
  pipelines.
- Retrieval metrics: Precision@K, Recall@K, Hit Rate@K and MRR, using doc-id or text-coverage
  matching with one-to-one credit.
- Answer metrics: exact match, token F1, character-trigram answer similarity, and term-level
  groundedness. Similarity and groundedness are labelled as heuristics.
- `AnswerJudge` extension point for external evaluators, with per-example error isolation.
- Evaluation engine with batching, per-example pipeline error capture, failure attribution to
  retrieval or generation, latency and token summaries, and reproducible run fingerprints.
- Terminal, JSON and self-contained HTML reports.
- `rag-eval` CLI with `run`, `validate` and `compare` commands, plus CI-oriented exit codes
  (`--fail-on-failure`, `--fail-on-regression`, `--strict`).
