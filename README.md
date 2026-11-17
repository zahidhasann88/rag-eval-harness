# RAG Evaluation Harness

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight framework for measuring Retrieval-Augmented Generation (RAG) pipelines. It scores **retrieval** and **answer generation** separately, so when a question fails you can tell which stage caused it.

- Runs offline. Every built-in metric is deterministic and local, with no API keys or model downloads.
- Works with any stack. You write a small adapter, and the harness doesn't require a particular vector database or LLM framework.
- Gives reproducible results. Each run records dataset and corpus hashes, its configuration, and a fingerprint.
- Produces terminal, JSON and self-contained HTML reports, plus baseline vs candidate comparison for CI.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Dataset format](#dataset-format)
- [Connecting your pipeline](#connecting-your-pipeline)
- [Metrics](#metrics)
- [Failure attribution](#failure-attribution)
- [CLI reference](#cli-reference)
- [Example results](#example-results)
- [Limitations](#limitations)

## Installation

Requires Python 3.11 or newer.

```bash
pip install ragsonar
```

This installs the `rag-eval` command and the `rag_eval` Python package. To work from source:

```bash
git clone https://github.com/zahidhasann88/rag-eval-harness.git
cd rag-eval-harness
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

```bash
# Check a dataset for schema errors and missing ground truth
rag-eval validate --dataset examples/dataset.jsonl

# Evaluate the built-in demo pipelines
rag-eval run --dataset examples/dataset.jsonl --pipeline mock --top-k 5
rag-eval run --dataset examples/dataset.jsonl --pipeline tfidf --report report.html

# Save JSON reports and compare two runs
rag-eval run -d examples/dataset.jsonl -c examples/corpus.jsonl -p mock  -r baseline.json
rag-eval run -d examples/dataset.jsonl -c examples/corpus.jsonl -p tfidf -r candidate.json
rag-eval compare --baseline baseline.json --candidate candidate.json
```

The built-in pipelines index the corpus you pass with `--corpus`. If you leave `--corpus` out, they index the `expected_context` passages pooled from the dataset. That works for a quick demo, but the corpus then contains no distractors, so retrieval scores come out optimistic.

## Architecture

```text
             dataset.jsonl            corpus.jsonl (optional)
                  │                          │
                  ▼                          ▼
          ┌──────────────┐          ┌──────────────────┐
          │   dataset    │          │    pipelines     │  mock · tfidf · module:attr
          │ load/validate│          │  RAGPipeline     │
          └──────┬───────┘          └────────┬─────────┘
                 │  EvalExample              │  PipelineOutput
                 └─────────────┬─────────────┘  (answer, contexts, latency, tokens)
                               ▼
                       ┌───────────────┐        ┌──────────────────┐
                       │    engine     │───────▶│  judges (opt.)   │  AnswerJudge
                       │   Evaluator   │        └──────────────────┘
                       └───┬───────┬───┘
            top-k contexts │       │ answer + contexts
                           ▼       ▼
              ┌──────────────┐  ┌──────────────┐
              │  retrieval   │  │    answer    │
              │   metrics    │  │   metrics    │
              └──────┬───────┘  └──────┬───────┘
                     └────────┬────────┘
                              ▼
                     EvaluationReport ──▶ terminal · JSON · HTML ──▶ compare
```

| Module | Responsibility |
|---|---|
| `rag_eval.dataset` | Parse and validate JSONL datasets and corpora, with line-numbered errors |
| `rag_eval.pipelines` | The `RAGPipeline` contract, the mock and TF-IDF pipelines, and plugin loading |
| `rag_eval.metrics.retrieval` | Precision@K, Recall@K, Hit Rate@K, reciprocal rank, context matching |
| `rag_eval.metrics.answer` | Exact match, token F1, answer similarity, groundedness |
| `rag_eval.engine` | Batched execution, per-example scoring, failure attribution, aggregation |
| `rag_eval.judges` | The `AnswerJudge` extension point for external evaluators such as an LLM judge |
| `rag_eval.reporting` | Terminal, JSON and HTML reporters |
| `rag_eval.compare` | Metric deltas and per-question status changes between two reports |

Each module depends only on the shared Pydantic models in `models.py` and `results.py`, so you can use the metrics, loaders and reporters on their own.

## Dataset format

A dataset is a JSONL file with one example per line. Blank lines are skipped and a UTF-8 BOM is accepted.

```json
{"id": "idempotency-key", "question": "What is an idempotency key?", "expected_answer": "A unique identifier used to prevent duplicate operations.", "expected_context": ["An idempotency key is a unique identifier used to prevent duplicate operations."]}
{"question": "What does a dead letter queue do?", "relevant_doc_ids": ["dead-letter-queue"]}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | Must not be blank |
| `id` | string | no | Defaults to `line-<n>` and must be unique |
| `expected_answer` | string | no | Reference answer for exact match, token F1 and similarity |
| `expected_context` | string or list | no | Passages that should be retrieved, matched by text |
| `relevant_doc_ids` | string or list | no | Ids of relevant documents. When present, these take precedence over `expected_context` |
| `metadata` | object | no | Free-form and not scored |

Unknown fields are rejected, so a typo such as `expected_contexts` shows up as an error instead of silently disabling a metric. Missing ground truth is allowed. The affected metrics are skipped for that example and left out of the averages, and `validate` warns about it.

A corpus uses the same JSONL layout, with records of the form `{"id": "...", "text": "...", "metadata": {...}}`.

## Connecting your pipeline

A pipeline is any object that has a `name` and a `run(question, *, top_k)` method returning a `PipelineOutput`:

```python
# my_app/eval_adapter.py
from rag_eval import PipelineOutput, RetrievedContext, TokenUsage

from my_app import llm, vector_store


class MyRagPipeline:
    name = "my-rag-v2"

    def run(self, question: str, *, top_k: int) -> PipelineOutput:
        hits = vector_store.search(question, k=top_k)
        completion = llm.answer(question, [hit.text for hit in hits])
        return PipelineOutput(
            answer=completion.text,
            contexts=[RetrievedContext(text=h.text, doc_id=h.id, score=h.score) for h in hits],
            token_usage=TokenUsage(
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            ),
            model=completion.model,
        )
```

```bash
rag-eval run --dataset evals/support.jsonl --pipeline my_app.eval_adapter:MyRagPipeline --report report.html
```

- `--pipeline` accepts `module:attribute`, where the attribute can be a class, an instance or a zero-argument factory. The current directory is on the import path.
- `contexts` can also be a plain list of strings.
- If you don't set `latency_ms`, the harness measures wall-clock time around `run`.
- For batched inference, add `run_batch(questions, *, top_k) -> list[PipelineOutput]`. The engine then calls it with `--batch-size` questions at a time.
- A plain function works too: `FunctionPipeline(lambda q, k: {"answer": ..., "contexts": [...]})`.

You can also run evaluations from Python:

```python
from rag_eval import EvalConfig, Evaluator, load_dataset
from rag_eval.reporting import write_report

report = Evaluator(MyRagPipeline(), EvalConfig(top_k=5)).evaluate(
    load_dataset("evals/support.jsonl")
)
write_report(report, "report.html")
print(report.metrics["recall_at_k"].mean)
```

### External judges

The default evaluators never call a model. To add one, such as an LLM-as-judge, implement `AnswerJudge`:

```python
class FaithfulnessJudge:
    name = "faithfulness"

    def score(self, example, output) -> dict[str, float]:
        verdict = my_llm_client.grade(
            question=example.question,
            answer=output.answer,
            contexts=[c.text for c in output.contexts],
        )
        return {"score": verdict.score}
```

```bash
rag-eval run -d evals/support.jsonl -p my_app.eval_adapter:MyRagPipeline -j my_app.judges:FaithfulnessJudge
```

Judge scores are stored as `<judge>.<score>`, averaged as `judge.faithfulness.score`, and shown in every report. If a judge raises an error, it's recorded for that example and the run continues.

## Metrics

Each example's retrieved contexts are truncated to the first *K*. Every metric is computed per example and then averaged over the examples that have the required ground truth. The report shows that count as `N`.

### Retrieval

Let `rel_i ∈ {0, 1}` mark whether the context at rank *i* is relevant, and let *G* be the set of ground-truth items for the example.

| Metric | Formula |
|---|---|
| Precision@K | `Σ_{i≤K} rel_i / K` |
| Recall@K | `Σ_{i≤K} rel_i / |G|` |
| Hit Rate@K | `1 if Σ_{i≤K} rel_i > 0 else 0` |
| MRR@K | mean of `1 / rank of first relevant context within K`, or 0 when no context in the top K is relevant |

The relevance judgement works like this:

- **Doc ids**: when an example has `relevant_doc_ids`, a context is relevant if its `doc_id` is in that set.
- **Text**: otherwise, a context matches an expected passage if, after normalization, it contains the passage, or if it covers at least `--context-match-threshold` (default `0.75`) of the passage's content terms. Content terms are lowercase, with stopwords removed and plurals folded.
- **One-to-one credit**: each ground-truth item is credited at most once, in rank order, so duplicate chunks can't inflate precision.
- **Fewer than K contexts**: Precision@K still divides by *K*. A pipeline that returns fewer than *K* contexts is scored as if the missing ranks were irrelevant.

### Answer

Answers are normalized before comparison: lowercased, ASCII punctuation and the articles *a/an/the* removed, whitespace collapsed (the SQuAD convention).

| Metric | Definition | Requires |
|---|---|---|
| Exact Match | `1 if norm(answer) == norm(reference) else 0` | `expected_answer` |
| Token F1 | `2PR / (P + R)` over the overlap of normalized token multisets, where `P = overlap / |answer|` and `R = overlap / |reference|` | `expected_answer` |
| Answer Similarity *(heuristic)* | Cosine similarity of character-trigram count vectors of the normalized strings | `expected_answer` |
| Groundedness *(heuristic)* | `|terms(answer) ∩ terms(contexts)| / |terms(answer)|` over distinct content terms; undefined for an answer with no content terms | retrieved contexts |

Exact match and token F1 are 1 when both strings are empty and 0 when only one is. Groundedness is 0 when no contexts were retrieved. The report also lists the answer terms that weren't found in any retrieved context.

## Failure attribution

Each example passes, or fails at one stage:

| Stage | Condition |
|---|---|
| `retrieval` | Recall@K < `--min-recall` (default 0.5). The generator never received the evidence |
| `generation` | Retrieval passed or had no ground truth, but token F1 < `--min-f1` (default 0.5) or groundedness < `--min-groundedness` (default 0.5) |
| `pipeline_error` | The pipeline raised an exception or returned invalid output |

Every listed reason is kept, so a retrieval failure that also produced a poor answer shows both.

## CLI reference

```text
rag-eval validate --dataset PATH [--corpus PATH] [--strict]
rag-eval run      --dataset PATH [--pipeline mock|tfidf|module:attr] [--corpus PATH]
                  [--top-k 5] [--report out.json] [--report out.html] [--limit N]
                  [--batch-size 16] [--judge module:attr] [--context-match-threshold 0.75]
                  [--min-recall 0.5] [--min-f1 0.5] [--min-groundedness 0.5]
                  [--show-failures 10] [--fail-on-failure]
rag-eval compare  --baseline PATH --candidate PATH [--tolerance 0.0] [--output diff.json]
                  [--fail-on-regression]
```

| Exit code | Meaning |
|---|---|
| `0` | Success |
| `1` | A check failed: invalid dataset, `--strict` warnings, `--fail-on-failure`, or `--fail-on-regression` |
| `2` | Usage or input error, such as an unreadable or empty dataset or an unknown pipeline |

`--report` can be repeated, and the file extension (`.json` or `.html`) picks the format. The HTML report is a single file with inline CSS and no scripts or external assets, so you can attach it to a CI run or open it offline. `compare` warns when the two reports used different datasets, corpora, `top_k` or match thresholds. In CI, `--fail-on-regression --tolerance 0.01` blocks merges that lower any metric by more than 0.01.

## Example results

Output from `tfidf` on the bundled 17-question dataset and 20-document corpus (`-c examples/corpus.jsonl`). The corpus includes distractor passages:

```text
| Metric                        |  Mean |  N |
|-------------------------------+-------+----|
| Precision@5                   | 0.176 | 17 |
| Recall@5                      | 0.882 | 17 |
| Hit Rate@5                    | 0.882 | 17 |
| MRR@5                         | 0.799 | 17 |
| Exact Match                   | 0.000 | 17 |
| Token F1                      | 0.390 | 17 |
| Answer Similarity (heuristic) | 0.472 | 17 |
| Groundedness (heuristic)      | 1.000 | 16 |
Passed 6 / 17   Failed 11 (retrieval 2, generation 9)
```

How to read these numbers:

- **Precision@5 is capped at 0.2.** Each question has exactly one relevant passage, so even a perfect retriever can't score higher.
- **The two retrieval failures are vocabulary mismatches.** `retry-storm` asks about "retried requests piling up", but the relevant passage says "jitter" and "synchronized retry storms". `instant-rollback` shares no content terms with the corpus, so nothing is retrieved, and groundedness is skipped for its empty answer (N = 16).
- **Most generation failures come from wording.** The extractive answer is a single corpus sentence, which often differs from the wording of the reference answer.
- **Groundedness is 1.0 by construction.** An extractive answer is copied from the retrieved text.

Comparing `mock` (baseline) with `tfidf` (candidate):

```text
| Metric            | Baseline | Candidate |  Delta | Status    |
|-------------------+----------+-----------+--------+-----------|
| MRR@5             |    0.853 |     0.799 | -0.054 | regressed |
| Token F1          |    0.366 |     0.390 | +0.024 | improved  |
| Answer Similarity |    0.513 |     0.472 | -0.041 | regressed |
| Pass Rate         |    0.176 |     0.353 | +0.176 | improved  |
```

TF-IDF ranks some relevant passages slightly lower than keyword overlap does, but answering with one sentence instead of the whole passage fixes five questions and breaks two. The per-question table in `compare` lists them by id.

## Limitations

- **Groundedness is a lexical approximation, not fact-checking.** It checks whether the answer's words appear in the retrieved text:
  - A paraphrased but faithful answer can score low.
  - An answer that recombines context words into a false claim ("the queue *never* stores failed messages") can score high.
  - It ignores negation, numbers in context, word order and entailment.

  Use it to spot answers that bring in terms not found in the retrieved text, and use an `AnswerJudge` backed by an NLI model or an LLM when you need semantic faithfulness.
- **Answer similarity measures surface overlap.** Character trigrams reward shared spelling, not shared meaning. Synonyms score low, and near-identical sentences with opposite meanings score high.
- **Text matching for retrieval depends on the threshold.** When your chunks differ a lot from the `expected_context` passages, prefer `relevant_doc_ids`, or tune `--context-match-threshold` and keep it fixed across compared runs.
- **Text processing is English-centric.** Stopwords, plural folding and article removal assume English. Other languages still tokenize on Unicode word boundaries, but they get no stopword removal or plural folding.
- **The built-in pipelines are baselines, not models.** `mock` and `tfidf` are extractive and involve no language model. Their scores describe those baselines only.

## Development

```bash
pip install -e ".[dev]"
ruff format --check . && ruff check .
mypy
pytest --cov
```

## License

MIT. See [LICENSE](https://github.com/zahidhasann88/rag-eval-harness/blob/main/LICENSE).
