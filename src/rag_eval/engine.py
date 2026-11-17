import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from rag_eval._version import __version__
from rag_eval.judges import AnswerJudge
from rag_eval.metrics.answer import evaluate_answer
from rag_eval.metrics.retrieval import evaluate_retrieval
from rag_eval.models import Dataset, EvalConfig, EvalExample, PipelineOutput, SourceInfo
from rag_eval.pipelines.base import BatchRAGPipeline, RAGPipeline
from rag_eval.results import (
    ANSWER_METRICS,
    RETRIEVAL_METRICS,
    AnswerScores,
    EvaluationReport,
    ExampleResult,
    Failure,
    FailureStage,
    LatencySummary,
    MetricSummary,
    RetrievalScores,
    RunMetadata,
    RunSummary,
    TokenSummary,
)

ProgressCallback = Callable[[int], None]


class EmptyDatasetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Outcome:
    output: PipelineOutput | None = None
    latency_ms: float | None = None
    error: str | None = None


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _coerce_output(raw: object) -> PipelineOutput:
    return raw if isinstance(raw, PipelineOutput) else PipelineOutput.model_validate(raw)


class Evaluator:
    """Runs a pipeline over a dataset and scores retrieval and answers independently."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        config: EvalConfig | None = None,
        judges: Sequence[AnswerJudge] = (),
    ) -> None:
        self._pipeline = pipeline
        self._config = config or EvalConfig()
        self._judges = list(judges)

    @property
    def config(self) -> EvalConfig:
        return self._config

    def evaluate(
        self,
        dataset: Dataset,
        *,
        corpus_source: SourceInfo | None = None,
        progress: ProgressCallback | None = None,
    ) -> EvaluationReport:
        examples = dataset.examples
        if not examples:
            raise EmptyDatasetError("dataset contains no examples")

        results: list[ExampleResult] = []
        size = self._config.batch_size
        for start in range(0, len(examples), size):
            batch = examples[start : start + size]
            for example, outcome in zip(batch, self._run_batch(batch), strict=True):
                results.append(self._score(example, outcome))
            if progress is not None:
                progress(len(batch))

        dataset_source = dataset.source or SourceInfo(sha256=_hash_examples(examples))
        return EvaluationReport(
            metadata=RunMetadata(
                fingerprint=self._fingerprint(dataset_source, corpus_source),
                tool_version=__version__,
                created_at=datetime.now(UTC),
                pipeline=self._pipeline.name,
                models=sorted({result.model for result in results if result.model}),
                dataset=dataset_source,
                corpus=corpus_source,
                config=self._config,
                judges=[judge.name for judge in self._judges],
                python_version=platform.python_version(),
            ),
            summary=_summarize(results),
            metrics=_aggregate(results),
            latency=_latency(results),
            tokens=_tokens(results),
            results=results,
        )

    def _run_batch(self, batch: Sequence[EvalExample]) -> list[_Outcome]:
        top_k = self._config.top_k
        if not isinstance(self._pipeline, BatchRAGPipeline):
            return [self._run_single(example.question) for example in batch]
        try:
            raw = list(self._pipeline.run_batch([ex.question for ex in batch], top_k=top_k))
        except Exception as exc:
            return [_Outcome(error=_describe(exc))] * len(batch)
        if len(raw) != len(batch):
            message = f"run_batch returned {len(raw)} outputs for {len(batch)} questions"
            return [_Outcome(error=message)] * len(batch)
        outcomes = []
        for item in raw:
            try:
                output = _coerce_output(item)
            except ValidationError as exc:
                outcomes.append(_Outcome(error=_describe(exc)))
                continue
            outcomes.append(_Outcome(output=output, latency_ms=output.latency_ms))
        return outcomes

    def _run_single(self, question: str) -> _Outcome:
        started = time.perf_counter()
        try:
            output = _coerce_output(self._pipeline.run(question, top_k=self._config.top_k))
        except Exception as exc:
            return _Outcome(error=_describe(exc))
        measured = (time.perf_counter() - started) * 1000
        latency = output.latency_ms if output.latency_ms is not None else measured
        return _Outcome(output=output, latency_ms=latency)

    def _score(self, example: EvalExample, outcome: _Outcome) -> ExampleResult:
        base = ExampleResult(
            id=example.id,
            question=example.question,
            expected_answer=example.expected_answer,
            expected_context=example.expected_context,
            relevant_doc_ids=example.relevant_doc_ids,
        )
        if outcome.output is None:
            error = outcome.error or "pipeline returned no output"
            return base.model_copy(
                update={
                    "error": error,
                    "failure": Failure(stage=FailureStage.PIPELINE_ERROR, reasons=[error]),
                }
            )

        output = outcome.output
        config = self._config
        contexts = output.contexts[: config.top_k]
        retrieval = evaluate_retrieval(
            example, contexts, config.top_k, config.context_match_threshold
        )
        answer = evaluate_answer(example, output.answer, [context.text for context in contexts])
        judge_scores, judge_errors = self._judge(example, output)
        return base.model_copy(
            update={
                "answer": output.answer,
                "contexts": contexts,
                "latency_ms": outcome.latency_ms,
                "token_usage": output.token_usage,
                "model": output.model,
                "retrieval": retrieval,
                "answer_scores": answer,
                "judge_scores": judge_scores,
                "judge_errors": judge_errors,
                "failure": classify_failure(retrieval, answer, config),
            }
        )

    def _judge(
        self, example: EvalExample, output: PipelineOutput
    ) -> tuple[dict[str, float], dict[str, str]]:
        scores: dict[str, float] = {}
        errors: dict[str, str] = {}
        for judge in self._judges:
            try:
                for key, value in judge.score(example, output).items():
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"score {key!r} is not finite")
                    scores[f"{judge.name}.{key}"] = number
            except Exception as exc:
                errors[judge.name] = _describe(exc)
        return scores, errors

    def _fingerprint(self, dataset: SourceInfo, corpus: SourceInfo | None) -> str:
        payload = {
            "dataset": dataset.sha256,
            "corpus": corpus.sha256 if corpus else None,
            "pipeline": self._pipeline.name,
            "config": self._config.model_dump(exclude={"batch_size"}),
            "judges": [judge.name for judge in self._judges],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def classify_failure(
    retrieval: RetrievalScores | None, answer: AnswerScores, config: EvalConfig
) -> Failure | None:
    """Attribute a failure to retrieval when the relevant context was not retrieved; otherwise to
    generation when the answer misses the reference or is poorly grounded."""
    retrieval_reasons = []
    if retrieval is not None and retrieval.recall_at_k < config.min_recall:
        retrieval_reasons.append(
            f"recall@{config.top_k} {retrieval.recall_at_k:.2f} < {config.min_recall:.2f}"
        )
    answer_reasons = []
    if answer.token_f1 is not None and answer.token_f1 < config.min_answer_f1:
        answer_reasons.append(f"token F1 {answer.token_f1:.2f} < {config.min_answer_f1:.2f}")
    if answer.groundedness is not None and answer.groundedness < config.min_groundedness:
        answer_reasons.append(
            f"groundedness {answer.groundedness:.2f} < {config.min_groundedness:.2f}"
        )
    if not retrieval_reasons and not answer_reasons:
        return None
    stage = FailureStage.RETRIEVAL if retrieval_reasons else FailureStage.GENERATION
    return Failure(stage=stage, reasons=retrieval_reasons + answer_reasons)


def _hash_examples(examples: Sequence[EvalExample]) -> str:
    encoded = json.dumps([example.model_dump() for example in examples], sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _aggregate(results: Sequence[ExampleResult]) -> dict[str, MetricSummary]:
    values: dict[str, list[float]] = {name: [] for name in (*RETRIEVAL_METRICS, *ANSWER_METRICS)}
    for result in results:
        if result.retrieval is not None:
            values["precision_at_k"].append(result.retrieval.precision_at_k)
            values["recall_at_k"].append(result.retrieval.recall_at_k)
            values["hit_rate_at_k"].append(result.retrieval.hit_rate_at_k)
            values["mrr"].append(result.retrieval.reciprocal_rank)
        if result.answer_scores is not None:
            for name in ANSWER_METRICS:
                value = getattr(result.answer_scores, name)
                if value is not None:
                    values[name].append(value)
        for name, value in result.judge_scores.items():
            values.setdefault(f"judge.{name}", []).append(value)
    return {
        name: MetricSummary(mean=_mean(items), count=len(items)) for name, items in values.items()
    }


def _percentile(ordered: Sequence[float], percent: float) -> float:
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def _latency(results: Sequence[ExampleResult]) -> LatencySummary:
    ordered = sorted(r.latency_ms for r in results if r.latency_ms is not None)
    if not ordered:
        return LatencySummary()
    return LatencySummary(
        count=len(ordered),
        mean_ms=_mean(ordered),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        max_ms=ordered[-1],
    )


def _tokens(results: Sequence[ExampleResult]) -> TokenSummary:
    usages = [result.token_usage for result in results if result.token_usage is not None]
    return TokenSummary(
        examples_with_usage=len(usages),
        prompt_tokens=sum(usage.prompt_tokens for usage in usages),
        completion_tokens=sum(usage.completion_tokens for usage in usages),
    )


def _summarize(results: Sequence[ExampleResult]) -> RunSummary:
    by_stage = dict.fromkeys(FailureStage, 0)
    for result in results:
        if result.failure is not None:
            by_stage[result.failure.stage] += 1
    failed = sum(by_stage.values())
    return RunSummary(
        num_examples=len(results),
        num_passed=len(results) - failed,
        num_failed=failed,
        failures_by_stage=by_stage,
    )
