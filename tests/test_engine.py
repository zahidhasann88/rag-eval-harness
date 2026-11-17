from collections.abc import Mapping, Sequence

import pytest

from rag_eval import (
    Dataset,
    EmptyDatasetError,
    EvalConfig,
    EvalExample,
    Evaluator,
    PipelineOutput,
    TokenUsage,
)
from rag_eval.results import FailureStage
from tests.conftest import StubPipeline

EXAMPLES = [
    EvalExample(
        id="e1", question="q1", expected_answer="red apple", expected_context=["red apple fact"]
    ),
    EvalExample(
        id="e2", question="q2", expected_answer="blue sky", expected_context=["blue sky fact"]
    ),
    EvalExample(id="e3", question="q3"),
]
OUTPUTS = {
    "q1": PipelineOutput(answer="red apple", contexts=["red apple fact", "other"], model="m1"),
    "q2": PipelineOutput(
        answer="green grass",
        contexts=["noise", "blue sky fact"],
        latency_ms=12.5,
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
    ),
    "q3": PipelineOutput(answer="anything goes"),
}


def _evaluate(
    outputs: Mapping[str, PipelineOutput] = OUTPUTS,
    examples: Sequence[EvalExample] = EXAMPLES,
    **config: int,
) -> tuple[StubPipeline, Evaluator]:
    pipeline = StubPipeline(outputs)
    return pipeline, Evaluator(pipeline, EvalConfig(**{"top_k": 2, **config}))


def test_aggregates_match_hand_computed_values() -> None:
    pipeline, evaluator = _evaluate()

    report = evaluator.evaluate(Dataset(examples=EXAMPLES))
    metrics = {name: (summary.mean, summary.count) for name, summary in report.metrics.items()}

    assert pipeline.calls == [("q1", 2), ("q2", 2), ("q3", 2)]
    assert metrics["precision_at_k"] == pytest.approx((0.5, 2))
    assert metrics["recall_at_k"] == pytest.approx((1.0, 2))
    assert metrics["hit_rate_at_k"] == pytest.approx((1.0, 2))
    assert metrics["mrr"] == pytest.approx((0.75, 2))
    assert metrics["exact_match"] == pytest.approx((0.5, 2))
    assert metrics["token_f1"] == pytest.approx((0.5, 2))
    assert metrics["groundedness"] == pytest.approx((1 / 3, 3))


def test_failures_are_attributed_to_the_right_stage() -> None:
    examples = [*EXAMPLES, EvalExample(id="e4", question="q4", expected_context=["missing"])]
    outputs = {**OUTPUTS, "q4": PipelineOutput(answer="", contexts=[])}
    _, evaluator = _evaluate(outputs, examples)

    report = evaluator.evaluate(Dataset(examples=examples))
    by_id = {result.id: result for result in report.results}

    assert by_id["e1"].passed
    assert by_id["e2"].failure is not None
    assert by_id["e2"].failure.stage is FailureStage.GENERATION
    assert by_id["e3"].failure is not None
    assert by_id["e3"].failure.stage is FailureStage.GENERATION
    assert by_id["e4"].failure is not None
    assert by_id["e4"].failure.stage is FailureStage.RETRIEVAL
    assert report.summary.num_failed == 3
    assert report.summary.failures_by_stage[FailureStage.RETRIEVAL] == 1


def test_missing_ground_truth_is_excluded_from_aggregates() -> None:
    _, evaluator = _evaluate()

    report = evaluator.evaluate(Dataset(examples=EXAMPLES))
    e3 = report.results[2]

    assert e3.retrieval is None
    assert e3.answer_scores is not None
    assert e3.answer_scores.token_f1 is None
    assert report.metrics["recall_at_k"].count == 2


def test_empty_dataset_raises() -> None:
    _, evaluator = _evaluate()

    with pytest.raises(EmptyDatasetError):
        evaluator.evaluate(Dataset(examples=[]))


def test_pipeline_errors_are_recorded_per_example() -> None:
    class Flaky(StubPipeline):
        def run(self, question: str, *, top_k: int) -> PipelineOutput:
            if question == "q2":
                raise RuntimeError("backend down")
            return super().run(question, top_k=top_k)

    report = Evaluator(Flaky(OUTPUTS), EvalConfig(top_k=2)).evaluate(Dataset(examples=EXAMPLES))
    failed = report.results[1]

    assert failed.error == "RuntimeError: backend down"
    assert failed.failure is not None
    assert failed.failure.stage is FailureStage.PIPELINE_ERROR
    assert failed.retrieval is None
    assert report.metrics["recall_at_k"].count == 1


def test_contexts_beyond_top_k_are_ignored() -> None:
    outputs = {"q1": PipelineOutput(answer="red apple", contexts=["a", "b", "red apple fact"])}
    _, evaluator = _evaluate(outputs, EXAMPLES[:1])

    result = evaluator.evaluate(Dataset(examples=EXAMPLES[:1])).results[0]

    assert len(result.contexts) == 2
    assert result.retrieval is not None
    assert result.retrieval.hit_rate_at_k == 0.0


def test_batch_pipelines_are_called_in_batches() -> None:
    class Batched(StubPipeline):
        def __init__(self) -> None:
            super().__init__(OUTPUTS)
            self.batches: list[list[str]] = []

        def run_batch(self, questions: Sequence[str], *, top_k: int) -> list[PipelineOutput]:
            self.batches.append(list(questions))
            return [self.outputs[question] for question in questions]

    pipeline = Batched()
    report = Evaluator(pipeline, EvalConfig(top_k=2, batch_size=2)).evaluate(
        Dataset(examples=EXAMPLES)
    )

    assert pipeline.batches == [["q1", "q2"], ["q3"]]
    assert pipeline.calls == []
    assert report.results[1].latency_ms == 12.5
    assert report.results[0].latency_ms is None


def test_batch_size_mismatch_is_a_pipeline_error() -> None:
    class Broken(StubPipeline):
        def run_batch(self, questions: Sequence[str], *, top_k: int) -> list[PipelineOutput]:
            return []

    report = Evaluator(Broken(OUTPUTS)).evaluate(Dataset(examples=EXAMPLES))

    assert all(r.failure and r.failure.stage is FailureStage.PIPELINE_ERROR for r in report.results)


def test_judges_add_namespaced_metrics_and_isolate_errors() -> None:
    class LengthJudge:
        name = "length"

        def score(self, example: EvalExample, output: PipelineOutput) -> dict[str, float]:
            if example.id == "e3":
                raise ValueError("cannot judge")
            return {"short": float(len(output.answer) < 10)}

    pipeline = StubPipeline(OUTPUTS)
    report = Evaluator(pipeline, EvalConfig(top_k=2), [LengthJudge()]).evaluate(
        Dataset(examples=EXAMPLES)
    )

    assert report.results[0].judge_scores == {"length.short": 1.0}
    assert report.results[2].judge_errors == {"length": "ValueError: cannot judge"}
    assert report.metrics["judge.length.short"].mean == pytest.approx(0.5)
    assert report.metadata.judges == ["length"]


def test_run_metadata_is_reproducible() -> None:
    first = _evaluate()[1].evaluate(Dataset(examples=EXAMPLES))
    second = _evaluate(batch_size=1)[1].evaluate(Dataset(examples=EXAMPLES))
    other_k = _evaluate(top_k=1)[1].evaluate(Dataset(examples=EXAMPLES))

    assert first.metadata.fingerprint == second.metadata.fingerprint
    assert first.metadata.fingerprint != other_k.metadata.fingerprint
    assert first.metrics == second.metrics
    assert first.tokens.total_tokens == 12
    assert first.latency.count == 3
    assert first.metadata.models == ["m1"]
