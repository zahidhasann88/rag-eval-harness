import pytest

from rag_eval import Dataset, EvalConfig, EvalExample, EvaluationReport, Evaluator, PipelineOutput
from rag_eval.compare import DeltaStatus, ExampleChangeKind, compare_reports
from tests.conftest import StubPipeline

EXAMPLES = [
    EvalExample(id="a", question="qa", expected_answer="alpha", expected_context=["alpha"]),
    EvalExample(id="b", question="qb", expected_answer="beta", expected_context=["beta"]),
]


def _report(answers: dict[str, str], examples: list[EvalExample] = EXAMPLES) -> EvaluationReport:
    outputs = {
        example.question: PipelineOutput(answer=answers[example.id], contexts=[answers[example.id]])
        for example in examples
    }
    return Evaluator(StubPipeline(outputs), EvalConfig(top_k=1)).evaluate(
        Dataset(examples=examples)
    )


def test_metric_deltas_and_statuses() -> None:
    baseline = _report({"a": "alpha", "b": "wrong"})
    candidate = _report({"a": "alpha", "b": "beta"})

    comparison = compare_reports(baseline, candidate)
    deltas = {metric.name: metric for metric in comparison.metrics}

    assert deltas["recall_at_k"].baseline == pytest.approx(0.5)
    assert deltas["recall_at_k"].candidate == pytest.approx(1.0)
    assert deltas["recall_at_k"].delta == pytest.approx(0.5)
    assert deltas["recall_at_k"].status is DeltaStatus.IMPROVED
    assert deltas["pass_rate"].delta == pytest.approx(0.5)
    assert comparison.regressions == []
    assert comparison.warnings == []


def test_regressions_and_example_changes() -> None:
    baseline = _report({"a": "alpha", "b": "beta"})
    candidate = _report({"a": "alpha", "b": "wrong"})

    comparison = compare_reports(baseline, candidate)

    assert {metric.name for metric in comparison.regressions} >= {"recall_at_k", "token_f1"}
    assert [(change.id, change.kind) for change in comparison.examples] == [
        ("b", ExampleChangeKind.BROKEN)
    ]


def test_tolerance_suppresses_small_changes() -> None:
    baseline = _report({"a": "alpha", "b": "beta"})
    candidate = _report({"a": "alpha", "b": "wrong"})

    comparison = compare_reports(baseline, candidate, tolerance=0.6)

    assert comparison.regressions == []


def test_added_and_removed_examples_and_dataset_warning() -> None:
    extra = EvalExample(id="c", question="qc", expected_answer="gamma", expected_context=["gamma"])
    baseline = _report({"a": "alpha", "b": "beta"})
    candidate = _report({"a": "alpha", "c": "gamma"}, [EXAMPLES[0], extra])

    comparison = compare_reports(baseline, candidate)

    assert [(change.id, change.kind) for change in comparison.examples] == [
        ("c", ExampleChangeKind.ADDED),
        ("b", ExampleChangeKind.REMOVED),
    ]
    assert "reports were produced from different datasets" in comparison.warnings


def test_identical_reports_are_unchanged() -> None:
    report = _report({"a": "alpha", "b": "beta"})

    comparison = compare_reports(report, report)

    assert all(metric.status is DeltaStatus.UNCHANGED for metric in comparison.metrics)
    assert comparison.examples == []
