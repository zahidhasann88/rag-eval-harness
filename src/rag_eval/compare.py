from enum import StrEnum

from pydantic import BaseModel, Field

from rag_eval.results import EvaluationReport, FailureStage

_EPSILON = 1e-12


class DeltaStatus(StrEnum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    UNAVAILABLE = "unavailable"


class ExampleChangeKind(StrEnum):
    FIXED = "fixed"
    BROKEN = "broken"
    STAGE_CHANGED = "stage_changed"
    ADDED = "added"
    REMOVED = "removed"


class MetricDelta(BaseModel):
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    status: DeltaStatus


class ExampleChange(BaseModel):
    id: str
    question: str
    kind: ExampleChangeKind
    baseline_stage: FailureStage | None = None
    candidate_stage: FailureStage | None = None


class Comparison(BaseModel):
    baseline_fingerprint: str
    candidate_fingerprint: str
    baseline_pipeline: str
    candidate_pipeline: str
    top_k: int
    tolerance: float
    metrics: list[MetricDelta]
    examples: list[ExampleChange]
    warnings: list[str] = Field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [metric for metric in self.metrics if metric.status is DeltaStatus.REGRESSED]


def _delta(
    name: str, baseline: float | None, candidate: float | None, tolerance: float
) -> MetricDelta:
    if baseline is None or candidate is None:
        return MetricDelta(
            name=name,
            baseline=baseline,
            candidate=candidate,
            delta=None,
            status=DeltaStatus.UNAVAILABLE,
        )
    delta = candidate - baseline
    if abs(delta) <= tolerance + _EPSILON:
        status = DeltaStatus.UNCHANGED
    elif delta > 0:
        status = DeltaStatus.IMPROVED
    else:
        status = DeltaStatus.REGRESSED
    return MetricDelta(
        name=name,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        status=status,
    )


def _warnings(baseline: EvaluationReport, candidate: EvaluationReport) -> list[str]:
    warnings = []
    base, cand = baseline.metadata, candidate.metadata
    if base.dataset.sha256 != cand.dataset.sha256:
        warnings.append("reports were produced from different datasets")
    if (base.corpus and base.corpus.sha256) != (cand.corpus and cand.corpus.sha256):
        warnings.append("reports were produced from different corpora")
    if base.config.top_k != cand.config.top_k:
        warnings.append(f"top_k differs ({base.config.top_k} vs {cand.config.top_k})")
    if base.config.context_match_threshold != cand.config.context_match_threshold:
        warnings.append("context_match_threshold differs; retrieval metrics are not comparable")
    return warnings


def _example_changes(
    baseline: EvaluationReport, candidate: EvaluationReport
) -> list[ExampleChange]:
    base_by_id = {result.id: result for result in baseline.results}
    cand_by_id = {result.id: result for result in candidate.results}
    changes = []
    for result in candidate.results:
        previous = base_by_id.get(result.id)
        new_stage = result.failure.stage if result.failure else None
        if previous is None:
            changes.append(
                ExampleChange(
                    id=result.id,
                    question=result.question,
                    kind=ExampleChangeKind.ADDED,
                    candidate_stage=new_stage,
                )
            )
            continue
        old_stage = previous.failure.stage if previous.failure else None
        if old_stage == new_stage:
            continue
        if old_stage is None:
            kind = ExampleChangeKind.BROKEN
        elif new_stage is None:
            kind = ExampleChangeKind.FIXED
        else:
            kind = ExampleChangeKind.STAGE_CHANGED
        changes.append(
            ExampleChange(
                id=result.id,
                question=result.question,
                kind=kind,
                baseline_stage=old_stage,
                candidate_stage=new_stage,
            )
        )
    changes.extend(
        ExampleChange(
            id=result.id,
            question=result.question,
            kind=ExampleChangeKind.REMOVED,
            baseline_stage=result.failure.stage if result.failure else None,
        )
        for result in baseline.results
        if result.id not in cand_by_id
    )
    return changes


def compare_reports(
    baseline: EvaluationReport, candidate: EvaluationReport, tolerance: float = 0.0
) -> Comparison:
    """Diff aggregate metrics and per-example pass/fail status between two reports."""
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    names = list(dict.fromkeys([*baseline.metrics, *candidate.metrics]))
    metrics = [
        _delta(
            name,
            baseline.metrics[name].mean if name in baseline.metrics else None,
            candidate.metrics[name].mean if name in candidate.metrics else None,
            tolerance=tolerance,
        )
        for name in names
    ]
    metrics.append(
        _delta(
            "pass_rate",
            baseline.summary.num_passed / baseline.summary.num_examples,
            candidate.summary.num_passed / candidate.summary.num_examples,
            tolerance=tolerance,
        )
    )
    return Comparison(
        baseline_fingerprint=baseline.metadata.fingerprint,
        candidate_fingerprint=candidate.metadata.fingerprint,
        baseline_pipeline=baseline.metadata.pipeline,
        candidate_pipeline=candidate.metadata.pipeline,
        top_k=candidate.metadata.config.top_k,
        tolerance=tolerance,
        metrics=metrics,
        examples=_example_changes(baseline, candidate),
        warnings=_warnings(baseline, candidate),
    )
