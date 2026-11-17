from rich.console import Console
from rich.table import Table

from rag_eval.compare import Comparison, DeltaStatus
from rag_eval.dataset import DatasetValidation
from rag_eval.results import HEURISTIC_METRICS, EvaluationReport, FailureStage, metric_label

_STATUS_STYLE = {
    DeltaStatus.IMPROVED: "green",
    DeltaStatus.REGRESSED: "red",
    DeltaStatus.UNCHANGED: "dim",
    DeltaStatus.UNAVAILABLE: "dim",
}
_STAGE_STYLE = {
    FailureStage.RETRIEVAL: "red",
    FailureStage.GENERATION: "yellow",
    FailureStage.PIPELINE_ERROR: "red",
}


def _score(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_report_summary(
    report: EvaluationReport, console: Console, max_failures: int = 10
) -> None:
    meta, summary = report.metadata, report.summary
    top_k = meta.config.top_k
    corpus = (meta.corpus.path or "pooled from dataset") if meta.corpus else "n/a"
    console.print(
        f"[bold]{meta.pipeline}[/] on {meta.dataset.path or 'in-memory dataset'} "
        f"(corpus: {corpus}, top-k {top_k}, fingerprint {meta.fingerprint})"
    )

    table = Table(title="Metrics", title_justify="left")
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    table.add_column("N", justify="right")
    for name, metric in report.metrics.items():
        label = metric_label(name, top_k)
        if name in HEURISTIC_METRICS:
            label += " [dim](heuristic)[/]"
        table.add_row(label, _score(metric.mean), str(metric.count))
    console.print(table)

    stages = ", ".join(
        f"{stage.value} {count}" for stage, count in summary.failures_by_stage.items() if count
    )
    console.print(
        f"Passed [green]{summary.num_passed}[/] / {summary.num_examples}"
        f"   Failed [red]{summary.num_failed}[/]" + (f" ({stages})" if stages else "")
    )
    latency = report.latency
    if latency.count:
        console.print(
            f"Latency p50 {_score(latency.p50_ms, 1)} ms, p95 {_score(latency.p95_ms, 1)} ms"
        )
    if report.tokens.examples_with_usage:
        console.print(f"Tokens {report.tokens.total_tokens} total")

    failures = report.failures
    if not failures or max_failures <= 0:
        return
    failed = Table(title="Failed cases", title_justify="left")
    failed.add_column("ID", overflow="fold")
    failed.add_column("Stage")
    failed.add_column("Question", overflow="fold")
    failed.add_column("Reasons", overflow="fold")
    for result in failures[:max_failures]:
        assert result.failure is not None
        stage = result.failure.stage
        failed.add_row(
            result.id,
            f"[{_STAGE_STYLE[stage]}]{stage.value}[/]",
            result.question,
            "; ".join(result.failure.reasons),
        )
    console.print(failed)
    if len(failures) > max_failures:
        console.print(f"[dim]{len(failures) - max_failures} more failures in the full report[/]")


def print_comparison(comparison: Comparison, console: Console) -> None:
    baseline = f"[bold]{comparison.baseline_pipeline}[/] ({comparison.baseline_fingerprint})"
    candidate = f"[bold]{comparison.candidate_pipeline}[/] ({comparison.candidate_fingerprint})"
    console.print(f"Baseline {baseline} vs candidate {candidate}")
    for warning in comparison.warnings:
        console.print(f"[yellow]warning:[/] {warning}")

    table = Table(title="Metric changes", title_justify="left")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status")
    for metric in comparison.metrics:
        style = _STATUS_STYLE[metric.status]
        delta = "-" if metric.delta is None else f"{metric.delta:+.3f}"
        table.add_row(
            metric_label(metric.name, comparison.top_k),
            _score(metric.baseline),
            _score(metric.candidate),
            f"[{style}]{delta}[/]",
            f"[{style}]{metric.status.value}[/]",
        )
    console.print(table)

    if not comparison.examples:
        console.print("No per-question status changes.")
        return
    changes = Table(title="Per-question status changes", title_justify="left")
    changes.add_column("ID", overflow="fold")
    changes.add_column("Change")
    changes.add_column("Baseline")
    changes.add_column("Candidate")
    changes.add_column("Question", overflow="fold")
    for change in comparison.examples:
        changes.add_row(
            change.id,
            change.kind.value,
            change.baseline_stage.value if change.baseline_stage else "pass",
            change.candidate_stage.value if change.candidate_stage else "pass",
            change.question,
        )
    console.print(changes)


def print_validation(validation: DatasetValidation, console: Console) -> None:
    for issue in validation.errors:
        console.print(f"[red]error[/]   {issue}", highlight=False)
    for issue in validation.warnings:
        console.print(f"[yellow]warning[/] {issue}", highlight=False)
    count = len(validation.examples)
    if validation.ok:
        console.print(
            f"[green]valid[/] {validation.path}: {count} examples, "
            f"{len(validation.warnings)} warnings"
        )
    else:
        console.print(
            f"[red]invalid[/] {validation.path}: {len(validation.errors)} errors, "
            f"{count} parseable examples"
        )
