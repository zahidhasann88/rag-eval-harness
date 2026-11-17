from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined

from rag_eval.results import (
    ANSWER_METRICS,
    HEURISTIC_METRICS,
    RETRIEVAL_METRICS,
    EvaluationReport,
    metric_label,
)


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} ms"


_environment = Environment(
    loader=PackageLoader("rag_eval", "reporting/templates"),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)
_environment.filters["score"] = _format_score
_environment.filters["ms"] = _format_ms


def render_html(report: EvaluationReport) -> str:
    """Render a single self-contained HTML document: inline CSS, no scripts or external assets."""
    top_k = report.metadata.config.top_k
    judge_metrics = [name for name in report.metrics if name.startswith("judge.")]
    groups = [
        ("Retrieval", RETRIEVAL_METRICS),
        ("Answer", ANSWER_METRICS),
        ("External judges", tuple(judge_metrics)),
    ]
    return _environment.get_template("report.html.j2").render(
        report=report,
        meta=report.metadata,
        groups=[(title, names) for title, names in groups if names],
        label=lambda name: metric_label(name, top_k),
        heuristic=HEURISTIC_METRICS,
    )


def write_html_report(report: EvaluationReport, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(report), encoding="utf-8")
    return path
