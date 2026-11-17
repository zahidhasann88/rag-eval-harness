from pathlib import Path

from rag_eval.reporting.html_report import render_html, write_html_report
from rag_eval.reporting.json_report import (
    ReportError,
    load_json_report,
    to_json,
    write_json_report,
)
from rag_eval.results import EvaluationReport

REPORT_WRITERS = {".json": write_json_report, ".html": write_html_report, ".htm": write_html_report}


def write_report(report: EvaluationReport, path: Path | str) -> Path:
    """Write a report, choosing the format from the file extension."""
    path = Path(path)
    writer = REPORT_WRITERS.get(path.suffix.lower())
    if writer is None:
        supported = ", ".join(sorted(REPORT_WRITERS))
        raise ReportError(f"{path}: unsupported report format (use {supported})")
    return writer(report, path)


__all__ = [
    "ReportError",
    "load_json_report",
    "render_html",
    "to_json",
    "write_html_report",
    "write_json_report",
    "write_report",
]
