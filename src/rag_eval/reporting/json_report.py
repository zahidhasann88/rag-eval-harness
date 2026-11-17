import json
from pathlib import Path

from pydantic import ValidationError

from rag_eval.results import SCHEMA_VERSION, EvaluationReport


class ReportError(Exception):
    pass


def to_json(report: EvaluationReport) -> str:
    return report.model_dump_json(indent=2)


def write_json_report(report: EvaluationReport, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(report) + "\n", encoding="utf-8")
    return path


def load_json_report(path: Path | str) -> EvaluationReport:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReportError(f"{path}: cannot read report: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ReportError(f"{path}: not a rag-eval report with schema_version {SCHEMA_VERSION}")
    try:
        return EvaluationReport.model_validate(payload)
    except ValidationError as exc:
        raise ReportError(
            f"{path}: malformed report: {exc.error_count()} validation errors"
        ) from exc
