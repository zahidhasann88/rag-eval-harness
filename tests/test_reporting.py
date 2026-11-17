import json
import re
from pathlib import Path

import pytest

from rag_eval import Dataset, EvalConfig, EvalExample, EvaluationReport, Evaluator, PipelineOutput
from rag_eval.reporting import ReportError, load_json_report, render_html, write_report
from tests.conftest import StubPipeline


@pytest.fixture
def report() -> EvaluationReport:
    examples = [
        EvalExample(
            id="ok",
            question="What is <b>bold</b>?",
            expected_answer="tag",
            expected_context=["tag"],
        ),
        EvalExample(
            id="miss", question="Where?", expected_answer="here", expected_context=["here"]
        ),
    ]
    pipeline = StubPipeline(
        {
            "What is <b>bold</b>?": PipelineOutput(
                answer="tag", contexts=["tag <script>alert(1)</script>"]
            ),
            "Where?": PipelineOutput(answer="nowhere", contexts=[]),
        }
    )
    return Evaluator(pipeline, EvalConfig(top_k=3)).evaluate(Dataset(examples=examples))


def test_json_report_round_trips(report: EvaluationReport, tmp_path: Path) -> None:
    path = write_report(report, tmp_path / "nested" / "report.json")

    loaded = load_json_report(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == report
    assert payload["schema_version"] == 1
    assert payload["results"][1]["failure"]["stage"] == "retrieval"


def test_html_report_is_self_contained_and_escaped(report: EvaluationReport) -> None:
    html = render_html(report)

    assert html.startswith("<!doctype html>")
    assert "<script" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "What is &lt;b&gt;bold&lt;/b&gt;?" in html
    assert not re.search(r"""(src|href)=["']?(https?:)?//""", html)
    assert "Recall@3" in html
    assert "Heuristic approximation" in html
    assert "Failed cases (1)" in html


def test_write_report_rejects_unknown_extension(report: EvaluationReport, tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="unsupported report format"):
        write_report(report, tmp_path / "report.csv")


def test_load_json_report_rejects_foreign_json(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")

    with pytest.raises(ReportError, match="schema_version"):
        load_json_report(path)
