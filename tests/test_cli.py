import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_eval.cli import app
from tests.conftest import EXAMPLES_DIR, WriteJsonl

runner = CliRunner()
DATASET = str(EXAMPLES_DIR / "dataset.jsonl")
CORPUS = str(EXAMPLES_DIR / "corpus.jsonl")


def test_validate_example_dataset() -> None:
    result = runner.invoke(app, ["validate", "--dataset", DATASET, "--corpus", CORPUS])

    assert result.exit_code == 0, result.output
    assert "valid" in result.output


def test_validate_reports_errors(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl("bad.jsonl", ['{"question": "ok"}', "{broken"])

    result = runner.invoke(app, ["validate", "--dataset", str(path)])

    assert result.exit_code == 1
    assert "line 2" in result.output


def test_validate_strict_fails_on_warnings(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl("warn.jsonl", [{"question": "no ground truth"}])

    assert runner.invoke(app, ["validate", "--dataset", str(path)]).exit_code == 0
    assert runner.invoke(app, ["validate", "--dataset", str(path), "--strict"]).exit_code == 1


def test_run_writes_reports_and_compare_reads_them(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    html = tmp_path / "report.html"

    mock = runner.invoke(
        app, ["run", "-d", DATASET, "-c", CORPUS, "-p", "mock", "-k", "3", "-r", str(baseline)]
    )
    reports = ["-r", str(candidate), "-r", str(html)]
    tfidf = runner.invoke(
        app, ["run", "-d", DATASET, "-c", CORPUS, "-p", "tfidf", "-k", "3", *reports]
    )
    diff = tmp_path / "diff.json"
    compared = runner.invoke(
        app,
        ["compare", "--baseline", str(baseline), "--candidate", str(candidate), "-o", str(diff)],
    )

    assert mock.exit_code == 0, mock.output
    assert tfidf.exit_code == 0, tfidf.output
    assert compared.exit_code == 0, compared.output
    assert "Precision@3" in mock.output
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert json.loads(candidate.read_text(encoding="utf-8"))["metadata"]["pipeline"] == "tfidf"
    assert "Metric changes" in compared.output
    assert json.loads(diff.read_text(encoding="utf-8"))["baseline_pipeline"] == "mock"


def test_run_without_corpus_pools_expected_contexts() -> None:
    result = runner.invoke(app, ["run", "--dataset", DATASET, "--limit", "2"])

    assert result.exit_code == 0, result.output
    assert "pooled from the dataset" in result.output


def test_run_fail_on_failure_sets_exit_code(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl(
        "data.jsonl",
        [{"question": "zebra?", "expected_answer": "x", "expected_context": ["alpha beta"]}],
    )

    result = runner.invoke(app, ["run", "-d", str(path), "--fail-on-failure"])

    assert result.exit_code == 1, result.output


def test_run_rejects_empty_dataset(write_jsonl: WriteJsonl) -> None:
    path = write_jsonl("empty.jsonl", [""])

    result = runner.invoke(app, ["run", "--dataset", str(path)])

    assert result.exit_code == 2


def test_run_rejects_unknown_pipeline() -> None:
    result = runner.invoke(app, ["run", "--dataset", DATASET, "--pipeline", "nope"])

    assert result.exit_code == 2


def test_compare_fail_on_regression(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    runner.invoke(app, ["run", "-d", DATASET, "-c", CORPUS, "-p", "tfidf", "-r", str(good)])
    runner.invoke(
        app, ["run", "-d", DATASET, "-c", CORPUS, "-p", "tfidf", "-k", "1", "-r", str(bad)]
    )

    result = runner.invoke(
        app, ["compare", "--baseline", str(good), "--candidate", str(bad), "--fail-on-regression"]
    )

    assert result.exit_code == 1, result.output
    assert "top_k differs" in result.output


def test_run_with_external_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "my_judges.py").write_text(
        "class Brevity:\n"
        "    name = 'brevity'\n"
        "    def score(self, example, output):\n"
        "        return {'short': float(len(output.answer.split()) <= 25)}\n"
        "class NotAJudge:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "my_judges", raising=False)
    report = tmp_path / "judged.json"

    ok = runner.invoke(app, ["run", "-d", DATASET, "-j", "my_judges:Brevity", "-r", str(report)])
    bad = runner.invoke(app, ["run", "-d", DATASET, "-j", "my_judges:NotAJudge"])

    assert ok.exit_code == 0, ok.output
    assert "judge.brevity.short" in json.loads(report.read_text(encoding="utf-8"))["metrics"]
    assert bad.exit_code == 2
