import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from rag_eval import PipelineOutput

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

WriteJsonl = Callable[[str, Iterable[Mapping[str, Any] | str]], Path]


@pytest.fixture
def write_jsonl(tmp_path: Path) -> WriteJsonl:
    def write(name: str, rows: Iterable[Mapping[str, Any] | str]) -> Path:
        path = tmp_path / name
        lines = [row if isinstance(row, str) else json.dumps(row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return write


class StubPipeline:
    """Returns canned outputs keyed by question so metric values can be computed by hand."""

    name = "stub"

    def __init__(self, outputs: Mapping[str, PipelineOutput]) -> None:
        self.outputs = dict(outputs)
        self.calls: list[tuple[str, int]] = []

    def run(self, question: str, *, top_k: int) -> PipelineOutput:
        self.calls.append((question, top_k))
        return self.outputs[question]
