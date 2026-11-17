import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval import (
    Document,
    FunctionPipeline,
    MockPipeline,
    PipelineOutput,
    RAGPipeline,
    TfidfPipeline,
    load_pipeline,
)
from rag_eval.pipelines import PipelineLoadError, TfidfIndex

DOCUMENTS = [
    Document(id="dlq", text="A dead letter queue stores messages that failed processing."),
    Document(id="cb", text="A circuit breaker stops calls to a failing dependency."),
    Document(id="mq", text="A message queue buffers messages. Consumers acknowledge them."),
]


@pytest.mark.parametrize("factory", [MockPipeline, TfidfPipeline])
def test_builtin_pipelines_satisfy_contract(factory: type[MockPipeline | TfidfPipeline]) -> None:
    pipeline = factory(DOCUMENTS)

    output = pipeline.run("What does a dead letter queue store?", top_k=2)

    assert isinstance(pipeline, RAGPipeline)
    assert isinstance(output, PipelineOutput)
    assert 0 < len(output.contexts) <= 2
    assert output.contexts[0].doc_id == "dlq"
    assert output.model is not None
    assert pipeline.run("What does a dead letter queue store?", top_k=2) == output


def test_pipelines_return_no_contexts_for_unrelated_questions() -> None:
    for pipeline in (MockPipeline(DOCUMENTS), TfidfPipeline(DOCUMENTS)):
        output = pipeline.run("zebra xylophone", top_k=3)
        assert output.contexts == []
        assert output.answer == ""


def test_tfidf_answer_is_extracted_sentence() -> None:
    output = TfidfPipeline(DOCUMENTS).run("Do consumers acknowledge messages?", top_k=3)

    assert output.answer == "Consumers acknowledge them."


def test_tfidf_index_scores_are_cosine_bounded() -> None:
    index = TfidfIndex([doc.text for doc in DOCUMENTS])

    hits = index.search(DOCUMENTS[1].text, k=3)

    assert hits[0][0] == 1
    assert hits[0][1] == pytest.approx(1.0)
    assert all(0 < score <= 1.0 + 1e-9 for _, score in hits)


def test_pipeline_output_coerces_string_contexts() -> None:
    output = PipelineOutput.model_validate({"answer": "a", "contexts": ["one", {"text": "two"}]})

    assert [context.text for context in output.contexts] == ["one", "two"]


def test_pipeline_output_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        PipelineOutput(answer="a", latency_ms=-1)


def test_function_pipeline_accepts_dict_results() -> None:
    def answer(question: str, top_k: int) -> dict[str, object]:
        return {"answer": question.upper(), "contexts": ["ctx"] * top_k, "model": "fn"}

    pipeline = FunctionPipeline(answer)
    output = pipeline.run("hi", top_k=2)

    assert isinstance(pipeline, RAGPipeline)
    assert pipeline.name == "answer"
    assert output.answer == "HI"
    assert len(output.contexts) == 2


def test_load_builtin_requires_corpus() -> None:
    assert isinstance(load_pipeline("tfidf", DOCUMENTS), TfidfPipeline)
    with pytest.raises(PipelineLoadError, match="non-empty corpus"):
        load_pipeline("mock", [])


def test_load_pipeline_from_import_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "my_rag.py").write_text(
        "from rag_eval import PipelineOutput\n"
        "class App:\n"
        "    name = 'my-app'\n"
        "    def run(self, question, *, top_k):\n"
        "        return PipelineOutput(answer=question)\n"
        "def build():\n"
        "    return App()\n"
        "not_a_pipeline = 42\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "my_rag", raising=False)

    assert load_pipeline("my_rag:App").name == "my-app"
    assert load_pipeline("my_rag:build").name == "my-app"
    with pytest.raises(PipelineLoadError, match="does not satisfy RAGPipeline"):
        load_pipeline("my_rag:not_a_pipeline")
    with pytest.raises(PipelineLoadError, match="cannot import"):
        load_pipeline("my_rag:missing")
    with pytest.raises(PipelineLoadError, match="expected 'module:attribute'"):
        load_pipeline("unknown")
