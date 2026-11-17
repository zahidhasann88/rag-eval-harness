from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from rag_eval.models import EvalExample, PipelineOutput
from rag_eval.plugins import PluginLoadError, import_object, materialize


@runtime_checkable
class AnswerJudge(Protocol):
    """Extension point for external evaluators, such as an LLM-as-judge.

    ``score`` returns named scores; they are stored per example as ``<judge name>.<score name>``
    and averaged in the report alongside the built-in metrics.
    """

    name: str

    def score(self, example: EvalExample, output: PipelineOutput) -> Mapping[str, float]: ...


class JudgeLoadError(Exception):
    pass


def load_judge(spec: str) -> AnswerJudge:
    try:
        judge = materialize(import_object(spec))
    except PluginLoadError as exc:
        raise JudgeLoadError(str(exc)) from exc
    if not isinstance(judge, AnswerJudge):
        raise JudgeLoadError(
            f"{spec!r} does not satisfy AnswerJudge: needs a 'name' attribute and "
            "a score(example, output) method"
        )
    return judge
