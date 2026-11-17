import sys
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from rag_eval._version import __version__
from rag_eval.compare import compare_reports
from rag_eval.dataset import (
    DatasetError,
    find_unknown_doc_ids,
    load_corpus,
    load_dataset,
    pooled_corpus,
    validate_dataset,
)
from rag_eval.engine import EmptyDatasetError, Evaluator
from rag_eval.judges import AnswerJudge, JudgeLoadError, load_judge
from rag_eval.models import Corpus, Dataset, EvalConfig
from rag_eval.pipelines import BUILTIN_PIPELINES, PipelineLoadError, load_pipeline
from rag_eval.reporting import ReportError, load_json_report, write_report
from rag_eval.reporting.terminal import print_comparison, print_report_summary, print_validation

EXIT_CHECK_FAILED = 1
EXIT_USAGE = 2

app = typer.Typer(
    name="rag-eval",
    help="Evaluate retrieval and answer quality of RAG pipelines.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()
error_console = Console(stderr=True)

DatasetOption = Annotated[
    Path,
    typer.Option("--dataset", "-d", help="JSONL evaluation dataset.", dir_okay=False),
]
CorpusOption = Annotated[
    Path | None,
    typer.Option(
        "--corpus",
        "-c",
        help="JSONL corpus for built-in pipelines. Defaults to the dataset's expected contexts.",
        dir_okay=False,
    ),
]


def _fail(message: str, code: int = EXIT_USAGE) -> typer.Exit:
    error_console.print(f"[red]error:[/] {message}", highlight=False)
    return typer.Exit(code)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"rag-eval {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def _load_inputs(
    dataset_path: Path, corpus_path: Path | None, pipeline: str
) -> tuple[Dataset, Corpus | None]:
    try:
        dataset = load_dataset(dataset_path)
        corpus = load_corpus(corpus_path) if corpus_path else None
    except DatasetError as exc:
        for issue in exc.issues:
            error_console.print(f"[red]error[/] {exc.path}: {issue}", highlight=False)
        raise typer.Exit(EXIT_USAGE) from exc

    if corpus is None and pipeline in BUILTIN_PIPELINES:
        if any(example.relevant_doc_ids for example in dataset.examples):
            raise _fail("dataset uses relevant_doc_ids; pass --corpus with matching document ids")
        corpus = pooled_corpus(dataset)
        if not corpus.documents:
            raise _fail("dataset has no expected_context to pool into a corpus; pass --corpus")
        console.print(
            f"[dim]No --corpus given: indexing {len(corpus.documents)} expected contexts "
            "pooled from the dataset.[/]"
        )
    return dataset, corpus


@app.command()
def run(
    dataset: DatasetOption,
    pipeline: Annotated[
        str,
        typer.Option(
            "--pipeline", "-p", help="Built-in pipeline (mock, tfidf) or module:attribute."
        ),
    ] = "mock",
    corpus: CorpusOption = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1, help="Contexts to score.")] = 5,
    report: Annotated[
        list[Path] | None,
        typer.Option(
            "--report", "-r", help="Write a report; format from extension (.json, .html)."
        ),
    ] = None,
    batch_size: Annotated[int, typer.Option(min=1, help="Questions per pipeline batch.")] = 16,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Evaluate only the first N examples.")
    ] = None,
    context_match_threshold: Annotated[
        float, typer.Option(min=0.01, max=1.0, help="Term coverage for a context match.")
    ] = 0.75,
    min_recall: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    min_f1: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    min_groundedness: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    judge: Annotated[
        list[str] | None,
        typer.Option("--judge", "-j", help="External AnswerJudge as module:attribute."),
    ] = None,
    show_failures: Annotated[int, typer.Option(min=0, help="Failed cases to print.")] = 10,
    fail_on_failure: Annotated[
        bool, typer.Option(help="Exit with status 1 when any example fails.")
    ] = False,
) -> None:
    """Run a pipeline over a dataset and score it."""
    data, documents = _load_inputs(dataset, corpus, pipeline)
    if limit is not None:
        data = data.head(limit)

    try:
        config = EvalConfig(
            top_k=top_k,
            batch_size=batch_size,
            context_match_threshold=context_match_threshold,
            min_recall=min_recall,
            min_answer_f1=min_f1,
            min_groundedness=min_groundedness,
        )
        rag = load_pipeline(pipeline, documents.documents if documents else ())
        judges: list[AnswerJudge] = [load_judge(spec) for spec in judge or []]
    except (ValidationError, PipelineLoadError, JudgeLoadError) as exc:
        raise _fail(str(exc)) from exc

    evaluator = Evaluator(rag, config, judges)
    try:
        with Progress(
            TextColumn("Evaluating"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("evaluate", total=len(data.examples))
            result = evaluator.evaluate(
                data,
                corpus_source=documents.source if documents else None,
                progress=lambda done: progress.advance(task, done),
            )
    except EmptyDatasetError as exc:
        raise _fail(str(exc)) from exc

    print_report_summary(result, console, show_failures)
    for path in report or []:
        try:
            written = write_report(result, path)
        except (ReportError, OSError) as exc:
            raise _fail(str(exc)) from exc
        console.print(f"Report written to {written}")

    if fail_on_failure and result.summary.num_failed:
        raise typer.Exit(EXIT_CHECK_FAILED)


@app.command()
def validate(
    dataset: DatasetOption,
    corpus: CorpusOption = None,
    strict: Annotated[bool, typer.Option(help="Treat warnings as errors.")] = False,
) -> None:
    """Check a dataset (and optionally its corpus) for schema and ground-truth problems."""
    validation = validate_dataset(dataset)
    if corpus is not None:
        try:
            documents = load_corpus(corpus)
        except DatasetError as exc:
            validation.errors.extend(exc.issues)
        else:
            validation.errors.extend(find_unknown_doc_ids(validation.examples, documents))
    print_validation(validation, console)
    if not validation.ok or (strict and validation.warnings):
        raise typer.Exit(EXIT_CHECK_FAILED)


@app.command()
def compare(
    baseline: Annotated[Path, typer.Option(help="Baseline JSON report.", dir_okay=False)],
    candidate: Annotated[Path, typer.Option(help="Candidate JSON report.", dir_okay=False)],
    tolerance: Annotated[
        float, typer.Option(min=0.0, help="Absolute change treated as unchanged.")
    ] = 0.0,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the comparison as JSON.")
    ] = None,
    fail_on_regression: Annotated[
        bool, typer.Option(help="Exit with status 1 when any metric regresses.")
    ] = False,
) -> None:
    """Show metric changes between two JSON reports."""
    try:
        comparison = compare_reports(
            load_json_report(baseline), load_json_report(candidate), tolerance
        )
    except ReportError as exc:
        raise _fail(str(exc)) from exc

    print_comparison(comparison, console)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(comparison.model_dump_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"Comparison written to {output}")
    if fail_on_regression and comparison.regressions:
        raise typer.Exit(EXIT_CHECK_FAILED)
