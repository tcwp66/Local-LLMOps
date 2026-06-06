from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from backend.config import PROJECT_ROOT

from .dataset import CorpusChunk, EvalCase, load_corpus, load_eval_cases
from .metrics import AggregateMetrics, CaseMetrics, aggregate, score_case
from .pipelines import DEFAULT_PIPELINES, EvalPipeline, PipelineResult


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evalbench"


class EvalRun(BaseModel):
    run_id: str
    created_at: str
    case_count: int
    pipelines: list[str]
    aggregate_metrics: list[AggregateMetrics]
    case_metrics: list[CaseMetrics]
    results: list[dict]


def run_eval(
    *,
    cases: list[EvalCase] | None = None,
    corpus: list[CorpusChunk] | None = None,
    pipelines: list[EvalPipeline] | None = None,
) -> EvalRun:
    active_cases = cases or load_eval_cases()
    active_corpus = corpus or load_corpus()
    active_pipelines = pipelines or DEFAULT_PIPELINES
    created_at = datetime.now(UTC).isoformat()
    run_id = created_at.replace(":", "").replace("+", "Z")

    results: list[PipelineResult] = []
    metrics: list[CaseMetrics] = []
    for pipeline in active_pipelines:
        for case in active_cases:
            try:
                result = pipeline.run(case, active_corpus)
            except Exception as exc:
                result = PipelineResult(
                    case_id=case.id,
                    pipeline=pipeline.name,
                    answer="",
                    contexts=[],
                    tool_calls=[],
                    latency_ms=0.0,
                    error=str(exc),
                )
            results.append(result)
            metrics.append(score_case(case, result))

    return EvalRun(
        run_id=run_id,
        created_at=created_at,
        case_count=len(active_cases),
        pipelines=[pipeline.name for pipeline in active_pipelines],
        aggregate_metrics=aggregate(metrics),
        case_metrics=metrics,
        results=[
            {
                "case_id": result.case_id,
                "pipeline": result.pipeline,
                "answer": result.answer,
                "source_ids": [context.source_id for context in result.contexts],
                "tool_calls": [call.__dict__ for call in result.tool_calls],
                "latency_ms": result.latency_ms,
                "error": result.error,
            }
            for result in results
        ],
    )


def save_eval_run(run: EvalRun, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    run_path = path / f"{run.run_id}.json"
    run_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    latest_path = path / "latest.json"
    latest_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run_path
