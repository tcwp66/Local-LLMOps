from __future__ import annotations

from pathlib import Path

from .runner import EvalRun


METRIC_COLUMNS = [
    "pipeline",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "tool_call_success",
    "avg_latency_ms",
    "error_rate",
]


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(run: EvalRun) -> str:
    lines = [
        "# LLM-EvalBench Report",
        "",
        f"- Run ID: `{run.run_id}`",
        f"- Created at: `{run.created_at}`",
        f"- Cases: `{run.case_count}`",
        f"- Pipelines: `{', '.join(run.pipelines)}`",
        "",
        "## Pipeline Comparison",
        "",
        "| " + " | ".join(METRIC_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in METRIC_COLUMNS) + " |",
    ]
    for item in run.aggregate_metrics:
        payload = item.model_dump()
        lines.append("| " + " | ".join(_fmt(payload[column]) for column in METRIC_COLUMNS) + " |")

    lines.extend(["", "## Bad Cases", ""])
    bad_cases = [
        metric
        for metric in run.case_metrics
        if metric.error
        or metric.hit_at_3 < 1.0
        or metric.context_recall < 1.0
        or metric.answer_relevancy < 0.75
        or metric.faithfulness < 0.75
    ]
    if not bad_cases:
        lines.append("No bad cases under the current thresholds.")
    else:
        lines.append("| case_id | pipeline | hit@3 | context_recall | faithfulness | answer_relevancy | error |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for metric in bad_cases:
            lines.append(
                "| "
                + " | ".join(
                    [
                        metric.case_id,
                        metric.pipeline,
                        _fmt(metric.hit_at_3),
                        _fmt(metric.context_recall),
                        _fmt(metric.faithfulness),
                        _fmt(metric.answer_relevancy),
                        metric.error or "",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def save_markdown_report(run: EvalRun, output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / f"{run.run_id}.md"
    report_path.write_text(render_markdown(run), encoding="utf-8")
    latest_path = path / "latest.md"
    latest_path.write_text(render_markdown(run), encoding="utf-8")
    return report_path
