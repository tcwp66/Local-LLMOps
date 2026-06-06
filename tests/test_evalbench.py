from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import Settings
from backend.main import create_app
from evalbench.dataset import load_corpus, load_eval_cases
from evalbench.reporting import render_markdown
from evalbench.runner import run_eval, save_eval_run


def test_evalbench_runs_all_default_pipelines() -> None:
    run = run_eval()

    assert run.case_count == 4
    assert set(run.pipelines) == {
        "no_rag",
        "naive_rag",
        "rag_metadata_filter",
        "rag_reranker",
        "rag_tools",
    }
    assert len(run.aggregate_metrics) == 5
    assert len(run.case_metrics) == run.case_count * len(run.pipelines)

    reranker = next(item for item in run.aggregate_metrics if item.pipeline == "rag_reranker")
    no_rag = next(item for item in run.aggregate_metrics if item.pipeline == "no_rag")
    assert reranker.context_recall >= no_rag.context_recall
    assert reranker.answer_relevancy >= no_rag.answer_relevancy


def test_evalbench_tool_pipeline_scores_tool_success() -> None:
    run = run_eval()
    tools = next(item for item in run.aggregate_metrics if item.pipeline == "rag_tools")

    assert tools.tool_call_success == 1.0


def test_evalbench_report_can_be_saved(tmp_path: Path) -> None:
    run = run_eval(cases=load_eval_cases(), corpus=load_corpus())
    report_path = save_eval_run(run, tmp_path)
    markdown = render_markdown(run)

    assert report_path.exists()
    assert (tmp_path / "latest.json").exists()
    assert "LLM-EvalBench Report" in markdown
    assert "Pipeline Comparison" in markdown


def test_evalbench_api_endpoint(tmp_path: Path) -> None:
    settings = Settings(provider="echo", db_path=tmp_path / "llmops.sqlite")
    client = TestClient(create_app(settings))

    response = client.post("/evalbench/run")

    assert response.status_code == 200
    body = response.json()
    assert body["case_count"] == 4
    assert len(body["aggregate_metrics"]) == 5
