from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import Settings
from backend.db import get_chat_log, get_summary
from backend.main import create_app


def test_chat_endpoint_logs_trace(tmp_path: Path) -> None:
    settings = Settings(provider="echo", db_path=tmp_path / "llmops.sqlite")
    client = TestClient(create_app(settings))

    response = client.post(
        "/chat",
        json={
            "prompt": "What does model deviation mean in DP-GEN?",
            "source_ids": ["deepmd-docs:model-deviation"],
            "tool_calls": [{"name": "rag_search", "status": "ok", "latency_ms": 5.0}],
            "metadata": {"task_type": "qa"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "echo"
    assert body["request_id"]
    assert body["source_ids"] == ["deepmd-docs:model-deviation"]

    stored = get_chat_log(settings.db_path, body["request_id"])
    assert stored is not None
    assert stored["status"] == "ok"
    assert stored["tool_calls"][0]["name"] == "rag_search"
    assert stored["metadata"]["task_type"] == "qa"


def test_logs_and_metrics(tmp_path: Path) -> None:
    settings = Settings(provider="echo", db_path=tmp_path / "llmops.sqlite")
    client = TestClient(create_app(settings))
    client.post("/chat", json={"prompt": "hello"})

    logs = client.get("/logs").json()
    assert len(logs) == 1

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "llmops_requests_total 1" in metrics.text

    summary = get_summary(settings.db_path)
    assert summary["total_requests"] == 1
    assert summary["failed_requests"] == 0
