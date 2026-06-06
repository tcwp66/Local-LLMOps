from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import quantiles
from typing import Any


@dataclass
class ChatLog:
    request_id: str
    created_at: str
    provider: str
    model: str
    prompt: str
    response: str
    status: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    source_ids: list[str] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_requests (
                request_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                error TEXT,
                source_ids TEXT NOT NULL,
                tool_calls TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_requests_created_at ON chat_requests(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_requests_status ON chat_requests(status)")


def insert_chat_log(db_path: str | Path, log: ChatLog) -> None:
    init_db(db_path)
    payload = asdict(log)
    payload["source_ids"] = json.dumps(log.source_ids or [], ensure_ascii=False)
    payload["tool_calls"] = json.dumps(log.tool_calls or [], ensure_ascii=False)
    payload["metadata"] = json.dumps(log.metadata or {}, ensure_ascii=False)

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_requests (
                request_id, created_at, provider, model, prompt, response, status,
                latency_ms, prompt_tokens, completion_tokens, error, source_ids,
                tool_calls, metadata
            )
            VALUES (
                :request_id, :created_at, :provider, :model, :prompt, :response,
                :status, :latency_ms, :prompt_tokens, :completion_tokens, :error,
                :source_ids, :tool_calls, :metadata
            )
            """,
            payload,
        )


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["source_ids"] = json.loads(item["source_ids"])
    item["tool_calls"] = json.loads(item["tool_calls"])
    item["metadata"] = json.loads(item["metadata"])
    return item


def list_chat_logs(db_path: str | Path, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    bounded_limit = max(1, min(limit, 500))
    with connect(db_path) as conn:
        if status:
            rows: Iterable[sqlite3.Row] = conn.execute(
                """
                SELECT * FROM chat_requests
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (status, bounded_limit),
            )
        else:
            rows = conn.execute(
                "SELECT * FROM chat_requests ORDER BY created_at DESC LIMIT ?",
                (bounded_limit,),
            )
        return [_decode_row(row) for row in rows]


def get_chat_log(db_path: str | Path, request_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM chat_requests WHERE request_id = ?", (request_id,)).fetchone()
        return _decode_row(row) if row else None


def get_summary(db_path: str | Path) -> dict[str, float | int]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT status, latency_ms FROM chat_requests").fetchall()

    total = len(rows)
    failures = sum(1 for row in rows if row["status"] != "ok")
    latencies = [float(row["latency_ms"]) for row in rows]
    avg_latency = sum(latencies) / total if total else 0.0
    p95_latency = quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0.0)

    return {
        "total_requests": total,
        "successful_requests": total - failures,
        "failed_requests": failures,
        "success_rate": ((total - failures) / total) if total else 1.0,
        "error_rate": (failures / total) if total else 0.0,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
    }
