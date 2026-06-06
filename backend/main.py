from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query, Response

from .config import Settings, load_settings
from .db import ChatLog, get_chat_log, get_summary, init_db, insert_chat_log, list_chat_logs
from .llm_clients import build_client, count_tokens
from .schemas import ChatRequest, ChatResponse, HealthResponse
from evalbench.reporting import save_markdown_report
from evalbench.runner import DEFAULT_OUTPUT_DIR, run_eval, save_eval_run


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_app(settings: Settings | None = None) -> FastAPI:
    active_settings = settings or load_settings()
    init_db(active_settings.db_path)
    client = build_client(active_settings)

    app = FastAPI(title=active_settings.app_name)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            app=active_settings.app_name,
            provider=active_settings.provider,
            db_path=str(active_settings.db_path),
            status="ok",
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        created_at = utc_now()

        try:
            result = client.generate(request.prompt, request.model)
            latency_ms = (time.perf_counter() - started) * 1000
            insert_chat_log(
                active_settings.db_path,
                ChatLog(
                    request_id=request_id,
                    created_at=created_at,
                    provider=active_settings.provider,
                    model=result.model,
                    prompt=request.prompt,
                    response=result.text,
                    status="ok",
                    latency_ms=latency_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    source_ids=request.source_ids,
                    tool_calls=[item.model_dump() for item in request.tool_calls],
                    metadata=request.metadata,
                ),
            )
            return ChatResponse(
                request_id=request_id,
                response=result.text,
                provider=active_settings.provider,
                model=result.model,
                latency_ms=latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                source_ids=request.source_ids,
                tool_calls=request.tool_calls,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            model = request.model or active_settings.ollama_model
            insert_chat_log(
                active_settings.db_path,
                ChatLog(
                    request_id=request_id,
                    created_at=created_at,
                    provider=active_settings.provider,
                    model=model,
                    prompt=request.prompt,
                    response="",
                    status="error",
                    latency_ms=latency_ms,
                    prompt_tokens=count_tokens(request.prompt),
                    completion_tokens=0,
                    error=str(exc),
                    source_ids=request.source_ids,
                    tool_calls=[item.model_dump() for item in request.tool_calls],
                    metadata=request.metadata,
                ),
            )
            raise HTTPException(status_code=502, detail={"request_id": request_id, "error": str(exc)}) from exc

    @app.get("/logs")
    def logs(limit: int = Query(default=50, ge=1, le=500), status: str | None = None) -> list[dict]:
        return list_chat_logs(active_settings.db_path, limit=limit, status=status)

    @app.get("/logs/{request_id}")
    def log_detail(request_id: str) -> dict:
        item = get_chat_log(active_settings.db_path, request_id)
        if item is None:
            raise HTTPException(status_code=404, detail="request_id not found")
        return item

    @app.get("/metrics")
    def metrics() -> Response:
        summary = get_summary(active_settings.db_path)
        lines = [
            "# HELP llmops_requests_total Total LLM gateway requests.",
            "# TYPE llmops_requests_total counter",
            f"llmops_requests_total {summary['total_requests']}",
            "# HELP llmops_requests_failed_total Failed LLM gateway requests.",
            "# TYPE llmops_requests_failed_total counter",
            f"llmops_requests_failed_total {summary['failed_requests']}",
            "# HELP llmops_success_rate Request success rate from 0 to 1.",
            "# TYPE llmops_success_rate gauge",
            f"llmops_success_rate {summary['success_rate']:.6f}",
            "# HELP llmops_latency_avg_ms Average request latency in milliseconds.",
            "# TYPE llmops_latency_avg_ms gauge",
            f"llmops_latency_avg_ms {summary['avg_latency_ms']:.6f}",
            "# HELP llmops_latency_p95_ms P95 request latency in milliseconds.",
            "# TYPE llmops_latency_p95_ms gauge",
            f"llmops_latency_p95_ms {summary['p95_latency_ms']:.6f}",
        ]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.post("/evalbench/run")
    def evalbench_run() -> dict:
        run = run_eval()
        json_path = save_eval_run(run, DEFAULT_OUTPUT_DIR)
        md_path = save_markdown_report(run, DEFAULT_OUTPUT_DIR)
        return {
            "run_id": run.run_id,
            "case_count": run.case_count,
            "pipelines": run.pipelines,
            "aggregate_metrics": [item.model_dump() for item in run.aggregate_metrics],
            "json_report": str(json_path),
            "markdown_report": str(md_path),
        }

    return app


app = create_app()
