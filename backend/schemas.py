from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    status: Literal["ok", "error"] = "ok"
    latency_ms: float | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    error: str | None = None


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20000)
    model: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    request_id: str
    response: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    source_ids: list[str]
    tool_calls: list[ToolCall]


class HealthResponse(BaseModel):
    app: str
    provider: str
    db_path: str
    status: Literal["ok"]
