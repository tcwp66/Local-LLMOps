from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.config import PROJECT_ROOT


DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "evalbench_corpus.json"
DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "evalbench_questions.json"


class CorpusChunk(BaseModel):
    id: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalCase(BaseModel):
    id: str
    question: str
    expected_answer_terms: list[str]
    expected_source_ids: list[str]
    relevant_facts: list[str]
    metadata_filter: dict[str, str] = Field(default_factory=dict)
    requires_tool: bool = False
    expected_tool: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)


def _load_json_list(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return payload


def load_corpus(path: str | Path = DEFAULT_CORPUS_PATH) -> list[CorpusChunk]:
    return [CorpusChunk.model_validate(item) for item in _load_json_list(path)]


def load_eval_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    return [EvalCase.model_validate(item) for item in _load_json_list(path)]
