from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str = "Local-LLMOps"
    provider: str = "echo"
    db_path: Path = PROJECT_ROOT / "data" / "llmops.sqlite"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:0.5b"
    request_timeout: float = 60.0


def load_settings() -> Settings:
    db_path = Path(os.getenv("LLMOPS_DB_PATH", str(PROJECT_ROOT / "data" / "llmops.sqlite")))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    return Settings(
        provider=os.getenv("LLMOPS_PROVIDER", "echo").strip().lower(),
        db_path=db_path,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        request_timeout=float(os.getenv("LLMOPS_REQUEST_TIMEOUT", "60")),
    )
