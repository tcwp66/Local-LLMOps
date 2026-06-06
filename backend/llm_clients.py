from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(Protocol):
    def generate(self, prompt: str, model: str | None = None) -> LLMResult:
        ...


def count_tokens(text: str) -> int:
    return max(1, len(text.split()))


class EchoClient:
    def __init__(self, default_model: str = "echo-model") -> None:
        self.default_model = default_model

    def generate(self, prompt: str, model: str | None = None) -> LLMResult:
        selected_model = model or self.default_model
        text = (
            "Echo provider response. Replace LLMOPS_PROVIDER with 'ollama' "
            f"to call a local model. Prompt summary: {prompt[:240]}"
        )
        return LLMResult(
            text=text,
            model=selected_model,
            prompt_tokens=count_tokens(prompt),
            completion_tokens=count_tokens(text),
        )


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url
        self.default_model = settings.ollama_model
        self.timeout = settings.request_timeout

    def generate(self, prompt: str, model: str | None = None) -> LLMResult:
        selected_model = model or self.default_model
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={"model": selected_model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            payload = response.json()

        text = payload.get("response", "")
        return LLMResult(
            text=text,
            model=selected_model,
            prompt_tokens=int(payload.get("prompt_eval_count") or count_tokens(prompt)),
            completion_tokens=int(payload.get("eval_count") or count_tokens(text)),
        )


def build_client(settings: Settings) -> LLMClient:
    if settings.provider == "echo":
        return EchoClient()
    if settings.provider == "ollama":
        return OllamaClient(settings)
    raise ValueError(f"Unsupported LLMOPS_PROVIDER: {settings.provider}")
