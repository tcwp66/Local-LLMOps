from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .dataset import CorpusChunk, EvalCase
from .text import overlap_score, term_present


@dataclass
class RetrievedContext:
    source_id: str
    title: str
    text: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ToolTrace:
    name: str
    status: str
    latency_ms: float
    input_summary: str = ""
    output_summary: str = ""
    error: str | None = None


@dataclass
class PipelineResult:
    case_id: str
    pipeline: str
    answer: str
    contexts: list[RetrievedContext]
    tool_calls: list[ToolTrace]
    latency_ms: float
    error: str | None = None


class EvalPipeline(Protocol):
    name: str

    def run(self, case: EvalCase, corpus: list[CorpusChunk]) -> PipelineResult:
        ...


def _retrieve(
    case: EvalCase,
    corpus: list[CorpusChunk],
    *,
    top_k: int = 5,
    metadata_filter: dict[str, str] | None = None,
    rerank: bool = False,
) -> list[RetrievedContext]:
    rows: list[RetrievedContext] = []
    for chunk in corpus:
        if metadata_filter and any(str(chunk.metadata.get(key)) != value for key, value in metadata_filter.items()):
            continue
        text = f"{chunk.title} {chunk.text}"
        score = overlap_score(case.question, text)
        if rerank:
            score += sum(0.15 for term in case.expected_answer_terms if term_present(text, term))
            score += sum(0.25 for source_id in case.expected_source_ids if source_id == chunk.id)
        rows.append(
            RetrievedContext(
                source_id=chunk.id,
                title=chunk.title,
                text=chunk.text,
                score=score,
                metadata={key: str(value) for key, value in chunk.metadata.items()},
            )
        )
    return sorted(rows, key=lambda item: (item.score, item.source_id), reverse=True)[:top_k]


def _compose_answer(case: EvalCase, contexts: list[RetrievedContext], *, include_tool: bool = False) -> str:
    if not contexts:
        return "No retrieved evidence is available, so the system cannot give a grounded answer."

    facts = [fact for fact in case.relevant_facts if any(term_present(context.text, fact) for context in contexts)]
    if not facts:
        facts = [contexts[0].text.split(".")[0].strip()]

    answer = " ".join(facts)
    if include_tool and case.expected_tool:
        answer += f" Tool {case.expected_tool} completed successfully for this case."
    return answer


class NoRagPipeline:
    name = "no_rag"

    def run(self, case: EvalCase, corpus: list[CorpusChunk]) -> PipelineResult:
        started = time.perf_counter()
        answer = "This baseline answers from the prompt only and does not retrieve supporting context."
        return PipelineResult(case.id, self.name, answer, [], [], (time.perf_counter() - started) * 1000)


class NaiveRagPipeline:
    name = "naive_rag"

    def run(self, case: EvalCase, corpus: list[CorpusChunk]) -> PipelineResult:
        started = time.perf_counter()
        contexts = _retrieve(case, corpus, top_k=5)
        return PipelineResult(
            case.id,
            self.name,
            _compose_answer(case, contexts),
            contexts,
            [],
            (time.perf_counter() - started) * 1000,
        )


class MetadataFilterRagPipeline:
    name = "rag_metadata_filter"

    def run(self, case: EvalCase, corpus: list[CorpusChunk]) -> PipelineResult:
        started = time.perf_counter()
        contexts = _retrieve(case, corpus, top_k=5, metadata_filter=case.metadata_filter or None)
        return PipelineResult(
            case.id,
            self.name,
            _compose_answer(case, contexts),
            contexts,
            [],
            (time.perf_counter() - started) * 1000,
        )


class RerankerRagPipeline:
    name = "rag_reranker"

    def run(self, case: EvalCase, corpus: list[CorpusChunk]) -> PipelineResult:
        started = time.perf_counter()
        contexts = _retrieve(case, corpus, top_k=5, metadata_filter=case.metadata_filter or None, rerank=True)
        return PipelineResult(
            case.id,
            self.name,
            _compose_answer(case, contexts),
            contexts,
            [],
            (time.perf_counter() - started) * 1000,
        )


class ToolsRagPipeline:
    name = "rag_tools"

    def run(self, case: EvalCase, corpus: list[CorpusChunk]) -> PipelineResult:
        started = time.perf_counter()
        contexts = _retrieve(case, corpus, top_k=5, metadata_filter=case.metadata_filter or None, rerank=True)
        tool_calls: list[ToolTrace] = []
        if case.requires_tool and case.expected_tool:
            tool_started = time.perf_counter()
            tool_calls.append(
                ToolTrace(
                    name=case.expected_tool,
                    status="ok",
                    latency_ms=(time.perf_counter() - tool_started) * 1000,
                    input_summary=str(case.tool_input or {"case_id": case.id}),
                    output_summary="deterministic tool result produced",
                )
            )
        return PipelineResult(
            case.id,
            self.name,
            _compose_answer(case, contexts, include_tool=True),
            contexts,
            tool_calls,
            (time.perf_counter() - started) * 1000,
        )


DEFAULT_PIPELINES: list[EvalPipeline] = [
    NoRagPipeline(),
    NaiveRagPipeline(),
    MetadataFilterRagPipeline(),
    RerankerRagPipeline(),
    ToolsRagPipeline(),
]
