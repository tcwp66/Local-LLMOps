from __future__ import annotations

from collections import defaultdict
from statistics import mean

from pydantic import BaseModel

from .dataset import EvalCase
from .pipelines import PipelineResult
from .text import term_present


class CaseMetrics(BaseModel):
    case_id: str
    pipeline: str
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    tool_call_success: float | None
    latency_ms: float
    error: str | None = None


class AggregateMetrics(BaseModel):
    pipeline: str
    cases: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    tool_call_success: float | None
    avg_latency_ms: float
    error_rate: float


def score_case(case: EvalCase, result: PipelineResult) -> CaseMetrics:
    retrieved_ids = [context.source_id for context in result.contexts]
    expected = set(case.expected_source_ids)
    relevant_retrieved = [source_id for source_id in retrieved_ids if source_id in expected]

    def hit_at(k: int) -> float:
        return 1.0 if expected.intersection(retrieved_ids[:k]) else 0.0

    context_precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
    context_recall = len(set(relevant_retrieved)) / len(expected) if expected else 1.0
    supporting_context = " ".join(context.text for context in result.contexts)
    supported_facts = [
        fact for fact in case.relevant_facts if term_present(result.answer, fact) and term_present(supporting_context, fact)
    ]
    answer_terms = [term for term in case.expected_answer_terms if term_present(result.answer, term)]
    tool_call_success = None
    if case.requires_tool:
        tool_call_success = 1.0 if any(call.name == case.expected_tool and call.status == "ok" for call in result.tool_calls) else 0.0

    return CaseMetrics(
        case_id=case.id,
        pipeline=result.pipeline,
        hit_at_1=hit_at(1),
        hit_at_3=hit_at(3),
        hit_at_5=hit_at(5),
        context_precision=context_precision,
        context_recall=context_recall,
        faithfulness=len(supported_facts) / len(case.relevant_facts) if case.relevant_facts else 1.0,
        answer_relevancy=len(answer_terms) / len(case.expected_answer_terms) if case.expected_answer_terms else 1.0,
        tool_call_success=tool_call_success,
        latency_ms=result.latency_ms,
        error=result.error,
    )


def aggregate(case_metrics: list[CaseMetrics]) -> list[AggregateMetrics]:
    by_pipeline: dict[str, list[CaseMetrics]] = defaultdict(list)
    for metric in case_metrics:
        by_pipeline[metric.pipeline].append(metric)

    summaries: list[AggregateMetrics] = []
    for pipeline, rows in sorted(by_pipeline.items()):
        tool_rows = [row.tool_call_success for row in rows if row.tool_call_success is not None]
        summaries.append(
            AggregateMetrics(
                pipeline=pipeline,
                cases=len(rows),
                hit_at_1=mean(row.hit_at_1 for row in rows),
                hit_at_3=mean(row.hit_at_3 for row in rows),
                hit_at_5=mean(row.hit_at_5 for row in rows),
                context_precision=mean(row.context_precision for row in rows),
                context_recall=mean(row.context_recall for row in rows),
                faithfulness=mean(row.faithfulness for row in rows),
                answer_relevancy=mean(row.answer_relevancy for row in rows),
                tool_call_success=mean(tool_rows) if tool_rows else None,
                avg_latency_ms=mean(row.latency_ms for row in rows),
                error_rate=mean(1.0 if row.error else 0.0 for row in rows),
            )
        )
    return summaries
