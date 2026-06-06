from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import PROJECT_ROOT
from backend.db import get_chat_log, get_summary, init_db, list_chat_logs


def resolve_db_path() -> Path:
    raw = os.getenv("LLMOPS_DB_PATH", str(PROJECT_ROOT / "data" / "llmops.sqlite"))
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


db_path = resolve_db_path()
init_db(db_path)

st.set_page_config(page_title="Local-LLMOps", layout="wide")
st.title("Local-LLMOps")

summary = get_summary(db_path)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Requests", int(summary["total_requests"]))
col2.metric("Success rate", f"{summary['success_rate'] * 100:.1f}%")
col3.metric("Avg latency", f"{summary['avg_latency_ms']:.1f} ms")
col4.metric("P95 latency", f"{summary['p95_latency_ms']:.1f} ms")

tab_overview, tab_requests, tab_detail, tab_evalbench = st.tabs(["Overview", "Requests", "Trace Detail", "LLM-EvalBench"])

with tab_overview:
    rows = list_chat_logs(db_path, limit=200)
    if not rows:
        st.info("No requests yet. Call the FastAPI /chat endpoint to populate the dashboard.")
    else:
        frame = pd.DataFrame(rows)
        frame["created_at"] = pd.to_datetime(frame["created_at"])
        st.subheader("Latency")
        st.line_chart(frame.sort_values("created_at").set_index("created_at")["latency_ms"])
        st.subheader("Status")
        st.bar_chart(frame["status"].value_counts())
        st.subheader("Provider usage")
        st.bar_chart(frame["provider"].value_counts())

with tab_requests:
    limit = st.slider("Rows", min_value=10, max_value=200, value=50, step=10)
    rows = list_chat_logs(db_path, limit=limit)
    if rows:
        frame = pd.DataFrame(rows)
        visible = frame[
            [
                "created_at",
                "request_id",
                "status",
                "provider",
                "model",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
            ]
        ]
        st.dataframe(visible, use_container_width=True, hide_index=True)
    else:
        st.info("No request logs found.")

with tab_detail:
    request_id = st.text_input("Request ID")
    if request_id:
        item = get_chat_log(db_path, request_id)
        if item is None:
            st.error("Request not found.")
        else:
            st.write(
                {
                    "status": item["status"],
                    "provider": item["provider"],
                    "model": item["model"],
                    "latency_ms": item["latency_ms"],
                    "error": item["error"],
                }
            )
            st.subheader("Prompt")
            st.code(item["prompt"])
            st.subheader("Response")
            st.code(item["response"] or "(empty)")
            st.subheader("RAG Sources")
            st.json(item["source_ids"])
            st.subheader("Tool Calls")
            st.json(item["tool_calls"])
            st.subheader("Metadata")
            st.code(json.dumps(item["metadata"], indent=2, ensure_ascii=False), language="json")

with tab_evalbench:
    latest_path = PROJECT_ROOT / "artifacts" / "evalbench" / "latest.json"
    if not latest_path.exists():
        st.info("No LLM-EvalBench run found. Run `python -m evalbench.cli` first.")
    else:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        st.write(
            {
                "run_id": payload["run_id"],
                "created_at": payload["created_at"],
                "case_count": payload["case_count"],
                "pipelines": payload["pipelines"],
            }
        )
        metrics_frame = pd.DataFrame(payload["aggregate_metrics"])
        st.subheader("Pipeline comparison")
        st.dataframe(metrics_frame, use_container_width=True, hide_index=True)

        bad_cases = pd.DataFrame(payload["case_metrics"])
        bad_cases = bad_cases[
            (bad_cases["hit_at_3"] < 1.0)
            | (bad_cases["context_recall"] < 1.0)
            | (bad_cases["faithfulness"] < 0.75)
            | (bad_cases["answer_relevancy"] < 0.75)
            | bad_cases["error"].notna()
        ]
        st.subheader("Bad cases")
        if bad_cases.empty:
            st.success("No bad cases under the current thresholds.")
        else:
            st.dataframe(
                bad_cases[
                    [
                        "case_id",
                        "pipeline",
                        "hit_at_3",
                        "context_recall",
                        "faithfulness",
                        "answer_relevancy",
                        "error",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
