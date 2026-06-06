from __future__ import annotations

import json

import httpx


payload = {
    "prompt": "Diagnose why a DeepMD training run has a flat validation loss after 20k steps.",
    "source_ids": [
        "deepmd-docs:lcurve",
        "project-notes:dpgen-iteration-checklist",
    ],
    "tool_calls": [
        {
            "name": "rag_search",
            "status": "ok",
            "latency_ms": 38.4,
            "input_summary": "query DeepMD flat validation loss",
            "output_summary": "returned 4 chunks",
        },
        {
            "name": "parse_lcurve",
            "status": "ok",
            "latency_ms": 12.7,
            "input_summary": "lcurve.out",
            "output_summary": "train and validation loss plateau detected",
        },
    ],
    "metadata": {
        "task_type": "deepmd_diagnostics",
        "copilot_module": "training_log_analysis",
        "experiment": "resume-demo",
    },
}


def main() -> None:
    response = httpx.post("http://127.0.0.1:8000/chat", json=payload, timeout=60)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
