# Local-LLMOps

Local-LLMOps is a small, resume-ready LLM serving and observability project. It provides a FastAPI gateway for local model calls, stores every request in SQLite, exposes simple Prometheus-style metrics, includes a Streamlit dashboard for latency, error rate, request history, and trace inspection, and now ships with LLM-EvalBench for RAG / Agent evaluation.

The default provider is `echo`, so the project can run without downloading a model. Switch `LLMOPS_PROVIDER=ollama` to call a local Ollama server.

## Architecture

```text
User / DeepMD Research Copilot
        |
        v
FastAPI Gateway
        |
        |-- LLM provider: echo or Ollama
        |-- SQLite request log
        |-- /metrics endpoint
        v
Streamlit Dashboard
```

LLM-EvalBench sits beside the serving layer and compares multiple pipelines on the same eval set:

```text
Eval questions + corpus
        |
        v
no_rag / naive_rag / metadata_filter / reranker / tools
        |
        v
hit@k, context precision/recall, faithfulness, answer relevancy,
tool call success, latency, error rate, bad-case report
```

## Quick Start

```powershell
cd local-llmops
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

In another terminal:

```powershell
cd local-llmops
streamlit run dashboard/app.py
```

Send a request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType application/json -Body '{"prompt":"Explain DP-GEN model deviation in one paragraph.","source_ids":["deepmd-docs:dpgen"],"tool_calls":[{"name":"rag_search","status":"ok","latency_ms":42}]}'
```

## Ollama Mode

Start Ollama locally, then run:

```powershell
$env:LLMOPS_PROVIDER="ollama"
$env:OLLAMA_MODEL="qwen2.5:0.5b"
uvicorn backend.main:app --reload --port 8000
```

## API

- `GET /health` returns service and database status.
- `POST /chat` calls the configured provider and logs a request trace.
- `GET /logs?limit=50` returns recent requests.
- `GET /logs/{request_id}` returns one request trace.
- `GET /metrics` returns Prometheus-style counters and latency gauges.
- `POST /evalbench/run` runs the bundled LLM-EvalBench suite and writes JSON / Markdown reports.

## LLM-EvalBench

Run the offline evaluation suite:

```powershell
cd local-llmops
python -m evalbench.cli
```

This creates:

- `artifacts/evalbench/latest.json`
- `artifacts/evalbench/latest.md`

The bundled benchmark compares:

- `no_rag`
- `naive_rag`
- `rag_metadata_filter`
- `rag_reranker`
- `rag_tools`

Metrics include `hit@1`, `hit@3`, `hit@5`, `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `tool_call_success`, latency, error rate, and bad-case rows. The Streamlit dashboard has an `LLM-EvalBench` tab that reads the latest report.

## DeepMD Copilot Integration

The `/chat` request accepts optional fields that make the service useful as the deployment and monitoring layer for a research copilot:

- `source_ids`: IDs for RAG chunks or files used in the answer.
- `tool_calls`: tool execution records such as parsers, retrievers, or agents.
- `metadata`: task type, user session, model settings, or experiment tags.

See `examples/deepmd_copilot_trace.py` for a minimal integration example.

## Docker Compose

```powershell
docker compose up --build
```

The backend is exposed on `http://127.0.0.1:8000` and the dashboard on `http://127.0.0.1:8501`.

## Resume Bullet

Built Local-LLMOps, a local LLM serving, observability, and evaluation platform based on FastAPI, SQLite, and Streamlit. The system provides a unified `/chat` gateway for local model calls, records request latency, errors, RAG source IDs, and tool traces, and includes LLM-EvalBench to compare no-RAG, RAG, metadata filtering, reranking, and tool-augmented pipelines with hit@k, context precision/recall, faithfulness, answer relevancy, latency, and bad-case reports.
