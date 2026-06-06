# Local-LLMOps

Local-LLMOps is a small, resume-ready LLM serving and observability project. It provides a FastAPI gateway for local model calls, stores every request in SQLite, exposes simple Prometheus-style metrics, and includes a Streamlit dashboard for latency, error rate, request history, and trace inspection.

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

Built Local-LLMOps, a local LLM serving and observability platform based on FastAPI, SQLite, and Streamlit. The system provides a unified `/chat` gateway for local model calls, records request latency, errors, RAG source IDs, and tool traces, and exposes dashboard and metrics views for debugging model behavior and service reliability.
