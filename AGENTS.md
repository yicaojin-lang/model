# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a **multi-LLM local blind-test and comparison tool** — a FastAPI app that benchmarks and compares Ollama models. All data stays in a local SQLite database (`data/modelTool.db`, auto-created on startup).

### Services

| Service | How to start | Default endpoint |
|---------|-------------|-----------------|
| **FastAPI app** | `source .venv/bin/activate && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000` | `http://127.0.0.1:8000` |
| **Ollama** | `ollama serve` (run in background) | `http://127.0.0.1:11434` |

Both services must be running for the app to function. Ollama must be started before the FastAPI app can list models or run evaluations.

### Key caveats

- **Ollama must be started manually** — systemd is not available in Cloud Agent VMs. Run `ollama serve &` or in a separate tmux session.
- **Pull at least one model** before testing: `ollama pull tinyllama` (smallest, ~637MB). The app's model listing and evaluation features require at least one pulled model.
- **FastAPI version constraint**: The codebase's `TemplateResponse` calls use the Starlette <1.0 API. Install `fastapi>=0.110.0,<0.116.0` (which pins `starlette<1.0`) to avoid `TypeError: unhashable type: 'dict'` on the web UI. The `requirements.txt` allows newer versions, so `pip install -r requirements.txt` may pull an incompatible version. After installing from requirements.txt, run: `pip install "fastapi>=0.110.0,<0.116.0"`.
- **API key**: All REST API endpoints require `x-api-key: local-modelTool-key` header (configured in `app/config.py`). The web UI does not require authentication.
- **No automated test suite** exists in this repository. Testing is done via the REST API and web UI manually.
- **SQLite database** is auto-created at `data/modelTool.db` on first startup. No migrations needed.

### Standard commands

See `README.md` for full setup instructions (in Chinese). Key commands:

- **Install deps**: `pip install -r requirements.txt && pip install "fastapi>=0.110.0,<0.116.0"`
- **Run dev server**: `uvicorn app.main:app --reload --host 127.0.0.1 --port 8000`
- **API docs**: `http://127.0.0.1:8000/docs`
