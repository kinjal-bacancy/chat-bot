# RAG Chatbot

An end-to-end Retrieval-Augmented Generation system: upload documents, and ask
questions that are answered **only** from those documents, with citations back
to the exact chunk the answer came from.

Built to make every stage of the RAG pipeline inspectable rather than a black box.

## Status

Under construction. See [docs/PLAN.md](docs/PLAN.md) for the build order.

## Architecture

```
Streamlit UI  ──HTTP──>  FastAPI  ──>  RAG pipeline  ──>  Vector store + BM25
                                            │
                                            └──>  LLM provider (grounded answer)
```

- `app/core/` — the pipeline itself: parsing, chunking, embedding, retrieval.
  Pure Python, no web framework, no vendor SDKs. Unit-testable in isolation.
- `app/api/` — thin HTTP handlers.
- `app/providers/` — swappable embedding and LLM backends, selected via `.env`.
- `ui/` — Streamlit client. Contains no RAG logic.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Run the UI (in a second terminal):

```bash
streamlit run ui/streamlit_app.py
```
