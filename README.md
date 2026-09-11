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
cp .env.example .env
```

Then put a Gemini API key in `.env` as `GEMINI_API_KEY`. Get one free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey). Upload and
retrieval work without it; embeddings and answer generation do not.

Run the API:

```bash
uvicorn app.main:app --reload
```

Run the UI (in a second terminal):

```bash
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501. Upload a document in the sidebar -- it is
parsed, chunked and embedded automatically -- and ask a question. Every
answer expands into a retrieval panel showing which chunks were retrieved,
how each scored, and which of them the answer actually cited.

## Using the API

Upload a document, then ingest it (parse, chunk and embed in one call):

```bash
ID=$(curl -s -X POST localhost:8000/documents -F "file=@yourfile.xlsx" | jq -r .id)
curl -s -X POST localhost:8000/documents/$ID/ingest
```

Then search:

```bash
curl -s -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "your question", "top_k": 5}'
```

The individual stages -- `/parse`, `/chunk`, `/embed` -- are also exposed
separately, along with `/pages` and `/chunks` to read their output. They are
for inspecting where a bad answer came from; `/ingest` is the normal path.
