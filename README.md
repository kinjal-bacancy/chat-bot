# RAG Chatbot

Upload documents, ask questions, get answers grounded **only** in those
documents -- with citations pointing at the exact chunk each claim came from,
and a visible refusal when the documents do not cover the question.

Built so that every stage of the pipeline is inspectable rather than a black
box. When an answer is wrong, you can tell whether retrieval or the model was
at fault, without re-running anything.

```
Streamlit UI ──HTTP──► FastAPI ──► pipeline ──► SQLite (documents, pages,
     │                    │                      chunks, vectors, FTS index)
     │                    │
     └── retrieval        └──► Gemini (embeddings, generation)
         inspector
```

## What it does

- **Ingests** PDF, DOCX, TXT, MD, HTML, XLSX and CSV
- **Retrieves** with dense vectors, BM25, or both fused by reciprocal rank
- **Answers** with `[n]` citations, and refuses when the sources cannot support one
- **Shows its work**: every answer expands to reveal which chunks were
  retrieved, how each scored, and which the answer actually cited
- **Measures itself**: a golden set and `python -m eval.run` for retrieval metrics

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps plus the test runner
cp .env.example .env
```

Put a Gemini API key in `.env` as `GEMINI_API_KEY` -- free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Run the API:

```bash
uvicorn app.main:app --reload
```

Run the UI, in a second terminal:

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501, drop a document in the sidebar, and ask.

## Using the API directly

Upload, then ingest -- parse, chunk and embed in one call:

```bash
ID=$(curl -s -X POST localhost:8000/documents -F "file=@yourfile.xlsx" | jq -r .id)
curl -s -X POST localhost:8000/documents/$ID/ingest
```

Ask:

```bash
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "what is the status of the ckeditor gem"}'
```

Interactive docs at http://localhost:8000/docs.

| Endpoint | Purpose |
|---|---|
| `POST /documents` | Upload a file |
| `POST /documents/{id}/ingest` | Parse, chunk and embed |
| `GET /documents/{id}/pages` | Read the extracted text |
| `GET /documents/{id}/chunks` | Read the chunks as they will be embedded |
| `POST /search` | Retrieve chunks with scores, no model involved |
| `POST /ask` | Grounded answer with citations |
| `POST /ask/stream` | The same, as server-sent events |

The stage endpoints (`/parse`, `/chunk`, `/embed`) exist for debugging where a
bad answer came from. `/ingest` is the normal path.

## Evaluation

```bash
python -m eval.run              # retrieval metrics for every mode
python -m eval.run --answers    # also generate answers and score refusals
```

Measured against `eval/dataset.json` on a 115-row spreadsheet (31 chunks),
16 answerable questions and 4 that should be refused:

| mode | hit@1 | hit@5 | MRR |
|---|---|---|---|
| dense | 0.938 | 1.000 | 0.969 |
| hybrid | 0.938 | 1.000 | 0.969 |
| keyword | 0.812 | 0.938 | 0.865 |

Dense and hybrid tie exactly on this corpus; hybrid is the default because it
holds up better as a corpus grows and rare exact terms start to matter. Write
your own `eval/dataset.json` and re-measure -- the numbers above describe one
document, not your data.

Answer-quality figures are not filled in: the free tier allows 20 generations
per day per model, which one `--answers` run over 20 questions exhausts. The
runner paces itself and reports partial results when the quota runs out. With
billing enabled, or spread across two days, it reports how often the system
answered when it should and refused when it should.

## Design notes

Four decisions that shaped the rest:

**Never ingest a rendering of structured data.** A spreadsheet exported to PDF
is paginated by column, so one row's cells land on different pages with
nothing connecting them. Found on real data here: a gem, its version and its
status ended up three page-groups apart, unrecoverably. Spreadsheets are read
directly, one row becoming one labelled record.

**Never split a record.** Chunks are built by packing whole blocks, never by
slicing at a character offset. Half a row -- a version with no gem name -- is
worse than none, because it still retrieves and then misleads.

**Refusal must be structural.** Scores are compressed: a question about
something entirely absent still retrieves chunks around 0.6, so no threshold
can decide when to refuse. Instead every claim must cite, and an answer that
cites nothing *is* the refusal.

**Embeddings are cached by content hash.** Re-chunking reuses vectors for text
that did not change, so tuning chunk size -- the parameter that matters most --
costs almost nothing.

Longer version in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the build
order and what went wrong along the way in [docs/PLAN.md](docs/PLAN.md).

## Layout

```
app/core/       the pipeline: parsing, chunking, embedding, retrieval, answering
app/api/        HTTP handlers
app/providers/  Gemini clients behind swappable interfaces
ui/             Streamlit app and its API client
eval/           golden set and metrics
tests/          156 tests, no network calls
```

`app/core` imports no web framework and no vendor SDK, which is what lets the
whole pipeline be tested against fake providers.

## Free-tier limits worth knowing

Measured on this project, not read off a page:

| Limit | Effect |
|---|---|
| 20 generations per day, per model | One evaluation run exhausts it |
| 5 generations per minute (`gemini-3.5-flash`) | Pace batch work, or wait out 429s |
| `gemini-3.8-flash` latency | 15-35s per answer; `gemini-3.5-flash` answered in 1.4s |

A per-day quota is indistinguishable from a per-minute one in the error, and
even carries a `retryDelay` -- but no amount of waiting clears it, so the
provider fails fast on it and retries only the per-minute kind.

## Tests

```bash
python -m pytest tests/ -q
```

162 tests, none of which touch the network -- providers are substituted with
fakes, so the suite runs without an API key and cannot fail on someone else's
rate limit.

## Deploying

The UI reaches the pipeline over HTTP when `RAG_API_URL` is set, and calls
`app.core` in-process when it is not -- so it runs unchanged on a host that
allows only one process. `Dockerfile` covers the two-process case.

See [docs/DEPLOY.md](docs/DEPLOY.md), which leads with the constraints rather
than the steps: the free Gemini tier allows 20 generations per day, nothing
persists between restarts, and there is no authentication.
