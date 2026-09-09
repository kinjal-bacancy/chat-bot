# RAG Chatbot — approach and build plan

## What this is

We've each been asked to build a RAG-based chatbot end to end: document upload
through parsing, chunking, embeddings, retrieval and grounded answer generation.

This is the approach I landed on, with the reasoning written down. **It isn't a
spec and nobody needs to follow it.** I got to a plan early, so it's here to
save you the blank-page problem and to flag the decisions you'll hit anyway.
The *stack* choices are the least interesting part — swap them freely. The
*build order* and the **Traps** section near the bottom are the parts I'd
actually suggest borrowing, because they're about avoiding wasted work rather
than about tooling.

If you disagree with something here, that's a useful conversation — several of
these calls are genuinely arguable.

## Scope

**In scope**

- Upload PDF / DOCX / TXT / MD / HTML documents
- Ask questions answered *only* from those documents
- Citations pointing back to the exact source chunk
- Visible refusal when the retrieved context can't support an answer
- A retrieval inspector: see which chunks were retrieved and their scores
- A small evaluation set with retrieval metrics

**Explicit non-goals** — cut to keep the pipeline the focus:

- No auth, no multi-user, no tenancy
- No cloud deploy; runs on localhost
- No OCR — scanned/image-only PDFs are rejected, not silently ingested empty
- No conversational memory beyond the current session
- No agentic tool use; retrieval is a single pass, not a loop

## Stack

Python + FastAPI backend, Streamlit frontend, talking over HTTP.

Python because the RAG ecosystem there is far ahead of anywhere else — parsers,
embedding models, rerankers all exist as one-line installs. Streamlit because
it turns a Python script into a working web UI with no HTML, CSS or JS, which
matters if you'd rather spend the time on the pipeline than on a frontend.

The FastAPI/Streamlit **split** is the deliberate bit. Keeping the pipeline
behind an HTTP API means every stage is `curl`-able and unit-testable without
opening a browser, and the UI stays a thin client with no RAG logic in it. A
single Streamlit script would have been faster to write and much harder to
debug once retrieval starts misbehaving.

Alternatives worth considering if you'd rather: **Streamlit only** (fastest to
a demo, ~1 file), **Next.js / all-TypeScript** (one language, nicer UI, thinner
parsing and reranking libraries), **FastAPI + React** (best-looking result,
most work).

## Architecture

```
Streamlit UI  ──HTTP──>  FastAPI  ──>  RAG pipeline  ──>  Vector store + BM25
                                            │
                                            └──>  LLM provider (grounded answer)
```

- `app/core/` — the pipeline: parsing, chunking, embedding, retrieval. Pure
  Python, no web framework and no vendor SDKs, so it's testable in isolation.
- `app/api/` — thin HTTP handlers.
- `app/providers/` — swappable embedding and LLM backends, selected via `.env`.
- `ui/` — Streamlit client. No RAG logic.

## Build order

Eleven steps, each independently runnable before the next begins.

### Phase 1 — Skeleton (~1h)
1. **Project scaffold** — layout, venv, requirements, env template, git init.
2. **Hello-world API** — FastAPI app with `/health`.

### Phase 2 — Ingestion (~5h)
3. **Document upload** — `POST /documents`, persist file + metadata row.
4. **Parsing** — PDF / DOCX / TXT / MD / HTML -> normalized text, page numbers kept.
5. **Chunking** — structure-aware overlapping chunks, inspectable via API.
6. **Embeddings + vector store** — embed chunks, persist vectors.

### Phase 3 — Retrieval & generation (~6h)
7. **Retrieval** — `POST /search` returns top-k chunks with scores.
8. **Hybrid search + reranking** — BM25 fused with vector search, then rerank.
9. **Grounded generation** — LLM answers with `[n]` citations; refuses when the
   retrieved context is insufficient.

### Phase 4 — Polish (~5h)
10. **Streamlit UI** — upload, streaming chat, and a retrieval-inspector panel
    exposing which chunks were used and why.
11. **Evaluation + docs** — golden Q&A set, retrieval hit-rate / MRR metrics,
    architecture docs, publish to GitHub.

Roughly two to three focused days. Phase 2 is where the time actually goes —
document parsing is consistently messier than it looks.

**Why this order:** steps 3→9 follow the exact path data takes through the
system, so at every point there's an intermediate output you can eyeball. The
ordering constraint that matters is retrieval (7) landing before generation (9).

## Design decisions

- **Providers are abstractions.** `EmbeddingProvider` and `LLMProvider` are
  interfaces chosen by `.env`, so local models, hosted embeddings and different
  LLMs are interchangeable and the pipeline never imports a vendor SDK.
- **Retrieval is debuggable before generation exists.** `/search` ships three
  steps before any LLM call, so a bad answer can be attributed to retrieval or
  to generation instead of guessed at.
- **Local embeddings first.** Free and unlimited, so chunking can be re-tuned
  and re-indexed as often as needed without cost. Swap to a hosted model later
  if quality demands it.
- **No framework.** Writing the pipeline directly rather than using LangChain or
  LlamaIndex. Those are the right call for production, but here they'd hide
  exactly the mechanics the exercise is about — and debugging their abstractions
  costs more than writing 200 lines of retrieval code.
- **Refusal is a feature.** "The documents don't cover this" is a correct
  answer, and worth building deliberately rather than hoping the prompt holds.

## Decision points you'll each hit

Not settled, and reasonable people differ:

- **Chunk size and strategy.** Fixed-token, recursive, or semantic. Big chunks
  retrieve less precisely; small chunks lose the context needed to answer. This
  has more effect on output quality than model choice does.
- **Embedding model.** Local `sentence-transformers` (free, decent) vs a hosted
  model (better, costs per re-index). I've deferred this behind the provider
  interface rather than committing.
- **Vector store.** Chroma, LanceDB, FAISS, or pgvector. For this scale it
  genuinely does not matter much — pick on API ergonomics, not benchmarks.
- **Hybrid weighting.** How to fuse BM25 with vector scores, and whether a
  reranker is worth the latency. Mine will need tuning against the eval set.

## Traps

Things I expect to cost people time:

- **Bad answers are usually a retrieval problem, not a prompt problem.** Build
  something that shows you the retrieved chunks *before* you start tuning
  prompts, or you'll optimise the wrong half of the system.
- **PDF text extraction is genuinely hard.** Multi-column layouts, tables and
  headers/footers all corrupt naive extraction, and the damage is invisible
  until answers get strange. Print the extracted text and read it.
- **Losing page numbers during parsing.** Easy to do, and it makes citations
  impossible to add later without redoing ingestion. Carry the metadata from
  the start.
- **Re-indexing friction.** You will change your chunking strategy several
  times. If re-indexing is slow or manual, you'll avoid doing it and settle for
  a worse strategy. Make it one command.

## Status

Step 1 complete. Everything past it is planned, not built.
