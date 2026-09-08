# Build plan

Each step is independently runnable and verifiable before the next begins.

## Phase 1 — Skeleton
1. **Project scaffold** — layout, venv, requirements, env template, git init.
2. **Hello-world API** — FastAPI app with `/health`.

## Phase 2 — Ingestion
3. **Document upload** — `POST /documents`, persist file + metadata row.
4. **Parsing** — PDF / DOCX / TXT / MD / HTML -> normalized text, page numbers kept.
5. **Chunking** — structure-aware overlapping chunks, inspectable via API.
6. **Embeddings + vector store** — embed chunks, persist vectors.

## Phase 3 — Retrieval & generation
7. **Retrieval** — `POST /search` returns top-k chunks with scores.
8. **Hybrid search + reranking** — BM25 fused with vector search, then rerank.
9. **Grounded generation** — Claude answers with `[n]` citations; refuses when
   the retrieved context is insufficient.

## Phase 4 — Polish
10. **Streamlit UI** — upload, streaming chat, and a retrieval-inspector panel
    exposing which chunks were used and why.
11. **Evaluation + docs** — golden Q&A set, retrieval hit-rate / MRR metrics,
    architecture docs, publish to GitHub.

## Design decisions

- **Providers are abstractions.** `EmbeddingProvider` and `LLMProvider` are
  interfaces chosen by `.env`, so local models, Voyage and Claude are
  interchangeable and the pipeline never imports a vendor SDK.
- **Retrieval is debuggable before generation exists.** `/search` ships in
  step 7, three steps before any LLM call, so bad answers can be attributed to
  retrieval or to generation rather than guessed at.
- **Local embeddings first.** Free and unlimited, so chunking strategy can be
  re-tuned and re-indexed as often as needed without cost.
