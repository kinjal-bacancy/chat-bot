# Architecture

## Shape

```
Streamlit UI ──HTTP──► FastAPI ──► pipeline ──► SQLite (documents, pages,
     │                    │                      chunks, vectors, FTS index)
     │                    │
     └── retrieval        └──► Gemini (embeddings, generation)
         inspector
```

Two processes. The UI is a thin client holding no retrieval logic, so every
stage of the pipeline is reachable with `curl` and testable without a browser.

## Layers

| Package | Holds | Knows about |
|---|---|---|
| `app/core/` | The pipeline: parsing, chunking, embedding, retrieval, answering | Nothing web, no vendor SDK |
| `app/api/` | HTTP handlers, request/response models | `app/core`, FastAPI |
| `app/providers/` | Gemini embedding and LLM clients behind interfaces | The vendor SDK |
| `ui/` | Streamlit app and an HTTP client | The API's shape only |

`app/core` importing no vendor SDK is what lets the whole pipeline be tested
with fake providers, and what would make swapping Gemini a change confined to
one package.

## Ingestion

```
upload ──► parse ──► chunk ──► embed ──► indexed
```

`POST /documents/{id}/ingest` runs all four. The individual stage endpoints
exist because each fails for different reasons and is worth inspecting alone,
but they are a debugging affordance, not the workflow.

Each stage replaces its own output, so re-running is safe and is how a fix is
applied -- a parser change means calling `/parse` again, not re-uploading.

**Parsing** produces pages. PDFs keep real page numbers, including for pages
that extract empty, because dropping one would renumber the rest and break
citations. Running headers are stripped. Spreadsheets are read as records --
one row becomes one labelled block -- rather than as a rendering of a table.

**Chunking** splits on blank lines and packs whole blocks to a character
target. A chunk boundary never falls inside a block, so a spreadsheet row
survives intact. Overlap carries whole trailing blocks for the same reason.
Each chunk is prefixed with its enclosing heading, which becomes part of what
is embedded.

**Embedding** uses `RETRIEVAL_DOCUMENT`, and queries use `RETRIEVAL_QUERY`.
Vectors are normalised on arrival: `gemini-embedding-001` only returns unit
vectors at its native 3072 dimensions, and cosine over unnormalised vectors
ranks partly by magnitude.

## Storage

Everything is in one SQLite file.

| Table | Holds |
|---|---|
| `documents` | Metadata and pipeline status |
| `pages` | Extracted text, one row per page |
| `chunks` | Chunk text, offsets, heading, content hash |
| `chunk_embeddings` | Vectors, keyed by `(sha256, model, dimensions)` |
| `chunks_fts` | FTS5 index for BM25 |

Two consequences worth knowing.

The embedding cache **is** the vector store: chunks join to
`chunk_embeddings` on content hash. Re-chunking produces new chunk rows whose
text is mostly byte-identical, so their vectors are reused and only genuinely
new text costs an API call. That is what makes tuning chunk size affordable.

FTS5 has no foreign keys, so `chunks_fts` is maintained by hand inside the
same transaction as the chunks themselves.

## Retrieval

Dense search compares the query vector against every stored vector with numpy.
Brute force: a few thousand chunks compare in well under a millisecond, and
there is no second service. The ceiling is real -- around a hundred thousand
chunks this wants a proper index.

Keyword search uses SQLite's FTS5 with BM25. Stopwords are dropped from the
query; left in, BM25 ranks on "what" and "is" and buries the rare term that
distinguishes the answer.

Hybrid fuses the two by **reciprocal rank**, not by score -- a cosine
similarity and a BM25 score are not comparable quantities. Fusion is weighted
toward dense, because keyword search returns something for every query,
including ones with no rare terms in them.

## Answering

Retrieved chunks are numbered and given to the model, which must cite `[n]`
for every claim. Citation numbers outside the range offered are stripped.

**An answer that cites nothing is a refusal.** This is structural rather than
pattern-matched, and it has to be: scores are compressed, so a question about
something entirely absent still retrieves chunks around 0.6. There is no
threshold to refuse at. Only the model reading the sources can tell.

When retrieval returns nothing, the model is not called at all -- handing it
an empty source list invites an answer from its own knowledge.

Responses carry everything retrieved alongside what was cited, which is what
makes a poor answer attributable: a passage missing from `retrieved` means
retrieval failed; one present but uncited means the model did.
