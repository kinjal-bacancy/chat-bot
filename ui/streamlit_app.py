"""RAG chatbot UI.

A thin client over the API. Streamlit re-runs this script top to bottom on
every interaction, so anything that must survive a re-run lives in
st.session_state.

The retrieval inspector is the point of this screen. A chat box alone shows
an answer and asks you to trust it; showing what was retrieved, how it
scored, and which of it the answer actually cited turns a wrong answer into
a diagnosable one.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from api import Api, ApiError  # noqa: E402

DEFAULT_API = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
STATUS_ICON = {
    "uploaded": "•", "parsed": "•", "chunked": "•",
    "indexed": "✓", "failed": "✕",
}

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")


def api() -> Api:
    return Api(st.session_state.get("api_url", DEFAULT_API))


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("api_url", DEFAULT_API)
    st.session_state.setdefault("uploaded", set())


# ----------------------------------------------------------------- sidebar


def render_connection() -> dict | None:
    st.sidebar.text_input("API URL", key="api_url")
    try:
        health = api().health()
    except ApiError as exc:
        st.sidebar.error(f"API unreachable\n\n{exc}")
        return None

    st.sidebar.success(f"Connected · {health['llm_model']}")
    if not health["credentials_configured"]:
        # Upload and parsing work without a key; embedding and answering do
        # not. Saying so here beats a 502 three clicks later.
        st.sidebar.warning(
            "No GEMINI_API_KEY configured. Uploading works; embedding and "
            "answering will fail."
        )
    return health


def render_upload() -> None:
    st.sidebar.subheader("Documents")
    uploaded = st.sidebar.file_uploader(
        "Add a document",
        type=["pdf", "docx", "txt", "md", "html", "htm", "xlsx", "xlsm", "csv", "tsv"],
        accept_multiple_files=True,
    )

    for file in uploaded or []:
        # file_uploader re-offers its files on every re-run, so without this
        # guard a single upload would re-ingest on every keystroke.
        key = f"{file.name}:{file.size}"
        if key in st.session_state.uploaded:
            continue
        st.session_state.uploaded.add(key)

        with st.sidebar.status(f"Ingesting {file.name}", expanded=False) as status:
            try:
                document = api().upload(file.name, file.getvalue())
                if document.get("duplicate"):
                    status.update(label=f"{file.name} — already uploaded", state="complete")
                    continue
                result = api().ingest(document["id"])
                status.update(
                    label=(
                        f"{file.name} — {result['chunk_count']} chunks, "
                        f"{result['embedded']} embedded"
                    ),
                    state="complete",
                )
            except ApiError as exc:
                status.update(label=f"{file.name} — failed", state="error")
                st.sidebar.error(str(exc))


def render_document_list() -> list[dict]:
    try:
        documents = api().documents()
    except ApiError as exc:
        st.sidebar.error(str(exc))
        return []

    if not documents:
        st.sidebar.caption("No documents yet.")
        return []

    for document in documents:
        icon = STATUS_ICON.get(document["status"], "•")
        columns = st.sidebar.columns([6, 1])
        columns[0].markdown(
            f"{icon} **{document['filename']}**  \n"
            f"<span style='color:gray;font-size:0.8em'>{document['status']}"
            f" · {document['size_bytes'] // 1024} KB</span>",
            unsafe_allow_html=True,
        )
        if columns[1].button("✕", key=f"del-{document['id']}", help="Delete"):
            api().delete(document["id"])
            st.rerun()

        if document["status"] == "failed" and document["error"]:
            st.sidebar.error(document["error"], icon="✕")
        elif document["status"] != "indexed":
            st.sidebar.warning(
                "Not searchable until ingested.", icon="!"
            )

    return documents


# -------------------------------------------------------------- inspector


def render_inspector(message: dict) -> None:
    """Show what retrieval returned and what the answer used."""
    retrieved = message.get("retrieved", [])
    if not retrieved:
        return

    cited = {source["number"] for source in message.get("sources", [])}
    label = (
        f"Retrieval · {len(retrieved)} chunks from {message.get('searched_chunks', 0)} "
        f"searched · {len(cited)} cited · mode={message.get('mode', '?')}"
    )

    with st.expander(label, expanded=False):
        for source in retrieved:
            used = source["number"] in cited
            heading = f" · {source['heading']}" if source.get("heading") else ""
            st.markdown(
                f"**[{source['number']}]** {'✓ cited' if used else 'not cited'} · "
                f"score `{source['score']:.3f}` · {source['filename']} "
                f"p{source['page_number']}, chunk {source['chunk_index']}{heading}"
            )
            st.code(source["text"], language=None, wrap_lines=True)


def render_sources(message: dict) -> None:
    if message.get("refused") or not message.get("sources"):
        return
    lines = [
        f"`[{s['number']}]` {s['filename']} · page {s['page_number']}"
        + (f" · {s['heading']}" if s.get("heading") else "")
        for s in message["sources"]
    ]
    st.caption("Sources — " + "  ".join(lines))


# ------------------------------------------------------------------- chat


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("refused"):
                st.info(message["content"], icon="○")
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message)
                render_inspector(message)


def ask(question: str, top_k: int, mode: str, document_ids: list[str] | None) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        message: dict = {"role": "assistant", "content": "", "retrieved": []}
        collected: list[str] = []

        try:
            for event, payload in api().ask_stream(
                question, top_k=top_k, mode=mode, document_ids=document_ids
            ):
                if event == "retrieved":
                    message.update(payload)
                elif event == "delta":
                    collected.append(payload["text"])
                    placeholder.markdown("".join(collected))
                elif event == "error":
                    placeholder.error(payload["detail"])
                    message["content"] = payload["detail"]
                    message["refused"] = True
                elif event == "done":
                    message.update(payload)
                    message["content"] = payload["answer"]
                    if payload["refused"]:
                        placeholder.info(payload["answer"], icon="○")
                    else:
                        placeholder.markdown(payload["answer"])
        except ApiError as exc:
            placeholder.error(str(exc))
            message["content"] = str(exc)
            message["refused"] = True

        render_sources(message)
        render_inspector(message)

    st.session_state.messages.append(message)


# ------------------------------------------------------------------- main


def main() -> None:
    init_state()

    st.title("RAG Chatbot")
    st.caption(
        "Answers come only from the documents you upload, with citations. "
        "If the documents do not cover a question, it says so."
    )

    health = render_connection()
    render_upload()
    documents = render_document_list()

    st.sidebar.subheader("Retrieval")
    mode = st.sidebar.selectbox(
        "Mode", ["dense", "hybrid", "keyword"],
        help="dense: embeddings · keyword: BM25 · hybrid: both, fused by rank",
    )
    top_k = st.sidebar.slider("Chunks per answer", 1, 12, 5)

    indexed = [d for d in documents if d["status"] == "indexed"]
    chosen = st.sidebar.multiselect(
        "Limit to documents",
        options=[d["id"] for d in indexed],
        format_func=lambda i: next(d["filename"] for d in indexed if d["id"] == i),
    )

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    render_history()

    if not indexed:
        st.info("Upload a document in the sidebar to get started.", icon="○")

    if question := st.chat_input("Ask a question about your documents"):
        if health is None:
            st.error("Not connected to the API.")
        else:
            ask(question, top_k, mode, chosen or None)


main()
