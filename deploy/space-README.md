---
title: RAG Chatbot
emoji: 📚
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Ask questions about your documents, answered with citations
---

# RAG Chatbot

Upload a document and ask questions about it. Answers come only from what you
uploaded, with citations pointing at the exact chunk each claim came from, and
a visible refusal when the documents do not cover the question.

Every answer expands into a retrieval panel showing which chunks were
retrieved, how each scored, and which of them the answer actually cited.

**This demo starts empty** — upload something to try it. Uploads live in the
container and disappear when it restarts; nothing is stored permanently.

Supports PDF, DOCX, TXT, MD, HTML, XLSX and CSV.

Source and design notes: https://github.com/kinjal-bacancy/chat-bot
