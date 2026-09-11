# Deploying the demo

The UI reaches the pipeline two ways, chosen by whether `RAG_API_URL` is set:

- **set** — HTTP to a FastAPI process. What you run locally, and what the
  container does.
- **unset** — in-process, calling `app.core` directly. For hosts that run a
  single process.

The UI branches on neither; `ui/backend.py` picks, and both present the same
methods. A test asserts their signatures stay identical, because a method
added to one and not the other would only break in the deployment nobody runs
locally.

That the in-process path is possible at all is the payoff of `app/core`
importing no web framework.

## Before you start

**The free Gemini tier allows 20 generations per day, per model.** A demo
shared with several people runs out quickly, and every question after that
returns a clear error saying so. This is the binding constraint on a public
demo, not the hosting. Enabling billing removes it.

**Nothing persists.** Free hosts have no disk. Uploads and the index live only
as long as the process, and a restart empties them. That is why the demo ships
with nothing indexed.

**There is no authentication.** Anyone with the URL can upload documents and
spend your quota. Fine for a short-lived demo; take it down afterwards.

## Streamlit Community Cloud (free, no card)

Runs one process, so the in-process backend is used. Nothing to configure --
`RAG_API_URL` is simply absent there.

1. Push to GitHub.

2. At https://share.streamlit.io, **Create app** from your repository.
   - **Main file path:** `ui/streamlit_app.py`
   - Branch: `main`

3. **Advanced settings > Secrets**, in TOML:

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

   The app copies secrets into the environment at startup, because settings
   are read from environment variables and Streamlit supplies secrets through
   `st.secrets` instead.

4. Deploy. First build installs dependencies and takes a few minutes.

Apps sleep after 12 hours idle and wake on the next visit. Redeploy by
pushing to `main`.

## Elsewhere

`Dockerfile` runs both processes in one container -- uvicorn on an internal
port, Streamlit on the public one -- and works anywhere that builds a
Dockerfile and lets you set `PORT`: Render's free tier, Railway, Fly.io, a
VPS. `deploy/to-space.sh` pushes to a Hugging Face Space, though Docker Spaces
now require a paid plan.

Render's free tier spins services down after 15 minutes idle, so most visits
pay a one-minute cold start. That is the reason to prefer Streamlit Cloud for
something people will click occasionally.

## If it does not start

**`ModuleNotFoundError: app`.** The main file path is wrong. It must be
`ui/streamlit_app.py`; the app adds the repository root to `sys.path` itself.

**"No GEMINI_API_KEY configured" in the sidebar.** The secret is missing or
misnamed. Uploading and parsing still work; embedding and answering do not.

**Answers fail with a quota message.** The daily limit, which resets at
midnight Pacific.

**Uploads vanish.** Expected. No persistent disk on any free tier.
