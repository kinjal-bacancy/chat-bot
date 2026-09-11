# Deploying the demo

One container runs both processes: uvicorn on an internal port, Streamlit on
the public one. The split that keeps the pipeline testable in development does
not warrant two deployments for a demo.

Target here is Hugging Face Spaces -- free, no credit card, and Docker Spaces
need no platform-specific code.

## Before you start

**The free Gemini tier allows 20 generations per day, per model.** A demo
shared with several people will run out, and every question after that returns
a clear error saying so. That is the binding constraint on a public demo, not
the hosting. Enabling billing on the Google Cloud project removes the cap.

**Nothing persists.** The free tier has no disk, so uploads and the index live
only as long as the container. A restart empties it. This is why the demo
starts with nothing indexed rather than shipping a corpus.

**There is no authentication.** Anyone with the URL can upload documents and
spend your quota. Acceptable for a short-lived demo; not for anything else.

## Steps

1. Create a Space at https://huggingface.co/new-space
   - **SDK: Docker**, blank template
   - Visibility: public, or private if you would rather invite people

2. Add the API key: Space **Settings > Variables and secrets > New secret**
   - Name `GEMINI_API_KEY`, value your key
   - A *secret*, not a variable -- variables are visible to anyone who can
     read the Space

3. Deploy the current commit:

   ```bash
   ./deploy/to-space.sh https://huggingface.co/spaces/<user>/<space>
   ```

   Pushing prompts for your Hugging Face username and an access token
   (Settings > Access Tokens, write scope). Not your account password.

4. Watch the build in the Space's **Logs** tab. First build takes a few
   minutes; later ones reuse cached layers.

Redeploy after any change by committing and running the same script again.

## If it does not start

**Build succeeds, app never loads.** Check the log for `API failed to start`.
The entrypoint waits up to 60 seconds for the API's health endpoint and exits
if it never answers, rather than leaving Streamlit serving a dead backend.

**"No GEMINI_API_KEY configured" in the sidebar.** The secret is missing or
misnamed. Uploading and parsing still work; embedding and answering do not.

**Answers fail with a quota message.** The daily limit. It resets at midnight
Pacific. `LLM_MODEL` can be set as a Space variable to try another model, but
every free-tier model shares the same daily cap per project.

**Space is asleep.** Free Spaces sleep after 48 hours idle and take about a
minute to wake on the next visit.

## Elsewhere

The container is not Hugging Face specific. Anywhere that runs a Dockerfile
and lets you set `PORT` will work -- Render, Railway, Fly.io, a VPS. Only the
Space README's front matter is platform-specific, and it lives in `deploy/`
rather than in the image.
