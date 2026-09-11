# Runs the API and the UI in one container.
#
# The two-process split is right for development -- it keeps the pipeline
# curl-able and testable without a browser -- but a demo does not warrant two
# deployments. Streamlit talks to uvicorn over localhost, so nothing about the
# architecture changes; only the supervision does.

FROM python:3.11-slim

# Hugging Face Spaces runs containers as UID 1000. Writing to a directory the
# image does not own is the usual reason a Space starts and then dies.
RUN useradd -m -u 1000 app

WORKDIR /app

COPY --chown=app:app requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app app/ ./app/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app eval/ ./eval/
COPY --chown=app:app docker/start.sh ./start.sh
RUN chmod +x start.sh

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Writable by UID 1000 and wiped on restart. The free tier has no
    # persistent disk, so uploads last for the life of the container.
    DATA_DIR=/home/app/data \
    API_PORT=8000 \
    # Spaces expects the public listener here.
    PORT=7860 \
    # Smaller than the local default: this endpoint is open to anyone.
    MAX_UPLOAD_MB=10

RUN mkdir -p /home/app/data/uploads

EXPOSE 7860

CMD ["./start.sh"]
