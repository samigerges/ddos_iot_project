FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# LightGBM and the other native estimators need the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir -p data models outputs tmp .cache/matplotlib \
    && chown -R app:app /app

COPY requirements.txt requirements-docker.txt ./
RUN python -m pip install --no-cache-dir -r requirements-docker.txt

COPY --chown=app:app app.py run.py requirements-paper-reference.txt ./
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app docs/ ./docs/
COPY --chown=app:app tests/ ./tests/
COPY --chown=app:app .streamlit/config.toml ./.streamlit/config.toml

USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
