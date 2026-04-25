FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    python-dotenv \
    pydantic>=2.5.0 \
    pydantic-settings>=2.2.0 \
    networkx>=3.2 \
    openai>=1.30.0 \
    rich>=13.7.0 \
    colorlog>=6.8.0 \
    plotly>=5.20.0 \
    dash>=2.16.0 \
    dash-bootstrap-components>=1.5.0 \
    numpy>=1.26.0 \
    pandas>=2.0.0 \
    matplotlib>=3.8.0 \
    huggingface_hub \
    flask>=3.0.0 \
    flask-cors \
    openenv>=0.1.13 \
    gunicorn

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Build React War Room ──────────────────────────────────────────────────────
RUN cd dashboard-react && npm ci --prefer-offline && npm run build && cd ..

# ── Create necessary directories ──────────────────────────────────────────────
RUN mkdir -p episode_traces episode_reports logs drop_vault data/finetune

# ── HF Spaces: port 7860 ─────────────────────────────────────────────────────
EXPOSE 7860

ENV PORT=7860
ENV HOST=0.0.0.0
ENV LLM_MODE=stub
ENV PYTHONUNBUFFERED=1
# No HF_TOKEN or API_BASE_URL here — set them as HF Space secrets

# ── Entrypoint: Flask API + React static ─────────────────────────────────────
# api_server.py serves React build from dist/ AND all /api/* endpoints on :7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", \
     "--chdir", "dashboard-react", "api_server:app"]
