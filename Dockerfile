FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libglib2.0-0 \
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
    openenv>=0.1.13 \
    gunicorn

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────────────────────────
RUN mkdir -p episode_traces episode_reports logs drop_vault data/finetune

# ── Copy demo traces from HF Dataset if available ────────────────────────────
# (Handled at runtime via HF_TRACES_REPO env var)

# ── HF Spaces: port 7860 ─────────────────────────────────────────────────────
EXPOSE 7860

ENV PORT=7860
ENV HOST=0.0.0.0
ENV LLM_MODE=stub
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "hf_app:server"]
