FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps before copying source so this layer is cached on code-only changes
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml .
RUN uv sync --no-dev

COPY . .

ENV MODEL_PATH=""
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV DEVICE="cuda"
ENV DTYPE="bfloat16"
ENV MAX_CONCURRENT="4"
ENV NOUL_YES_TOKEN="yes"
ENV NOUL_NO_TOKEN="no"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
