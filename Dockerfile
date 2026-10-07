# devel image: nvcc required to compile flash-linear-attention and causal-conv1d
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install deps before copying source so this layer is cached on code-only changes
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml uv.lock ./
# clef extra: torch==2.11.*, transformers==5.10.2, flash-linear-attention, causal-conv1d
# MAX_JOBS caps parallel CUDA compilation — raise on machines with more cores
RUN MAX_JOBS=4 uv sync --extra clef --no-dev

COPY . .

# ── server ──────────────────────────────────────────────────────────────────
ENV MODEL_BACKEND="clef"
ENV MODEL_PATH="Cloudflare/clef"
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV DEVICE="cuda"
ENV NOUL_YES_TOKEN="yes"
ENV NOUL_NO_TOKEN="no"

# ── clef performance knobs ──────────────────────────────────────────────────
# FP8: load kurcontko/clef-flash-FP8-Dynamic (halves VRAM, enables large state cache)
ENV CLEF_FP8="1"
# torch.compile: fuses kernel launches on the backbone (1.5-2x on short prefills)
ENV CLEF_COMPILE="1"
# packed batching: eliminates padding waste; requires fla, which is installed above
ENV CLEF_PACKED="1"
# state cache budget — raise on H100 with FP8 loaded (leaves ~53 GB free)
ENV CLEF_STATE_CACHE_GB="40"
ENV CLEF_CHUNK_TOKENS="2048"
ENV CLEF_LONG_STATE_TOKENS="1024"
ENV CLEF_BATCH_TOKENS="8192"
ENV CLEF_BATCH_MAX="32"
ENV CLEF_MAX_QUEUED_TOKENS="524288"
ENV CLEF_ANSWER_CACHE_SIZE="10000"
# Never set to 1 in production: forces the ~20-launch PyTorch DeltaNet fallback
ENV CLEF_ALLOW_SLOW_KERNELS="0"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
