# DevOps Deployment Guide — jev-inference

Everything you need to get this running on EC2 and keep it there.

---

## What this is

A FastAPI inference server for Cloudflare Clef (and any jev-compatible decision model).
Exposes `POST /v1/systemone` — returns typed decisions (Choice, Score, Noul) with calibrated probabilities.
No text generation. Single forward pass per question. Sub-100ms target latency.

---

## EC2 Instance Requirements

| | Minimum | Recommended |
|---|---|---|
| **Instance** | g5.xlarge (A10G 24GB) | g5.2xlarge or p3.2xlarge (V100 16GB) |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| **VRAM** | 16GB (clef-flash) | 24GB+ (clef full) |
| **RAM** | 32GB | 64GB |
| **Storage** | 100GB gp3 | 200GB gp3 |
| **CUDA** | 12.1+ | 12.4 |

> Clef full (27B) needs ~48GB VRAM. Use clef-flash (9B) on a single A10G.

---

## Step 1 — EC2 Setup (one-time)

```bash
# CUDA drivers (Ubuntu 22.04)
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit

# Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Verify GPU
nvidia-smi
```

---

## Step 2 — Clone & Install

### Option A — vLLM plugin (recommended if you already run vLLM)

Installs `/v1/systemone` directly into your existing vLLM server.
Same port, same `VLLM_API_KEY`, same TLS — nothing changes for your existing clients.

```bash
git clone <your-repo-url> /opt/jev-inference
cd /opt/jev-inference

# Install the plugin into the same venv that runs vLLM
pip install ./vllm_plugin

# Launch vLLM with the plugin enabled
VLLM_PLUGINS=jev_systemone \
SYSTEMONE_MODEL=Cloudflare/clef-flash \
SYSTEMONE_DEVICE=cuda \
SYSTEMONE_MAX_CONCURRENT=4 \
vllm serve <your-llm-model> --host 0.0.0.0 --port 8000
```

Your vLLM server now exposes both:
- `POST /v1/chat/completions` — handled by vLLM (your existing LLM)
- `POST /v1/systemone` — handled by the plugin (Clef joint head)
- `GET /v1/systemone/health` — Clef health check

### Option B — Standalone server (no vLLM dependency)

```bash
git clone <your-repo-url> /opt/jev-inference
cd /opt/jev-inference

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Extra dep for Clef's joint head
pip install huggingface_hub pillow
```

---

## Step 3 — Environment Variables

Create `/opt/jev-inference/.env`:

```bash
# Which model to load (HuggingFace repo ID or local path)
MODEL_PATH=Cloudflare/clef-flash        # use clef-flash for A10G, clef for A100/H100

# MUST be "clef" to use Cloudflare's joint decision head
MODEL_BACKEND=clef

# GPU settings
DEVICE=cuda
DTYPE=bfloat16

# Server
HOST=0.0.0.0
PORT=8000
MAX_CONCURRENT=4

# Noul token override (only used for generic backend, not clef)
NOUL_YES_TOKEN=yes
NOUL_NO_TOKEN=no
```

> **If you're NOT using Clef** (e.g. Kev, OpenJev, or your own RLCD model):
> Set `MODEL_BACKEND=generic` and set `MODEL_PATH` to the HF repo ID.

---

## Step 4 — Run with Docker (recommended)

```bash
# Build
docker build -t jev-inference .

# Run (Clef)
docker run -d \
  --gpus all \
  --name jev-inference \
  -p 8000:8000 \
  --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  jev-inference
```

Check it started:
```bash
docker logs jev-inference -f
curl http://localhost:8000/health
```

---

## Step 5 — Run without Docker (alternative)

```bash
source /opt/jev-inference/.venv/bin/activate
cd /opt/jev-inference

set -a && source .env && set +a
python main.py
```

To run as a systemd service:

```ini
# /etc/systemd/system/jev-inference.service
[Unit]
Description=jev-inference server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/jev-inference
EnvironmentFile=/opt/jev-inference/.env
ExecStart=/opt/jev-inference/.venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable jev-inference
sudo systemctl start jev-inference
sudo systemctl status jev-inference
```

---

## Step 6 — Verify it works

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Cloudflare/clef-flash",
    "state": "My order has not arrived in two weeks and I am very upset.",
    "scoring": "single",
    "questions": {
      "dept": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
          "shipping": "delivery issues",
          "billing": "payment issues",
          "returns": "return requests"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this need urgent human attention?",
        "criteria": {"true": "customer is upset", "false": "routine enquiry"}
      }
    }
  }'
```

Expected response shape:
```json
{
  "model": "clef-flash",
  "answers": {
    "dept": {"type": "choice", "choice": "shipping", "confidence": 0.91, "probabilities": {...}},
    "urgent": {"type": "noul", "noul": 0.87}
  },
  "usage": {"input_tokens": 142, "output_tokens": 0}
}
```

---

## Security checklist before going public

- [ ] Put Nginx or Caddy in front — do not expose FastAPI directly
- [ ] Add an `Authorization: Bearer <token>` check (see note below)
- [ ] Open port 443 only in the EC2 security group, not 8000
- [ ] Enable HTTPS (Certbot / ACM)

Quick bearer token guard (add to `main.py` until a proper auth layer is built):
```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
import os

_bearer = HTTPBearer()
_TOKEN = os.getenv("API_TOKEN", "")

def verify_token(creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if _TOKEN and creds.credentials != _TOKEN:
        raise HTTPException(401, "invalid token")
```
Then add `verify_token` as a dependency on the `/v1/systemone` route.
Add `API_TOKEN=<your-secret>` to `.env`.

---

## Improvement Roadmap for DevOps

These are the planned improvements from `INFERENCE_ENGINE_PLAN.md`, in priority order.
Each one is a separate PR — do not bundle them.

### Phase 1 — This week
| Task | What to do | File |
|---|---|---|
| ✅ Clef joint head | Already done — set `MODEL_BACKEND=clef` | `engine.py` |
| ✅ Single-pass scoring | Already done — `scoring: "single"` in request | `engine.py` |
| Add bearer token auth | Add `API_TOKEN` env var + dependency on endpoint | `main.py` |
| Nginx reverse proxy | Put Nginx in front, terminate TLS, forward to 8000 | New `nginx.conf` |

### Phase 2 — Next sprint
| Task | What to do |
|---|---|
| Replace semaphore with async batch queue | Collect requests within a 5ms window, batch forward pass — handles burst traffic without OOMs |
| INT8 KV cache | Swap `DTYPE=bfloat16` → `DTYPE=int8` for KV cache, 2x concurrent sequences |
| Prometheus `/metrics` endpoint | Add `prometheus-fastapi-instrumentator` — latency histograms, queue depth, GPU utilization |

### Phase 3 — Following sprint
| Task | What to do |
|---|---|
| Radix-tree prefix caching | Cache state encodings keyed by state hash — massive TTFT win for repeated states (RAG, shared context) |
| GGUF support via llama.cpp | Add llama-cpp-python backend for quantized models on smaller instances |
| Chunked prefill | Split long states into chunks to prevent decode starvation |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` on startup | Model too large for instance | Switch to `clef-flash` or upgrade to p3.8xlarge |
| `ModuleNotFoundError: joint_schema_model` | Clef snapshot not downloaded | Check `MODEL_PATH` is the HF repo ID, not a local path. Run `huggingface-cli download Cloudflare/clef-flash` manually. |
| 500 on `/v1/systemone` | `MODEL_BACKEND` not set | Confirm `.env` has `MODEL_BACKEND=clef` |
| Slow first request (30–60s) | `torch.compile` warmup | Expected on first request. Pre-warm by calling `/health` after startup. |
| OOM after hours of traffic | KV cache fragmentation (generic backend) | Restart container. Phase 2 batch queue reduces frequency. |

---

## Quick reference

```bash
# Restart
docker restart jev-inference

# Logs
docker logs jev-inference --tail 100 -f

# Health
curl http://localhost:8000/health

# Stop
docker stop jev-inference
```
