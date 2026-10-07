# Deployment guide — jev-inference

FastAPI inference server for Cloudflare Clef and any jev-compatible decision model.
Exposes `POST /v1/systemone`. No text generation; single forward pass per decision.

---

## EC2 instance requirements

| | Clef-Flash 9B | Clef 27B (recommended) |
|---|---|---|
| **Instance** | p3.2xlarge (V100 16GB) | p4d.xlarge / p4de.xlarge (H100 80GB) |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| **VRAM** | 19 GB BF16 / 10 GB FP8 | 54 GB BF16 / **27 GB FP8** |
| **RAM** | 32 GB | 64 GB |
| **Storage** | 100 GB gp3 | 200 GB gp3 |
| **CUDA** | 12.1+ | 12.4+ |

> **27B + FP8 on H100:** weights occupy ~27 GB, leaving ~53 GB for the state cache.
> The state cache is what beats vllm-jev — FP8 is near-mandatory to make it large enough.

---

## Step 1 — EC2 setup (one-time)

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
nvidia-smi   # verify GPU visible
```

---

## Step 2 — Deploy

### Option A — vLLM plugin (recommended if vLLM is already running)

Adds `/v1/systemone` to your vLLM server via a thin proxy. No model is loaded inside
vLLM's process — the plugin forwards to the engine, which runs as a second process.

> **Two processes mean two GPU allocations.** vLLM reserves ~90% of the card by default
> (`--gpu-memory-utilization 0.9`), so it will starve the engine if they share a GPU. Pick one:
>
> - **Two GPUs (recommended).** Give the engine its own card with `CUDA_VISIBLE_DEVICES=0`
>   and vLLM the other with `CUDA_VISIBLE_DEVICES=1`. No memory negotiation needed, and the
>   values below work as written.
> - **One GPU.** Budget it explicitly. On an 80 GB H100 with Clef 27B in FP8 (~27 GB), a
>   `CLEF_STATE_CACHE_GB=20` engine leaves roughly 30 GB, so pass
>   `--gpu-memory-utilization 0.35` to vLLM and verify with `nvidia-smi` before load.
>   The 40 GB cache below will OOM in this configuration.

```bash
git clone <your-repo-url> /opt/jev-inference
cd /opt/jev-inference

# 1. Start the engine (owns the model; give it its own GPU if you have two)
uv sync --extra clef
export VLLM_API_KEY=$(openssl rand -hex 32)
CUDA_VISIBLE_DEVICES=0 \
MODEL_BACKEND=clef MODEL_PATH=Cloudflare/clef PORT=8001 \
  CLEF_FP8=1 CLEF_COMPILE=1 CLEF_PACKED=1 CLEF_STATE_CACHE_GB=40 \
  uv run python main.py &

# 2. Install the plugin into vLLM's venv
pip install ./vllm_plugin

# 3. Start vLLM with the plugin (second GPU; on a shared GPU add --gpu-memory-utilization)
CUDA_VISIBLE_DEVICES=1 \
VLLM_PLUGINS=jev_systemone \
CLEF_ENGINE_URL=http://127.0.0.1:8001 \
VLLM_API_KEY=$VLLM_API_KEY \
vllm serve <your-llm-model> --host 0.0.0.0 --port 8000 --api-key $VLLM_API_KEY
```

`VLLM_PLUGINS` loads the entry point `jev_systemone`. The caller's `Authorization`
header is forwarded unchanged. Both processes share one `VLLM_API_KEY`.

Engine health is on the engine's own port, not vLLM's: `GET http://127.0.0.1:8001/health`.

### Option B — Standalone Docker (recommended for Clef-only deploys)

```bash
git clone <your-repo-url> /opt/jev-inference
cd /opt/jev-inference

docker build -t jev-inference .

docker run -d \
  --gpus all \
  --name jev-inference \
  -p 8000:8000 \
  --env-file .env \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --restart unless-stopped \
  jev-inference
```

---

## Step 3 — Environment variables

Create `/opt/jev-inference/.env`:

```bash
# ── required ─────────────────────────────────────────────────────────────────
VLLM_API_KEY=<your-secret>           # Bearer token; required, no default
MODEL_BACKEND=clef                   # clef | kev | laya | strands
MODEL_PATH=Cloudflare/clef           # HF repo ID or absolute local path

# ── server ───────────────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
DEVICE=cuda

# ── performance (H100 + 27B defaults) ────────────────────────────────────────
CLEF_FP8=1                   # load FP8 checkpoint; halves VRAM, enables large cache
CLEF_COMPILE=1               # torch.compile backbone (1.5-2x on short prefills)
CLEF_PACKED=1                # packed batching via cu_seqlens (requires fla, installed)
CLEF_STATE_CACHE_GB=40       # GPU state cache budget; H100+FP8 has ~53 GB free
CLEF_CHUNK_TOKENS=2048       # max tokens per prefill chunk for long states
CLEF_LONG_STATE_TOKENS=1024  # states longer than this take the chunked path
CLEF_BATCH_TOKENS=8192       # token budget per short batch
CLEF_BATCH_MAX=32            # max records per short batch
CLEF_MAX_QUEUED_TOKENS=524288
CLEF_ANSWER_CACHE_SIZE=10000

# ── optional overrides ────────────────────────────────────────────────────────
# CLEF_REVISION=<git-sha>    pin model snapshot for deterministic cache keys
# CLEF_ATTN_IMPL=flash_attention_2   override attention backend
# CLEF_ALLOW_SLOW_KERNELS=0  never set 1 in production (forces PyTorch fallback)
# SYSTEMONE_MODEL=Cloudflare/clef    model ID used by smoke_test.py
```

> **Clef-Flash 9B on A10G:** set `MODEL_PATH=Cloudflare/clef-flash`, `CLEF_STATE_CACHE_GB=8`.
> FP8 is still recommended: use `kurcontko/clef-flash-FP8-Dynamic` as `MODEL_PATH`.

---

## Step 4 — Verify

```bash
# Check the engine started and fused kernels loaded
curl http://localhost:8000/health

# Expected: "stats" block with state_cache_bytes > 0 after the first repeated-state request
# Startup log should show:
#   "DeltaNet fast kernels available" (or ClefRuntime refuses to start)
#   "FP8 checkpoint active" or "FP8 dynamic quantization applied"
#   "startup prefix precomputed: N tokens"
#   "torch.compile applied to backbone"

# Send a real request
curl -X POST http://localhost:8000/v1/systemone \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Cloudflare/clef",
    "state": "My order has not arrived in two weeks and I am very upset.",
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
        "instructions": "Does this need urgent human attention?"
      }
    }
  }'
```

---

## Step 5 — Systemd service

```ini
# /etc/systemd/system/jev-inference.service
[Unit]
Description=jev-inference (Clef /v1/systemone)
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
```

---

## Production checklist

- [ ] Put Nginx or Caddy in front (terminate TLS; don't expose FastAPI directly on port 443)
- [ ] Open only port 443 in the EC2 security group — not 8000
- [ ] `VLLM_API_KEY` is set (server refuses to start without it)
- [ ] Verify on first deploy: `curl /health` returns `"status": "ok"` and the startup log shows fused kernels and FP8 active
- [ ] Point a load balancer health check at `GET /health` on the engine port, not vLLM's port

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Startup: `flash-linear-attention and causal-conv1d are not installed` | Wrong `uv sync` — base deps only | `uv sync --extra clef` (or Docker rebuild) |
| `CUDA out of memory` on startup | BF16 27B needs 54 GB | Set `CLEF_FP8=1` or switch to `clef-flash` |
| Startup log shows `running in original precision` | torchao not installed | `uv add torchao`, or load a pre-quantized FP8 checkpoint directly |
| Answers differ from reference on cache hits | DeltaNet conv state divergence | Increase tolerance in parity gate; check `CLEF_REVISION` matches the checkpoint |
| Long first request (30–120 s) | `torch.compile` warmup traces at startup | Expected. `/health` returns only after warmup completes. |
| `RuntimeError: VLLM_API_KEY must be set` | No API key in env | Add `VLLM_API_KEY=<secret>` to `.env` |
| vLLM plugin returns 502 | Engine process not running | Start `main.py` on `PORT=8001` before starting vLLM. Check `GET /v1/systemone/health` on vLLM's port — it proxies the engine's `/health` |
| `CUDA out of memory` when co-hosting with vLLM | vLLM reserves ~90% of the card by default and starves the engine | Separate GPUs via `CUDA_VISIBLE_DEVICES`, or budget one card explicitly: lower `CLEF_STATE_CACHE_GB` and pass `--gpu-memory-utilization` to vLLM |

---

## Quick reference

```bash
# Docker
docker logs jev-inference --tail 100 -f
docker restart jev-inference
curl http://localhost:8000/health

# Stats (cache hit rates, queue depth, tokens saved)
curl http://localhost:8000/health | python3 -m json.tool

# Parity test on real weights (run before first production use)
SYSTEMONE_MODEL=Cloudflare/clef SYSTEMONE_DEVICE=cuda \
  uv run --extra clef python smoke_test.py
```
