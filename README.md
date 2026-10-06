<div align="center">

# jev-inference

**A fast `/v1/systemone` server for System One decision models, built for Cloudflare Clef**

</div>

Send a state (text or JSON) and a set of typed questions. Get back a probability for every allowed answer. Nothing is generated, so each request is one pass through the model plus a small scoring step.

> **Status.** Every engine path matches Cloudflare's reference implementation on a small random model (CPU). It has not yet run on the real Clef weights, and it has not been benchmarked on an H100.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Serving through vLLM's port](#serving-through-vllms-port)
- [Project layout](#project-layout)
- [How it works](#how-it-works)
- [Testing](#testing)
- [Not in it yet](#not-in-it-yet)
- [Further reading](#further-reading)

## Features

| | |
|---|---|
| **Short states** | Batched through Cloudflare's own forward pass. Batches are sized by exact token counts and start as soon as the GPU is free. |
| **Long states** | Computed in chunks and saved, so the same state with new questions only computes the questions. |
| **Repeat requests** | Exact repeats come from an answer cache. Identical requests that arrive together run once. |
| **Scheduling** | Short batches and long chunks take turns. When too much work is queued, new requests get HTTP 429. |
| **Safety** | API-key auth with `VLLM_API_KEY`. Refuses to start on the slow kernel fallback. |
| **Observability** | Counters at `/health`, Prometheus metrics at `/metrics`. |

## Quick start

Clef-flash needs about 19 GB of GPU memory for its weights. Leave room for activations and for saved states (16 GB by default).

```bash
uv sync --extra clef                # Linux + CUDA only: builds the DeltaNet kernels
python smoke_test.py                # real weights vs Cloudflare's reference: run this first

export VLLM_API_KEY=$(openssl rand -hex 32)
MODEL_BACKEND=clef MODEL_PATH=Cloudflare/clef-flash \
CLEF_REVISION=17f0b0ad64efb65d273590632833508766b2aae6 PORT=8001 python main.py
```

Call it:

```bash
curl -s localhost:8001/v1/systemone \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{
        "model": "clef-flash",
        "state": "My order #1234 has not arrived in two weeks and I am very upset.",
        "questions": {
          "dept":   {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"shipping": "delivery issues", "billing": "payment issues", "returns": "return requests"}},
          "urgent": {"type": "noul", "instructions": "Does this need urgent human attention?"},
          "mood":   {"type": "score", "instructions": "How frustrated is the customer?",
                     "criteria": ["Calm", "Annoyed", "Furious"]}
        }
      }'
```

The response looks like this (the numbers are made up):

```json
{
  "model": "clef-flash",
  "answers": {
    "dept":   {"type": "choice", "choice": "shipping", "confidence": 0.91,
               "probabilities": {"shipping": 0.91, "billing": 0.05, "returns": 0.04}},
    "urgent": {"type": "noul", "noul": 0.87},
    "mood":   {"type": "score", "score": 1.62, "confidence": 0.68,
               "legend": {"0": "Calm", "1": "Annoyed", "2": "Furious"},
               "probabilities": {"0": 0.06, "1": 0.26, "2": 0.68}}
  },
  "usage": {"input_tokens": 231, "output_tokens": 0}
}
```

The three question types:
- **choice:** pick one option.
- **score:** the probability-weighted average of ordered levels.
- **noul:** the probability that the answer is "yes".

## Configuration

All settings are environment variables.

| Variable | Default | What it does |
|---|---|---|
| `VLLM_API_KEY` | *required* | Bearer token clients must send. The same key vLLM uses. |
| `MODEL_BACKEND` | `clef` | `clef` for Cloudflare Clef. `kev` for Kev. `laya`/`strands` for vendor-packaged models. `generic` is prompt-scoring on a plain causal LM — an approximation with no trained head. |
| `MODEL_PATH` | | Hugging Face id or local folder, e.g. `Cloudflare/clef-flash` |
| `CLEF_REVISION` | latest | Model repo commit to pin. Clef-flash today: `17f0b0ad64efb65d273590632833508766b2aae6` |
| `DEVICE` | `cuda` | Where the model runs |
| `HOST`, `PORT` | `0.0.0.0`, `8000` | Where the server listens |
| `CLEF_LONG_STATE_TOKENS` | `1024` | States this long or longer take the long path and are saved |
| `CLEF_CHUNK_TOKENS` | `2048` | Tokens per long-path step |
| `CLEF_STATE_CACHE_GB` | `16` | GPU memory for saved states |
| `CLEF_BATCH_TOKENS` | `8192` | Padded tokens per short batch |
| `CLEF_BATCH_MAX` | `32` | Records per short batch |
| `CLEF_MAX_QUEUED_TOKENS` | `524288` | Queued work allowed before new requests get HTTP 429 |
| `CLEF_ANSWER_CACHE_SIZE` | `10000` | Answers remembered for exact repeats |
| `CLEF_ATTN_IMPL` | transformers default | Override the attention kernel, e.g. `flash_attention_2` |
| `CLEF_ALLOW_SLOW_KERNELS` | unset | Set to `1` to run without the fast DeltaNet kernels |

## Serving through vLLM's port

If your LLM already runs on vLLM, the plugin adds `/v1/systemone` to vLLM's port. It loads no model; it forwards each request to the engine with the caller's `Authorization` header. vLLM checks the key first, so one `VLLM_API_KEY` covers both servers.

```bash
pip install ./vllm_plugin
VLLM_PLUGINS=jev_systemone CLEF_ENGINE_URL=http://127.0.0.1:8001 vllm serve <your-llm> --port 8000
```

| Variable | Default | What it does |
|---|---|---|
| `CLEF_ENGINE_URL` | `http://127.0.0.1:8001` | Where the plugin forwards requests |
| `CLEF_PROXY_TIMEOUT_S` | `30` | How long the plugin waits for the engine |

## Project layout

```
jev-inference/
├── main.py                   HTTP server: auth, /v1/systemone, /health, /metrics
├── schema.py                 request and response shapes
├── config.py                 environment settings
├── clef_engine/
│   ├── runtime.py            GPU work: batches, chunked prefill, saved states
│   └── scheduler.py          queues, caches, merging, admission control
├── vllm_plugin/              optional: serve /v1/systemone on vLLM's port
├── smoke_test.py             checks every path against Cloudflare's reference
├── engine.py                 older generic engine for jev-style decoder models
├── requirements.txt
├── INFERENCE_ENGINE_PLAN.md  research: what a top-tier inference engine needs
├── SYSTEMONE_PERF_PLAN.md    plan and decision rules vs vLLM-based Clef
├── DEVOPS.md                 older deployment notes, partly out of date
└── Dockerfile                out of date
```

## How it works

### Six ideas to know first

**Tokens.** Models read integers, not text. A tokenizer splits text into pieces and maps each piece to an id. One token is roughly three quarters of an English word. Every cost in this code is counted in tokens.

**Forward pass (prefill).** One run of the model over all input tokens. A chatbot then generates text one token at a time, which is called decode. Clef never decodes: one forward pass plus a small scoring step and it is done. That is why it is fast, and why it doesn't need the decode machinery vLLM is built around.

**Hidden states.** As tokens pass through Clef's 32 layers, each token becomes a list of 4,096 numbers that describes what it means in context. The last layer's lists, the final hidden states, are what the scoring step reads.

**Clef has two parts.**
- The **backbone**, a Qwen3.5 model with about 9 billion parameters, does the heavy reading.
- The **decision head**, a small network from Cloudflare, gives each allowed option a score. It reads the hidden states of the question and option tokens, and every other token as background "memory". Softmax turns the scores into probabilities that add up to 1.

**Causal models and the cache.** Each token can only look at tokens before it, never after. So the hidden states for your state don't depend on the questions that follow. While reading, the model keeps a cache:
- its 8 full-attention layers keep each token's keys and values (K/V), which grow with length;
- its 24 DeltaNet layers keep a fixed-size running summary.

With that cache you can stop after the state and continue with the questions later, and get the same answer as one full pass. This is the engine's core trick.

**Batching and padding.** Every forward pass reads all of the model's weights from GPU memory, about 14 GB or roughly 4 ms on an H100, however few tokens it processes. A short request doesn't give the GPU enough math to fill that time, so running several together costs barely more than one. Shorter requests get filler tokens at the end (padding), and a mask tells the model to ignore them. The filler sits at the end and the model only looks backward, so real tokens never see it.

### What Clef reads

```
[ system prompt + "STATE:" ][ your state ............ ][ questions + allowed options ][ closing tokens ]
     36 tokens, fixed             any length               depends on the questions         fixed
|<------- the same for every question about this state ->|<-------- differs per request -------->|
```

The state comes before the questions. Together with causality, that is what makes a state reusable.

### A request's journey

```
 client ── POST /v1/systemone
   │
   ▼
 main.py              check the API key, validate the questions
   │
   ▼
 ClefEngine.submit    answer cache hit?           → return the stored answer
   │                  identical request running?  → wait for that one
   ▼
 prepare              tokens, where the state ends, short or long
   │                  too much work queued?       → HTTP 429
   ▼
 scheduler loop       takes turns between the two queues:
   │                  • short batch → run_short: one padded forward pass + decision head
   │                  • long step   → long_step: one 2,048-token chunk of the state;
   │                                  the last chunk saves the state, then runs the questions
   ▼
 answers              Cloudflare's format → HTTP response
```

### The code, file by file

#### `main.py`: the front door

- **`lifespan`** runs once at startup:
  - refuses to start without `VLLM_API_KEY`;
  - loads Clef and runs a warmup;
  - starts the engine.

  Loading runs in a separate thread because it is slow, blocking work.
- **`require_key`** checks the `Authorization: Bearer <key>` header with a constant-time comparison, so the key can't be guessed from response timing.
- **`systemone`**:
  - checks the questions (a score needs 2–10 levels, a choice 1–255 options);
  - hands the request to the engine;
  - turns errors into HTTP codes: 429 means busy, try again; 400 means a bad request, such as a schema that is too long.
- **`/health`** shows counters: cache hits, queue sizes, memory used by saved states.
- **`schema.py`** describes the request and response shapes. FastAPI rejects malformed JSON automatically.

#### `clef_engine/scheduler.py`: traffic control

The server uses **asyncio**: one loop handles all HTTP requests, and while one request waits, the loop serves others. Slow work must never run on that loop, so the engine has two helper threads:
- `clef-encode` does tokenizing, which is CPU work.
- `clef-gpu` runs every model call, one at a time, so GPU work never overlaps or competes for memory.

**`submit`** checks three things in order:
1. **Answer cache.** If the same state and questions came in recently, return the stored answer. This is safe because Clef has no randomness. The last 10,000 answers are kept.
2. **Merging.** If the identical request is already running, wait for that one instead of starting another.
3. Otherwise create a **future**, a placeholder for a result that arrives later, and queue the work.

`asyncio.shield` keeps shared work running if one client disconnects. The result still lands in the cache, so a retry is instant.

**`_enqueue`** tokenizes, then applies **admission control**. If accepting the request would push queued work past the limit, it answers 429 right away. Failing fast is better than making everyone wait.

**`_run`** loops forever. When both queues have work it alternates between them, so a short request waits behind at most one chunk of long work.

**`_step_short`**:
- takes the oldest short jobs while (longest job × number of jobs) stays under the batch token budget, because padding makes every row as long as the longest;
- if the batch fails, reruns each job alone, so one bad request can't fail its neighbours.

**`_step_long`**:
- continues a half-finished long job first, so at most one partial cache sits in GPU memory;
- otherwise picks the job with the least work left, minus a bonus for time spent waiting, so a big job can't be skipped forever.

#### `clef_engine/runtime.py`: the GPU work

**Loading (`load`).**
1. `_check_kernels` runs first. DeltaNet layers need two GPU libraries (flash-linear-attention and causal-conv1d) for their fast version. Without them, transformers quietly falls back to a much slower one. On a GPU the server refuses to start rather than run slowly. The check happens before the 18 GB download, so it fails fast.
2. `load_reference` imports Cloudflare's own `joint_schema_model.py` from the model download. Tokenizing, batching, the model and answer formatting all reuse Cloudflare's code, because anything rewritten can drift from how Clef was trained.

**`_prefix_len`** works out how many tokens come before the state. It encodes two requests that differ only in the state (`"a"` vs `"b"`); where their tokens first differ is where the state starts.

**`prepare`** runs on the encode thread and builds a `Job`:
- `state_end`: where the state stops. It encodes the same questions with an empty state; the length difference is the state's size.
- `state_key`: a SHA-256 fingerprint of every token up to `state_end`. Identical tokens give the same fingerprint, so a match always means truly the same state.
- `long`: true when the state reaches `CLEF_LONG_STATE_TOKENS`. Short states are cheap to recompute and gain from batching. Long states are expensive and gain from saving.
- `cost`: tokens left to compute, used for scheduling and admission control.

**Short path (`run_short`).** Cloudflare's `collate_records` pads the jobs into one batch, and the model runs exactly as in Cloudflare's reference. `@torch.inference_mode()` tells PyTorch nothing will be trained, so it skips training bookkeeping and uses less memory.

**Long path (`long_step`).** The scheduler calls it repeatedly, one chunk per call:
1. On the first call it looks up `state_key`. On a hit, it copies the saved cache and skips to step 3.
2. It computes the next chunk of the state, continuing from the cache. It returns `None` ("call me again") until the state is finished, then saves the cache and the hidden states.
3. It runs the questions on top of the cache, joins them with the state's hidden states, and runs Cloudflare's decision head over the whole sequence.

Chunks keep each step around 45 ms on an H100, so short requests get a turn in between, and memory per step stays bounded.

**`_prefill`** is one model call that says "continue from this cache and give me the updated cache back". transformers works out token positions from what the cache already holds.

**`StateCache`** stores saved states within a memory budget and drops the least recently used when full. A 4,000-token state takes roughly 210 MB. Two details matter:
- **The deep copy.** DeltaNet layers overwrite their cache in place, so resuming straight from the saved copy would destroy it. Every resume starts from a copy.
- **Hidden states are saved too.** Cloudflare's head reads every token's hidden state. vLLM's prefix cache can't keep those, which is why vLLM-based Clef servers run with prefix caching off.

**`_result`** applies softmax to the scores, then uses Cloudflare's `systemone_answer` to format each answer. Options are matched by Clef's own ids, because Clef sorts choice options alphabetically.

**`warmup`** runs one short and one long request at startup. GPU kernels compile the first time they are used, so without a warmup the first real user would wait several seconds.

#### `vllm_plugin/`: a doorway, not a second model

The plugin adds `/v1/systemone` to vLLM's server and forwards each request to the engine. Clef doesn't run inside vLLM because vLLM reserves about 90% of GPU memory for its own model by default, and a second model of about 19 GB wouldn't fit.

### Example: one state, three requests

Estimates for an H100.

**Request A.** A 3,000-token support transcript, asking for department and urgency.
- `prepare` finds about 3,036 tokens before the questions, so it takes the long path. Nothing is saved yet, so the cost is about 3,190 tokens.
- Chunk 1 covers tokens 0–2,047. A short batch may run next. Chunk 2 covers tokens 2,048–3,035.
- The engine saves the state (about 170 MB), runs the ~150 question tokens, and scores.
- Total: about **70 ms** of GPU time.

**Request B, a minute later.** Same transcript, now asking for sentiment.
- The answer cache misses, but `state_key` matches, so the cost is about 150 tokens.
- One step: copy the saved state, run the questions, score. About **5–8 ms**.

**Request C.** An exact copy of A. Served from the answer cache, with no GPU work.

## Testing

A faster path is worthless if its answers differ. `smoke_test.py` runs every path and compares it with Cloudflare's own `systemone()` on the same records:
- a short batch of two records;
- a long state, split into chunks;
- that state reused with a new question, which must come from the saved state;
- three identical requests at once, which must run once, then a fourth, which must come from the answer cache.

```bash
python smoke_test.py          # real weights on the GPU; tolerance 0.02 (bfloat16 rounds more)
python smoke_test.py --tiny   # small random model on the CPU; tolerance 0.001
```

`--tiny` uses a small random model with the real tokenizer. It checks the logic, not the answer quality.

## Not in it yet

- FP8 weights, CUDA graphs, and batches without padding (speed work).
- Images and video. Clef supports them; this API accepts only text and JSON.
- Partial reuse when a transcript grows. Today only exact state matches are reused.
- A benchmark script for comparing against vllm-jev.

## Further reading

- [INFERENCE_ENGINE_PLAN.md](INFERENCE_ENGINE_PLAN.md): what a top-tier inference engine is made of, and where vLLM falls short
- [SYSTEMONE_PERF_PLAN.md](SYSTEMONE_PERF_PLAN.md): the plan and decision rules for beating vLLM-based Clef
- [Cloudflare: Introducing Clef](https://blog.cloudflare.com/clef-decision-models/)
- [Clef-flash model card](https://huggingface.co/Cloudflare/clef-flash)
