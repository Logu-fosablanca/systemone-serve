# Clef /v1/systemone: Performance Plan

**Goal:** serve Cloudflare Clef-flash on our H100 faster than the existing vLLM-based System One servers, and prove it with our own measurements before we commit.

**Status (2026-10-07):** nothing *about Clef* has been measured on our hardware. Numbers marked *est.* come from the cost model in section 4. Every phase ends with a pass/fail gate.

What has been measured, on a different model and a laptop GPU, is in [BENCHMARKS.md](BENCHMARKS.md): Kev-0.8B bf16 on an RTX 3050. It does not validate any number in this document, but it does validate three of its assumptions — that repeat traffic is the dominant lever (+110% throughput at 75% repeat), that batching gains need the GPU to be bandwidth-bound (on CPU the same sweep is flat), and that in-flight merging is worthless below concurrency 2. It also contradicts nothing here. Phase 0's baseline table is still empty.

---

## 1. Summary

Two community projects already serve Clef through vLLM: [vllm-jev](https://github.com/mode-io/vllm-jev/pull/4) and [clef-flash-NVFP4](https://huggingface.co/kurcontko/clef-flash-NVFP4). They run Clef's backbone on vLLM's optimised kernels. We assume they are close to the GPU's limit on requests they haven't seen before, so we won't try to out-run their kernels. We use the same kernel libraries and aim to match them there.

We get ahead by not repeating work they are forced to repeat. Clef's joint head reads the final hidden state of every input token, and vLLM's prefix cache doesn't keep those. So those servers run with prefix caching off ("Prefix caching must be off", clef-flash-NVFP4 README), and every request recomputes its whole input, including Clef's fixed system prompt. Our engine saves exactly what the head needs and resumes from it.

Three things their design doesn't do today:

1. **Reuse that works with the joint head.** Clef's fixed system prompt is computed once at startup. A state asked about twice is computed once. A growing agent transcript computes only its new part.
2. **Use determinism.** Clef doesn't sample, so identical requests can share one result: an answer cache, plus merging identical requests that are in flight at the same time.
3. **Schedule for prefill-only work.** Each request's cost is known when it arrives. We run short work first (with a cap on how long anything waits) and split long inputs into chunks.

One prerequisite: today our runtime is slower than vLLM on new requests (eager HF code, padded batches, guessed token counts). Phase 1 closes that gap. If it can't, the decision rules in section 8 choose the vLLM path instead.

---

## 2. How Clef inference works (from its source)

- **Input order:** `system prompt + "STATE:" + media + state + questions/options + closing tokens`. The state comes before the questions.
- **Backbone (clef-flash):** Qwen3.5, 32 layers:
  - 24 Gated DeltaNet layers: linear attention that carries a fixed-size running state instead of per-token keys and values (K/V);
  - 8 full-attention layers, one in every four (16 query heads, 4 K/V heads, head size 256);
  - causal. Clef calls it with `use_cache=False` and takes only the final hidden states; no vocabulary logits are computed.
- **Joint head** (`joint_head_config.json`: width 1024, 2 routing layers, 4 decoder layers, 16 heads) reads:
  - every position, through `memory_projection` (4096 → 1024), as "memory";
  - the mean hidden state of each question span and each option span;
  - the last position;
  - for each option, the mean of its tokens' rows in the output embedding (a lexical prior).

  It loops over records, questions and options in Python.
- **Reference `systemone()`** runs one record per call.

What follows:
- There is no decode loop and no sampling. The whole job is one prefill pass plus the head.
- The backbone is causal and the state comes before the questions. Everything computed for the system prompt and the state is independent of the questions, so it can be saved and reused exactly.
- The head needs every position's hidden state. A prefix cache skips recomputing cached tokens and so never produces those hidden states. That is why vLLM's prefix cache can't be used with Clef.

---

## 3. What we're comparing against

Both projects subclass vLLM's `Qwen3_5ForConditionalGeneration` as a pooling model (vLLM's term for models that return vectors instead of text). The pooler collects each request's final hidden states across chunked prefill, then runs Cloudflare's original joint head. Question and option spans travel in `PoolingParams.extra_kwargs`.

| | vllm-jev (PR #4) | clef-flash-NVFP4 |
|---|---|---|
| vLLM version | 0.29.0, pinned | 0.28.x, pinned ("uses vLLM pooling internals") |
| Precision | BF16 | NVFP4 (Blackwell GPUs only) |
| GPU tested | A800-SXM4-80GB | RTX 5070 Ti |
| Prefix caching | not stated; the same head constraint applies | must be off |
| Chunked prefill | yes | yes |
| Images / video | yes | rejected |
| Published numbers | 1 client: p50 44 ms. 64 clients: p50 1,846 ms at 37.2 req/s | 64 clients: 39.7 req/s, p50 0.95 s, p95 5.1 s |
| Accuracy note | top-1 identical to Cloudflare's reference | variance on borderline cases |

**What it does well, and we must match:** packed batches, chunked prefill, vLLM's DeltaNet and attention kernels, CUDA graphs, FP8/FP4 matrix multiplies.

**What it doesn't do:**
1. Reuse any prefix. The system prompt and the state are recomputed on every request.
2. Notice that two requests are identical.
3. Order work by cost. Both use first come, first served. At 64 clients their published median latency is about 1-2 s, so short requests wait behind long ones.
4. Upgrade vLLM independently. Each is pinned to one vLLM version.

---

## 4. Cost model: why each lever matters

Estimates for clef-flash on one H100 SXM (3.35 TB/s memory bandwidth, 989 dense BF16 TFLOPS), derived from `config.json`:

- About 6.9B parameters outside the embeddings: about **13.8 GFLOP per input token**, and about **13.8 GB of weights read per forward pass**.
- Reading the weights takes about **4.1 ms** per forward pass, however few tokens it carries.
- Compute costs about **21 µs per token** at ~65% utilisation.
- The two cross at about **200 tokens per forward pass**:
  - below it, the GPU waits on memory, so batching several short requests together is nearly free;
  - above it, time grows with tokens, and the only big levers are fewer tokens (reuse) or cheaper tokens (FP8, up to ~1.5-1.8x).
- Full attention is about 1% of the work at 2K tokens and 8% at 16K, because only 8 of 32 layers use it. **FlashAttention-3 vs FlashAttention-2 is worth a few percent for this model, not the 1.5x claimed earlier in this project.**

Prefill time per request (*est.*): 300 tokens ≈ 7-10 ms · 1K ≈ 22 ms · 4K ≈ 90 ms · 16K ≈ 380 ms.

Memory needed to keep one saved state:

| Part | Size |
|---|---|
| K/V for the 8 attention layers | 32 KB per token |
| Head memory, already projected to width 1024 | 2 KB per token |
| DeltaNet running state, 24 layers (fp32) | ≈50 MB, fixed |
| Convolution state | ≈1.2 MB, fixed |

So a saved 1K-token state ≈ 85 MB, 4K ≈ 190 MB, 16K ≈ 600 MB. On a dedicated H100, about 45 GB is left for saved states after weights and working memory: roughly 230 states of 4K tokens. Host RAM holds more. Reloading 190 MB over PCIe takes ≈4-8 ms, against ≈90 ms to recompute it.

---

## 5. Where the gains come from (*est.*)

| Scenario | vLLM-based Clef | Our engine | Why |
|---|---|---|---|
| New, unique request | baseline | about the same (±10%) | same kernel libraries |
| Clef's fixed system prompt | recomputed every request | computed once at startup | saved prefix |
| 2nd+ decision on the same 4K state | ≈90 ms | ≈7 ms | saved state |
| Same, 16K state | ≈380 ms | ≈8 ms | saved state |
| 20-turn agent session, +300 tokens per turn | ≈69K tokens computed | ≈10K tokens | resume from the previous turn |
| Exact duplicate or retry | full recompute | no GPU work | answer cache, in-flight merge |
| p95 of short requests under mixed load | waits behind long ones | lower | shortest first, chunk long inputs |

The agent-session row assumes a 200-token question block and a 100-token system prompt.

How much of this we actually get depends on our traffic. Phase 0 measures it.

---

## 6. Target architecture

```
client ──► nginx (TLS + one VLLM_API_KEY check)
             ├── /v1/systemone ──► Clef engine (own process; own GPU or MIG slice)
             └── everything else ─► vLLM serving the LLM (unchanged, upgradable)

Clef engine, per request:
  1. validate, encode on a CPU thread pool (exact token counts)
  2. same request answered recently, or already in flight? → reuse that result
  3. find the longest saved snapshot (startup prefix, earlier state, previous turn)
  4. scheduler: shortest remaining work first, max wait, chunk long inputs, 429 when full
  5. backbone: resume from the snapshot, compute only the new tokens
  6. joint head, batched across records → answers; save a snapshot if worth keeping
```

One mechanism does three jobs: **resume the backbone from a saved snapshot** (attention K/V + DeltaNet state + head memory). It gives us the startup prefix, the state cache and chunked prefill. We build it once.

Snapshots sit on 64-token block boundaries, found by cumulative hashes of token ids (the same idea as vLLM's prefix cache). A hit resumes from the deepest matching block and recomputes at most 63 tokens. Keys include the model revision and image hashes, so a hit is always exact.

---

## 7. Phases

### Phase 0: Ground truth (2-3 days)
- Turn `smoke_test.py` into a parity harness: 1,000 records, including images and max-length states, compared with Cloudflare's `systemone()`.
- Analyse a sample of request logs (hashed): repeated states, exact duplicates, growing transcripts, length and option-count distributions. Measure Clef's system-prompt length (`len(prefix_ids)`).
- Baselines on our H100 with one benchmark client: Cloudflare's reference, our current runtime, and vllm-jev as a separate vLLM 0.29.0 instance.
- Profile one batch: backbone vs head vs encoding vs Python overhead. Confirm the fast DeltaNet kernel is the one running; HF transformers falls back to a much slower PyTorch version when it isn't available.
- Read the other Clef servers before writing new code ([TensorFold PR #241](https://github.com/ashhart/TensorFold/pull/241), [hachidori PR #241](https://github.com/yohn-jp/hachidori/pull/241)). One may already solve part of Phase 1 or 2.
- **Exit:** baseline table filled in; share of reusable tokens estimated.

### Phase 1: Match vLLM on new requests (3-5 days)
- Pin torch 2.11 and transformers 5.10.2 (Cloudflare's tested versions) and the model repo revision.
- Bundle the DeltaNet kernel in the image and fail startup if the slow path is active. An EC2 host without Hugging Face Hub access may not be able to fetch it at runtime.
- Encode records on a CPU thread pool when they arrive, so batching uses exact token counts.
- Start a batch as soon as the GPU is idle. The current batcher waits up to 5 ms even then.
- Group similar lengths to cut padding waste.
- Answer cache and in-flight merging of identical requests.
- Remove the head's per-record and per-option Python loops, but only if profiling shows the head above ~10% of latency (likely for schemas with dozens of options).
- **Exit:** within 10% of vllm-jev on throughput at 64 clients and on p50 at 1 client; parity gate passes.

### Phase 2: Snapshot, resume, state cache (1-2 weeks)
- Run the backbone in two segments: `[system prompt + state]`, then `[questions + closing tokens]` resuming from the saved cache. Current HF code resumes DeltaNet layers from a cached state in multi-token calls (`initial_state=recurrent_state`); confirm it in the pinned version with the parity gate.
- Save the head's projected memory (2 KB per token), not raw hidden states (8 KB). First confirm in the source that the head reads state positions only through `memory_projection`.
- Compute the system prompt once at startup; every request resumes from it.
- State cache: GPU tier plus a host-RAM tier; evict by tokens saved per byte.
- Growing transcripts (only if Phase 0 finds them): also keep the second-to-last block of each state. Appending to a transcript changes its last few tokens (closing JSON characters, tokenizer merges), so the next turn diverges just before the end.
- **Exit:** cached vs uncached on 1,000 records (including images) give the same top answer on ≥99.9%, with every probability within 0.005; speedup measured on repeated-state traffic.

### Phase 3: Scheduler (≈1 week)
- Chunked prefill through the same resume path (for example, 2K-token chunks).
- Shortest remaining work first, with a maximum wait so long requests aren't starved; a token budget per forward pass; HTTP 429 when the queue holds more work than the latency target allows.
- **Exit:** in a 90% short / 10% 16K mix, short-request p95 improves vs first come, first served; long-request p99 is no worse than 2x its first-come value.

### Phase 4: Throughput ceiling (only if profiling calls for it, 1-2 weeks)
- FP8 from a pre-quantized checkpoint ([kurcontko/clef-flash-FP8-Dynamic](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)). Check that FP8 matrix multiplies actually run; HF may convert weights back to BF16 on the fly, which gives no speedup. Re-run the parity gate.
- CUDA graphs for small token counts. Budget the buffers before enabling: on a 4 GB card they cost ~800 MiB and collapsed throughput 12-17x under concurrency ([BENCHMARKS.md](BENCHMARKS.md#cuda-graphs-a-negative-result)). An H100 has the room; the lesson is that graphs must be measured against the memory left for activations, not assumed free.
- Packed batches mixing new and resumed sequences. HF's qwen3_5 DeltaNet path already accepts packed sequences (`cu_seqlens`).

### Phase 5: Production and upstream (≈1 week)
- nginx routing as in section 6, both services checking the same `VLLM_API_KEY`.
- Replace the current vLLM endpoint plugin with a thin proxy. As built, it loads a second Clef copy (≈19 GB) inside vLLM's API-server process. That won't fit beside vLLM's default 90% GPU memory reservation, and with several API-server processes each would load its own copy.
- Metrics: cache hit rates, tokens saved, queued work, tokens per batch, p50/p95/p99.
- Write the vLLM proposals in section 10, with our measured numbers.

---

## 8. Decision rules (fixed before we measure)

Let **R** be the share of prefill tokens a saved snapshot would have covered (from Phase 0), and **G** our throughput on unique traffic relative to vllm-jev (from Phase 1; G = -15% means we are 15% slower). Our GPU cost per request, relative to vLLM-based Clef, is about **(1 - R) / (1 + G)**.

| Result | Action |
|---|---|
| below 0.9 | Ship our engine. |
| 0.9 to 1.1 | Ship ours only if short-request p95 under mixed load is better (Phase 3). |
| above 1.1 | Run vllm-jev as its own vLLM instance; keep our front door (answer cache, in-flight merging, admission control) in front of it. |
| any | Never load Clef inside the LLM's vLLM process. |

Clef's system prompt alone adds to R on every request. If it is 100 tokens and the average request is 500 tokens, R is at least 20% even with no repeated states.

R lowers GPU time per request, which matters under load. When the GPU is idle, a short request's latency is set by the ~4 ms weight read regardless.

---

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Resumed computation differs from a full pass (DeltaNet conv state, image position offsets) | Wrong answers on cache hits | Cached-vs-uncached gate with image cases; cache stays off until it passes |
| Slow DeltaNet fallback active in production | Several times slower | Kernel bundled in the image; startup check |
| Our traffic has little reuse | Main advantage disappears | Phase 0 measures R before Phase 2 starts; section 8 rules |
| FP8 in HF gives no speedup | Phase 4 gain lost | Measure; swap linear layers to `torch._scaled_mm`, or skip FP8 |
| H100 shared with the LLM | Less room for saved states; tail latency from contention | Dedicated GPU or MIG slice (open question 1) |
| Cloudflare updates weights or code | Mismatched outputs or stale cache | Pin model revision; revision in every cache key |
| Batch composition changes rounding | Tiny probability differences between runs | Tolerance-based gates; a cached answer is still a valid output of the model |

---

## 10. What vLLM can learn (upstream proposals)

1. **Prefix caching for pooling models whose heads read every token.** Store a model-declared projection of the final hidden states (2 KB per token for Clef) next to the K/V blocks.
2. **Request-declared snapshot points for DeltaNet/Mamba-style layers.** Save the running state at the end of a request's state, not only at fixed block boundaries.
3. **Result cache and in-flight merging** for deterministic pooling requests.
4. **A cost-aware scheduling policy** (shortest first, with aging) for prefill-only work.

---

## 11. Not building

- Custom CUDA kernels. We reuse flash-linear-attention, FlashAttention and existing FP8 matrix multiplies.
- Merging different requests' questions into one pass. Questions attend to each other in the backbone and in the head, so it would change answers.
- Tensor parallelism for a 9B model. One engine per GPU instead.
- NVFP4 on H100. It has no native FP4 support.
- Speculative decoding. Clef has no decode step.

---

## 12. Open questions for the team

1. How many GPUs does the EC2 instance have, and is the H100 shared with the LLM?
2. What p95 latency does the product need?
3. Do production requests include images or video?
4. Clef-flash (9B) or Clef (27B)? Numbers here are for clef-flash; Clef needs roughly 3x the compute.
5. Can we get 1-2 days of hashed request logs for Phase 0?

---

## Interim: serving Clef before this lands

If Clef has to take production traffic before Phase 2, run vllm-jev as its own vLLM 0.29.0 instance (never inside the LLM's vLLM), behind nginx with the same API key. Its PR reports BF16, top-1 answers identical to Cloudflare's reference, and image support. Run our parity harness against it first.

---

## Sources

- [vllm-jev PR #4: Clef-Flash support](https://github.com/mode-io/vllm-jev/pull/4)
- [kurcontko/clef-flash-NVFP4](https://huggingface.co/kurcontko/clef-flash-NVFP4)
- [kurcontko/clef-flash-FP8-Dynamic](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)
- [Cloudflare/clef-flash: config.json, joint_head_config.json](https://huggingface.co/Cloudflare/clef-flash)
- [Cloudflare/clef: joint_schema_model.py](https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py)
- [transformers: modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Cloudflare blog: Introducing Clef](https://blog.cloudflare.com/clef-decision-models/)
- [TensorFold PR #241: Clef on CUDA](https://github.com/ashhart/TensorFold/pull/241)
- [hachidori PR #241: batched Clef inference](https://github.com/yohn-jp/hachidori/pull/241)
