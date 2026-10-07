# Clef serving implementations: comparison and lessons

**Date:** 2026-10-07
**Purpose:** identify concrete improvements for our decisions API speed by comparing every
known Clef serving implementation against our engine.

> **Provenance.** Every figure for a third-party implementation is quoted from that project's
> own README, pull request, release notes, or model card — linked in the table below and under
> [Sources](#sources). None of it was independently measured or reproduced here. The only rows
> measured on our own hardware are the Kev 0.8B numbers, and those are a different model on a
> different GPU class, included for calibration only.
>
> Architectural claims about how each project handles the joint schema head are read from
> published code and project descriptions, not from profiling their runtime.

---

## Implementations surveyed

| # | Project | Approach | Hardware tested | Clef variant |
|---|---------|----------|-----------------|--------------|
| 1 | **vllm-jev** ([PR #4](https://github.com/mode-io/vllm-jev/pull/4)) | vLLM pooling model + joint head in-process | A800-SXM4-80GB | Flash 9B, 27B |
| 2 | **open-jevlike-infer** ([v0.1.0](https://github.com/Arcobalneo/open-jevlike-infer/releases/tag/v0.1.0)) | vLLM 0.30 pooling + dynamic micro-batching | A800-SXM4-80GB | Flash 9B |
| 3 | **clef-flash-NVFP4** ([kurcontko](https://huggingface.co/kurcontko/clef-flash-NVFP4)) | NVFP4 W4A4 quantization + vLLM plugin | RTX 5070 Ti (Blackwell) | Flash 9B |
| 4 | **clef-flash-FP8-Dynamic** ([kurcontko](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)) | FP8 W8A8 quantization + vLLM plugin | RTX 5070 Ti | Flash 9B |
| 5 | **clef-NVFP4** ([kurcontko](https://huggingface.co/kurcontko/clef-NVFP4)) | NVFP4 of full 27B model | 2x RTX 5070 Ti | Clef 27B |
| 6 | **TensorFold** ([PR #241](https://github.com/ashhart/TensorFold/pull/241)) | Custom CUDA engine, cuBLAS bf16 projections | DGX Spark (GB10) | Flash 9B |
| 7 | **hachidori** ([PR #241](https://github.com/yohn-jp/hachidori/pull/241)) | Go supervisor + Python provider, adaptive batching | RTX 3060 (unverified) | Flash 9B |
| 8 | **This engine** (systemone-serve) | HF transformers + state cache + scheduler | RTX 3050 Laptop (Kev only) | Flash 9B (planned) |

---

## Published numbers

All numbers below are what each project reports, not what we measured. Hardware differs
across rows, so direct comparison is misleading — the takeaway column says what matters.

### Clef-Flash 9B

| Implementation | GPU | c1 p50 ms | c16 p50 ms | c64 p50 ms | Peak req/s | Accuracy vs BF16 |
|---|---|---|---|---|---|---|
| vllm-jev BF16 | A800 | 44 | 465 | 1,846 | 37.2 | 100% top-1 |
| open-jevlike-infer | A800 | 45 | — | — | 28 @ c16 | 98.6% ARC |
| clef-flash-FP8 | 2x 5070 Ti | — | — | 132 (p50) | 19.5 | 98.8% top-1 |
| clef-flash-NVFP4 | 1x 5070 Ti | 35 | — | 950 | 39.8 | 94.3% top-1 |
| TensorFold | DGX Spark | 95 | — | — | — | 128/128 top-1 |
| Reference (transformers) | A800 | 207 | — | — | — | baseline |

### Clef 27B

| Implementation | GPU | Throughput | Accuracy vs BF16 |
|---|---|---|---|
| clef-NVFP4 27B | 2x 5070 Ti | 8.4 req/s | 94.7% top-1 |

### Kev 0.8B (our measured numbers, for calibration only)

| Configuration | GPU | c1 p50 ms | c8 p50 ms | c16 p50 ms | Peak req/s |
|---|---|---|---|---|---|
| This engine (forward_batch, bf16) | RTX 3050 | 197 | 736 | 1,116 | 14.0 |
| This engine (75% repeat, c8) | RTX 3050 | 407 | — | — | 19.5 |

---

## Architecture comparison

### How each handles the joint schema head

The joint head is the central constraint. It reads *every* position's hidden state, not
just the last token. This breaks vLLM's prefix cache, which skips recomputing cached
tokens and therefore never produces their hidden states.

| Implementation | Head strategy | Prefix caching |
|---|---|---|
| vllm-jev | Runs Cloudflare's head on vLLM's pooled hidden states | Off (structurally incompatible) |
| open-jevlike-infer | Same as vllm-jev | Off |
| clef-flash-NVFP4 | Same, via custom `ClefFlashForDecision` pooler | Explicitly off |
| TensorFold | `hidden_rows()` extracts every position, feeds cuBLAS head | No cache |
| hachidori | Delegates to Cloudflare's Python head | No cache |
| **This engine** | Saves `memory_projection(hidden)` (2KB/tok) + DeltaNet state + K/V; resumes from it | **Yes: startup prefix, state cache, growing transcripts** |

**Key insight:** every vLLM-based implementation recomputes the entire input on every
request, including Clef's fixed system prompt. They get good absolute latency from
vLLM's kernel stack, but cannot amortise repeated work.

### Kernel stack

| Implementation | DeltaNet layers | Attention | Matrix multiply |
|---|---|---|---|
| vllm-jev | vLLM's built-in (FlashQLA since 0.29) | FlashAttention-2/3 | cuBLAS BF16 |
| clef-flash-NVFP4 | vLLM + FP4 tensor cores | FA-2/3 | FP4 tensor cores |
| clef-flash-FP8 | vLLM + FP8 matmul | FA-2/3 | FP8 (E4M3) |
| TensorFold | Custom CUDA engine | Custom | cuBLAS BF16 |
| **This engine** | **Reference PyTorch fallback** | SDPA | PyTorch matmul |

**This is our single biggest gap.** The fused Triton kernel from flash-linear-attention
runs the entire Gated DeltaNet recurrence in 1 kernel launch; the reference PyTorch path
takes ~20 launches for the same work, each round-tripping through global memory.
Published speedups are up to 50x for the DeltaNet layers alone. Since 24 of 32 Clef
layers are DeltaNet, this dominates wall time.

### Batching and scheduling

| Implementation | Batching | Scheduling |
|---|---|---|
| vllm-jev | vLLM continuous batching | FCFS |
| open-jevlike-infer | Dynamic micro-batching across concurrent requests | FCFS |
| clef-flash-NVFP4 | vLLM continuous batching | FCFS |
| TensorFold | Single request | None |
| hachidori | Adaptive batch collector, bounded by tokens and count | FCFS |
| **This engine** | Length-grouped batching, token budget | **Shortest-first with max wait** |

### State/prefix reuse

| Implementation | Any form of state reuse |
|---|---|
| vllm-jev | Experimental prefix cache for Open-Jev-2B text only; nothing for Clef |
| open-jevlike-infer | No |
| clef-flash-NVFP4 | No |
| TensorFold | No (explicitly refuses state resumption) |
| hachidori | No |
| **This engine** | **Yes: startup prefix, state cache, answer cache, in-flight merging** |

---

## What we should adopt

### Priority 1: Install flash-linear-attention and causal-conv1d (days, not weeks)

Every competitive implementation runs fused DeltaNet kernels. We run the reference
PyTorch path because `flash-linear-attention` is absent from the venv. This is the
single largest gap — estimated 5-10x on the DeltaNet layers, which are 75% of Clef's
backbone.

**Action:** `uv add flash-linear-attention causal-conv1d`. Verify at startup that
`modeling_qwen3_5.is_fast_path_available` is `True`. The existing `_check_kernels()`
in `runtime.py` already gates on this — it just needs the packages installed.

For Kev: `flash-linear-attention` also enables `kev`'s fused path. The `LoadOptions.from_env()`
call already respects `KEV_FUSED=1`. This alone could cut our Kev latency in half.

**Risk:** These packages require Linux + CUDA. They don't build on Windows, which is
why they're absent. Benchmark on the EC2 H100, not the laptop.

### Priority 2: FP8 quantization (Phase 4, already planned)

kurcontko's FP8-Dynamic achieves 98.8% top-1 agreement at 3.2x throughput on Ada/Hopper
hardware. Our H100 has FP8 tensor cores.

**Action:** Use [kurcontko/clef-flash-FP8-Dynamic](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)
as the weight source. Verify that `torch._scaled_mm` is invoked (not silent dequant to BF16).
FP8 is near-lossless (KL 0.0019) so it passes the parity gate easily.

**Not NVFP4:** H100 lacks FP4 tensor cores. NVFP4 is Blackwell-only.

### Priority 3: CUDA graphs for the fused DeltaNet path

Once flash-linear-attention is installed and the runtime uses `probs_batch` (for Kev)
or the fused backbone (for Clef), CUDA graphs become viable. Our negative result on the
RTX 3050 was a memory-pressure artifact: 800 MiB of graph buffers on a 4 GB card. An
H100 with 80 GB has room.

**Action:** Enable after Priority 1. Measure on H100. The kernel launch overhead
CUDA graphs eliminate matters most when individual kernels are fast (which fused
kernels are).

### Priority 4: Packed batching (Phase 4, already scaffolded)

Replace `collate_records` (pads every record to the longest in the batch) with
`cu_seqlens` packed sequences. HF's Qwen3.5 DeltaNet path already accepts them.
Our length-grouped scheduler reduces but does not eliminate padding; packing eliminates
it entirely.

**Action:** Implement after Priorities 1-2 are measured. The `# ponytail:` comment in
`runtime.py:run_short` marks the seam.

### Priority 5: Head computation batching

TensorFold routes head projections to cuBLAS explicitly. Cloudflare's reference head
loops over records, questions, and options in Python. If profiling shows the head above
~10% of latency (likely for schemas with dozens of options), vectorize the inner loops.

**Action:** Profile on H100 first. This is already in SYSTEMONE_PERF_PLAN.md Phase 1.

---

## What we should NOT adopt

| Idea | Why not |
|---|---|
| Replace our serving stack with vllm-jev | Pins us to one vLLM version; loses state cache, answer cache, scheduling; see SYSTEMONE_PERF_PLAN.md section 8 |
| NVFP4 quantization | H100 has no FP4 tensor cores; Blackwell-only |
| TensorFold's custom CUDA engine | Trades maintainability for moderate gains; fused kernels + FP8 gets us closer with less code |
| hachidori's Go supervisor | Unverified; adds a language boundary; our Python scheduler already does adaptive batching |
| Merging different requests' questions into one pass | Questions attend to each other in the backbone and head — would change answers |

---

## What none of them do (our advantages)

These are capabilities unique to this engine, confirmed absent in every surveyed
implementation:

1. **State cache for Clef's joint head.** Every other implementation recomputes the
   entire input per request. We save the projected hidden states (2 KB/token), K/V
   cache, and DeltaNet state, then resume from them. For a 4K-token state seen twice,
   that's ~90 ms saved per hit on H100.

2. **Startup prefix precompute.** Clef's system prompt is identical across all requests.
   We compute it once at warmup. No other implementation does this.

3. **Answer cache + in-flight merging.** Identical requests (same state, same questions)
   return immediately. Under concurrency, duplicates in flight merge into one GPU job.
   Measured: +110% throughput at 75% repeat rate on Kev.

4. **Cost-aware scheduling.** Short requests run first; long inputs are chunked. Every
   other implementation uses FCFS, so under mixed load their short-request p95 suffers.

5. **vLLM independence.** The engine runs as its own process and loads no model inside vLLM,
   so it is not tied to a vLLM version. An optional plugin forwards `/v1/systemone` from
   vLLM's port to the engine. vllm-jev and the kurcontko quantizations each pin a specific
   vLLM version. (No reverse-proxy config ships with this repo; DEVOPS.md recommends putting
   one in front for TLS.)

---

## Concrete improvement roadmap for decisions API speed

The gain column below is **estimate only**. Nothing in this table has been measured on Clef,
on an H100, or against vllm-jev. Figures are order-of-magnitude guesses from each technique's
general reputation, not from profiling this engine.

| # | Action | Estimated gain (unvalidated) | Status | Notes |
|---|---|---|---|---|
| 1 | Install fused DeltaNet kernels | large; the PyTorch fallback is ~20 kernel launches per layer vs 1 | **Install step only** | `uv sync --extra clef`; the code already uses them when present and refuses to start without them unless `CLEF_ALLOW_SLOW_KERNELS=1` |
| 2 | FP8 weights | halves weight bytes; throughput effect unmeasured | **CODED** (`CLEF_FP8=1`) | Path A (pre-quantized checkpoint) and Path B (TorchAO) both handled; silent dequant detected. Needs SM89+ to execute |
| 3 | torch.compile (includes CUDA graphs) | unknown on Clef | **CODED** (`CLEF_COMPILE=1`) | `mode="reduce-overhead"` enables CUDAGraph trees. Standalone CUDA graphs measured **12-17x worse** under concurrency on a 4 GB card — see [BENCHMARKS.md](BENCHMARKS.md#cuda-graphs-a-negative-result). Whether compile helps on an 80 GB card is untested |
| 4 | Packed batching | removes padding waste; size of effect unmeasured | **CODED** (`CLEF_PACKED=1`) | Parity verified on a small random model only. Gated behind `fla` because the fallback DeltaNet path does not reset recurrent state at `cu_seqlens` boundaries |
| 5 | Head vectorization | unknown; depends on head share of latency | **NOT CODED** | Deferred until profiling shows the head dominates |
| 6 | State cache | unknown | **BUILT** | LRU byte-bounded, deepcopy on resume, budget and hit counts in `/health`. Never yet engaged in a benchmark run — see below |
| 7 | Startup prefix | the fixed system-prompt tokens (~36) per long request | **BUILT** | Precomputed once at warmup; every long request resumes from it |

**What is actually established:** items 2, 3, 4, 6 and 7 exist in the code and match
Cloudflare's reference implementation on a small random model. Item 1 is an install step. That
is the whole of it. No throughput or latency comparison against any other implementation has
been run.

Items 6 and 7 are the two entries that a vLLM-based server cannot straightforwardly adopt,
because they depend on retaining final hidden states that a KV-block prefix cache does not
hold. That is an architectural difference, not a demonstrated win: their value depends on how
often states actually repeat in production traffic, which has not been measured.

One caution from the only repeat-rate sweep run so far (Kev 0.8B, RTX 3050): the state cache
never engaged. The generated states were shorter than `CLEF_LONG_STATE_TOKENS`, so every
request took the short path, and the measured gain came from the answer cache and in-flight
merging — both of which any server can implement. Check `state_cache_bytes` in `/health`
before attributing any result to the state cache.

---

## Sources

- [vllm-jev: Clef-Flash support PR #4](https://github.com/mode-io/vllm-jev/pull/4)
- [vllm-jev README](https://github.com/mode-io/vllm-jev)
- [open-jevlike-infer v0.1.0](https://github.com/Arcobalneo/open-jevlike-infer/releases/tag/v0.1.0)
- [kurcontko/clef-flash-NVFP4](https://huggingface.co/kurcontko/clef-flash-NVFP4)
- [kurcontko/clef-flash-FP8-Dynamic](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)
- [kurcontko/clef-NVFP4 (27B)](https://huggingface.co/kurcontko/clef-NVFP4)
- [TensorFold PR #241: Clef on CUDA](https://github.com/ashhart/TensorFold/pull/241)
- [hachidori PR #241: batched Clef inference](https://github.com/yohn-jp/hachidori/pull/241)
- [Cloudflare: Introducing Clef](https://www.marktechpost.com/2026/10/01/cloudflare-releases-clef-and-clef-flash/)
- [DeltaLog: Deferred Materialization for Linear Attention Decoding](https://arxiv.org/pdf/2608.15533)
- [FlashQLA / flash-linear-attention fused Triton kernels](https://discuss.huggingface.co/t/decay-gated-o-n-causal-linear-attention-with-fused-triton-kernel/178456)
- [MLSys 2026 FlashInfer Contest: Gated DeltaNet optimization](https://arxiv.org/pdf/2607.16831)
