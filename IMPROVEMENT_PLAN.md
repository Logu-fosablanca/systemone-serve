# Improvement plan: beating vllm-jev on Clef decisions API

**Date:** 2026-10-07
**Target:** Clef-Flash 9B on our EC2 H100-SXM (80 GB, 3.35 TB/s, 989 BF16 TFLOPS)
**Benchmark:** [vllm-jev PR #4](https://github.com/mode-io/vllm-jev/pull/4) on A800-SXM4-80GB

---

## Current state

| | vllm-jev (A800) | Us (RTX 3050, Kev only) | Gap |
|---|---|---|---|
| Serial p50 | 44 ms | 197 ms (Kev-0.8B, different model) | Kernel stack |
| c64 throughput | 37.2 req/s | Not measured on Clef | Kernel stack + batching |
| State reuse | None for Clef | Built (startup prefix, state cache, answer cache) | We're ahead |
| Scheduling | FCFS | Shortest-first with aging | We're ahead |
| DeltaNet kernels | vLLM FlashQLA (fused Triton) | PyTorch reference (~20 launches) | **5-10x** |
| Attention | FlashAttention-2/3 | SDPA | ~20-30% |
| Precision | BF16 | BF16 | Even |
| CUDA graphs | Yes (via vLLM) | No | 10-30% on short inputs |

**Bottom line:** we have better architecture (state cache, scheduling, answer cache) but
worse kernel execution. The plan closes the kernel gap, then our architectural advantages
compound on top.

---

## Improvement 1: Fused DeltaNet kernels

### What

Install [flash-linear-attention](https://pypi.org/project/flash-linear-attention/) v0.5.2
and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) so the 24 Gated DeltaNet
layers run fused Triton kernels instead of the reference PyTorch path.

### Why this is the biggest single improvement

24 of 32 Clef layers are Gated DeltaNet. The fused kernel runs the entire recurrence
(gate, delta rule update, output projection) in **1 kernel launch**, keeping state in
registers/shared memory. The reference path takes **~20 separate kernel launches**, each
reading from and writing to global memory.

Published numbers:
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention): up to 50x
  speedup on DeltaNet layers vs PyTorch reference
- [FlashQLA](https://github.com/QwenLM/FlashQLA) (Qwen's own kernel, optional backend for
  flash-linear-attention): 2-3x forward speedup over the FLA Triton baseline on Hopper/Blackwell
- [MLSys 2026 FlashInfer Contest](https://arxiv.org/pdf/2607.16831): top submission achieved
  1.58x over baseline fused kernel for Gated DeltaNet prefill

Since the DeltaNet layers dominate Clef's wall time (~75% of the backbone), even a
conservative 5x on those layers translates to **3-4x overall backbone speedup**.

### How

```bash
# On EC2 H100 (Linux + CUDA required; will NOT build on Windows RTX 3050)
uv add "flash-linear-attention[cuda,conv1d]"
```

The `[cuda]` extra pulls the correct torch/triton wheels. The `[conv1d]` extra installs
causal-conv1d for the fast 1D convolution path.

**Verification:** our existing `_check_kernels()` in `runtime.py:96-107` already checks
`modeling_qwen3_5.is_fast_path_available` and refuses to start on CUDA without it (unless
`CLEF_ALLOW_SLOW_KERNELS=1`). No code changes needed — just install the packages.

For Kev: `kev`'s `LoadOptions.from_env()` already respects `KEV_FUSED=1` when
flash-linear-attention is present. Same install, same benefit.

### Optional: FlashQLA on top

[FlashQLA](https://github.com/QwenLM/FlashQLA) adds another 2-3x on Hopper (SM90) over
FLA's Triton kernels. Requires CUDA 12.8+ and PyTorch 2.8+.

```bash
pip install flash-qla
```

FlashQLA registers as a backend for flash-linear-attention automatically. No code changes.

### Expected gain

| Scenario | Before | After | Speedup |
|---|---|---|---|
| New 300-token request (backbone) | ~30 ms (estimated H100 with ref kernels) | ~8-10 ms | 3-4x |
| New 4K-token request (backbone) | ~90 ms | ~25-30 ms | 3x |
| New 16K-token request (backbone) | ~380 ms | ~100-130 ms | 3x |

### Risk

None. These are standard packages. The only constraint is Linux + CUDA, which the EC2 H100
already has. If they don't install cleanly, we fall back to the current path (which works).

### Effort: 1 hour install + 1 day benchmarking

---

## Improvement 2: FP8 quantization

### What

Load Clef-Flash from [kurcontko/clef-flash-FP8-Dynamic](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)
instead of BF16. FP8 (E4M3) weights with per-token dynamic activation scaling.

### Why

H100 has native FP8 tensor cores. FP8 halves weight memory (19 GB → ~10 GB) and roughly
doubles matrix multiply throughput.

Published numbers from kurcontko:
- **98.8% top-1 agreement** with BF16 (KL divergence 0.0019 — near-lossless)
- **3.2x throughput** vs original BF16 transformers code
- Components preserved in BF16: joint schema head, vision tower, embeddings, norms

Published numbers from [TorchAO/PyTorch](https://github.com/pytorch/ao/issues/574):
- Llama3.1-8B in FP8 on H100: **28% throughput increase, 21% latency reduction** vs BF16
- **47% VRAM reduction**

### How

Two paths, try in order:

**Path A: Load kurcontko's pre-quantized checkpoint directly**

```python
# In runtime.py ClefRuntime.load():
model, processor = jsm.load_release_model(path, device=device)
# FP8 weights load as float8_e4m3fn tensors; torch._scaled_mm handles matmul
```

Verify FP8 is actually running:
```python
# After loading, check a linear layer's weight dtype
for name, param in model.named_parameters():
    if "linear" in name.lower():
        print(f"{name}: {param.dtype}")  # Should be torch.float8_e4m3fn
        break
```

**Path B: Dynamic quantization with TorchAO (if Path A silently dequants)**

```python
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_
config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
quantize_(model.language_model, config)
```

**Critical check:** HF transformers can silently convert FP8 weights back to BF16 during
loading. After loading, verify with:
```python
import torch
# Profile one forward pass
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    model(batch)
# Look for sm90_xmma_gemm_e4m3 kernels (FP8) vs sm90_xmma_gemm_bf16 (BF16)
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Config

```bash
# New env var
CLEF_FP8=1  # Load FP8 checkpoint; default 0 (BF16)
```

### Expected gain

| Metric | BF16 | FP8 | Improvement |
|---|---|---|---|
| Weight memory | ~19 GB | ~10 GB | 47% less VRAM |
| Matmul throughput | 989 TFLOPS | ~1,979 TFLOPS | ~2x |
| End-to-end throughput | baseline | +50-80% | 1.5-1.8x |
| Accuracy | baseline | 98.8% top-1 agreement | Near-lossless |

### What NOT to do

**NVFP4:** H100 has no FP4 tensor cores. That's Blackwell-only (SM 10.x/12.x). The
[clef-flash-NVFP4](https://huggingface.co/kurcontko/clef-flash-NVFP4) checkpoint won't
help on our hardware.

### Risk

Medium. Silent dequant is the main risk — FP8 weights loaded but matmuls running in BF16.
The profiler check above catches this. If it happens, Path B (TorchAO dynamic quant) is
the fallback.

### Effort: 2-3 days (load, verify, parity gate, benchmark)

---

## Improvement 3: torch.compile

### What

Compile the backbone with `torch.compile(mode="reduce-overhead")` to fuse kernel launches
and eliminate Python overhead in the forward pass.

### Why

For prefill-only workloads (which Clef is — no decode loop), torch.compile delivers:
- [Llama-3.2-3B prefill](https://huggingface.co/docs/transformers/en/perf_torch_compile):
  **2.42x speedup**
- Kernel fusion: adjacent elementwise ops, matmuls, and norms become single fused kernels
- Eliminates Python-to-CUDA dispatch overhead per layer

This matters more for Clef than for generative models because every request is a single
prefill pass — there's no decode phase where other optimizations dominate.

### How

```python
# In runtime.py ClefRuntime.__init__(), after loading:
import torch
self.text_model = torch.compile(self.text_model, mode="reduce-overhead")
```

`reduce-overhead` uses CUDA graphs under the hood for the compiled regions, which
subsumes a separate CUDA graphs implementation.

**Caveats:**
- First call is slow (compilation). Our `warmup()` already runs both paths, so this is
  handled — compilation happens during warmup, not on the first real request.
- Variable input lengths cause recompilation. `torch.compile` with `dynamic=True`
  generates kernels that handle dynamic shapes:
  ```python
  self.text_model = torch.compile(self.text_model, mode="reduce-overhead", dynamic=True)
  ```
- The joint schema head has Python control flow (loops over questions/options). Don't
  compile it — compile only the backbone (`text_model`), not the whole model.

### Why this replaces separate CUDA graphs

Our earlier CUDA graph experiment on RTX 3050 failed because:
1. 800 MiB of graph buffers on a 4 GB card
2. Graphs were unreachable from the code path we were using

`torch.compile(mode="reduce-overhead")` handles both:
1. H100 has 80 GB — graph buffers are negligible
2. It captures graphs for the compiled regions automatically, no manual graph management

[SGLang benchmarks](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/) show
**1.93x speedup** with full CUDA graph capture on prefill-only workloads, **constant
across a 32x range in prompt length**.

### Expected gain

| Input length | Without compile | With compile | Speedup |
|---|---|---|---|
| 300 tokens | baseline | ~1.5-2x faster | Launch overhead dominates |
| 4K tokens | baseline | ~1.3-1.5x faster | Fusion helps |
| 16K tokens | baseline | ~1.1-1.2x faster | Compute dominates |

### Risk

Low. If compilation fails on some code path, torch.compile falls back to eager execution
automatically. The main risk is unexpected graph breaks in the Qwen3.5 DeltaNet code that
prevent full fusion — fixable by marking those points with `torch._dynamo.allow_in_graph`.

### Effort: 1-2 days (add one line, warmup, benchmark, fix any graph breaks)

---

## Improvement 4: Packed batching (cu_seqlens)

### What

Replace `collate_records` (pads every record to the longest in the batch) with packed
sequences using `cu_seqlens` (cumulative sequence lengths).

### Why

Current `run_short` pads all records to the same length. In a batch of [150, 300, 280, 160]
tokens, every record is padded to 300 — that's 310 wasted tokens of compute (26% waste).

Our length-grouped scheduler (Improvement already shipped) reduces this by grouping similar
lengths, but packing eliminates it entirely: all tokens are concatenated into one flat
sequence, and `cu_seqlens` tells the kernel where each record starts and ends.

Key 2026 finding: [Qwen3.5 DeltaNet optimization work](https://github.com/verl-project/verl/pull/7833)
discovered that reading packed lengths from `cu_seqlens_cpu` instead of `cu_seqlens[-1].item()`
eliminated 72 stream syncs per micro-batch across 24 DeltaNet layers. This means
**flash-linear-attention already supports cu_seqlens for DeltaNet layers**.

### How

The seam is marked in `runtime.py:192` with a `# ponytail:` comment:

```python
# Current (padded):
batch = self.jsm.collate_records([job.enc for job in jobs], self.tokenizer.pad_token_id, self.device)

# New (packed):
# 1. Concatenate all token ids into one flat tensor
all_ids = torch.cat([torch.tensor(job.enc.input_ids, device=self.device) for job in jobs])
# 2. Build cu_seqlens
lengths = [len(job.enc.input_ids) for job in jobs]
cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=self.device)
torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=self.device), dim=0, out=cu_seqlens[1:])
# 3. Run backbone with packed input
out = self.text_model(input_ids=all_ids.unsqueeze(0), cu_seqlens=cu_seqlens, use_cache=False)
# 4. Split hidden states back per record using cu_seqlens
```

The joint head already processes records one at a time, so it just needs the per-record
hidden states — which we split from the packed output using `cu_seqlens`.

### Expected gain

Depends on length variance in real traffic. Estimated:

| Length variance | Padding waste (current) | Gain from packing |
|---|---|---|
| Low (all ~300 tok) | ~5% | ~5% throughput |
| Medium (150-600 tok) | ~25% | ~20% throughput |
| High (100-4000 tok) | ~60% | ~40% throughput |

### Risk

Low-medium. Needs Clef weights to verify parity (the head reads every position, so packed
positions must match padded positions exactly). DeltaNet layers' running state accumulates
differently when sequences are packed vs padded — must verify with the parity gate.

### Prerequisite

Improvement 1 (fused DeltaNet kernels). The reference PyTorch DeltaNet path may not
support cu_seqlens; flash-linear-attention does.

### Effort: 3-5 days (implement, parity gate, benchmark)

---

## Improvement 5: Joint head vectorization

### What

Replace the Python loops in Cloudflare's joint schema head with batched tensor operations.

### Why

The reference `joint_schema_model.py` head implementation loops over records, questions,
and options in Python:
```python
for record in records:
    for question in record.questions:
        for option in question.options:
            # compute score
```

Each iteration dispatches small CUDA kernels. For a schema with 10 questions × 5 options
= 50 kernel launches per record, plus Python interpreter overhead per iteration.

[TensorFold](https://github.com/ashhart/TensorFold/pull/241) already does this: their
`hidden_rows()` routes bf16 projections to cuBLAS, achieving 95 ms end-to-end on
DGX Spark vs 174 ms with the reference head (1.83x, but includes backbone too).

### How

**Only do this if profiling shows the head > 10% of total latency.** Profile first:

```python
import torch
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    # Run one batch through backbone
    hidden = backbone(batch)
    # Time the head separately
    head_start = torch.cuda.Event(enable_timing=True)
    head_end = torch.cuda.Event(enable_timing=True)
    head_start.record()
    logits = model.head(hidden, ...)
    head_end.record()
    torch.cuda.synchronize()
    print(f"Head: {head_start.elapsed_time(head_end):.1f} ms")
```

If head > 10% of latency, vectorize:
1. Batch the `memory_projection` across all positions (already a single matmul)
2. Gather question-span and option-span means using `torch.segment_reduce` or index_select
3. Run the routing/decoder layers on the batched spans
4. Scatter results back per question

### Expected gain

| Schema complexity | Head share of latency | Gain from vectorization |
|---|---|---|
| 3 questions × 3 options | ~5% | Not worth it |
| 10 questions × 5 options | ~15% | 10-12% end-to-end |
| 20 questions × 10 options | ~25% | 15-20% end-to-end |

### Risk

Low — the head is a small, self-contained module. Parity is easy to verify (same inputs,
same probability outputs).

### Prerequisite

Profiling data from H100. Don't build this speculatively.

### Effort: 3-5 days (profile, implement, verify)

---

## Improvement 6: Async encoding pipeline

### What

Pipeline CPU encoding and GPU execution so they overlap instead of running sequentially.

### Why

Current flow per batch:
```
[CPU: encode request] → [wait] → [GPU: forward pass] → [wait] → [CPU: encode next] → ...
```

With pipelining:
```
[GPU: forward batch N] ──────────────┐
[CPU: encode batch N+1] ─────┐       │
                              ▼       ▼
                    [GPU: forward batch N+1] ──────────────┐
                    [CPU: encode batch N+2] ─────┐         │
```

vllm-jev's big serial win (207ms → 44ms on the reference) partly comes from this: vLLM's
async engine overlaps encoding with GPU execution. We already encode on a CPU thread pool
(`_cpu` executor in `scheduler.py:62`), but the scheduler waits for encoding to complete
before submitting to the GPU.

### How

The scheduler's `_enqueue` already runs encoding on `self._cpu`. The improvement is to
let `_step_short` start encoding the *next* batch's requests while the GPU processes the
current batch. This is mostly a scheduler change:

```python
# In _step_short: after dispatching current batch to GPU, immediately
# start encoding pending requests on CPU thread pool
gpu_future = loop.run_in_executor(self._gpu, self.rt.run_short, batch)
# While GPU works, encode any pending raw requests
for pending in self._raw_queue:
    loop.run_in_executor(self._cpu, self._encode_and_enqueue, pending)
results = await gpu_future
```

### Expected gain

Depends on encoding time vs GPU time:
- Short requests (~300 tokens): encoding ~2-5 ms, GPU ~8-10 ms → ~20-30% latency reduction
- Long requests (~4K tokens): encoding ~10-20 ms, GPU ~25-30 ms → ~30-40% latency reduction

Throughput gain under load: encoding no longer blocks GPU batching, so batches form faster.

### Risk

Low. The encoding is already threaded; this just changes when it runs relative to GPU work.

### Effort: 2-3 days

---

## Improvements already built (compound advantage)

These are shipped and working. They don't need implementation — they need *Clef weights*
to be verified and benchmarked.

### 7. State cache

**What it does:** saves DeltaNet running state + K/V cache + `memory_projection(hidden)`
(2 KB/token) after computing a state. Next request with the same state skips straight to
the questions.

**Why vllm-jev can't do this:** vLLM's prefix cache doesn't store hidden states. Clef's
joint head reads every position's hidden state. Prefix cache skips recomputing those
positions, so the hidden states are never produced.
[clef-flash-NVFP4](https://huggingface.co/kurcontko/clef-flash-NVFP4) says explicitly:
"prefix caching must be off."

**Estimated gain on H100:**

| State length | Recompute cost (with fused kernels) | Cache hit cost | Saved |
|---|---|---|---|
| 1K tokens | ~22 ms | ~7 ms (questions only) | 15 ms |
| 4K tokens | ~30 ms | ~7 ms | 23 ms |
| 16K tokens | ~130 ms | ~8 ms | 122 ms |

### 8. Startup prefix precompute

Clef's system prompt is identical across all requests. We compute it once at warmup.
Every request starts from this point instead of from scratch.

**System prompt length:** measured by `_prefix_len()` — typically ~100 tokens.

**Gain:** ~100 tokens saved per request. At ~21 µs/token on H100, that's ~2 ms per
request. Small per-request, but it's free (already built) and it applies to every
single request.

### 9. Answer cache + in-flight merging

**Answer cache:** LRU, 10K entries. Exact repeats (same state, same questions) return
immediately with zero GPU work.

**In-flight merging:** if two identical requests are in flight at the same time, the
second one shares the first one's GPU job instead of running a duplicate.

**Measured gain (Kev-0.8B on RTX 3050):** +110% throughput, -51% p50 at 75% repeat rate.

### 10. Shortest-first scheduling

Short requests run before long ones (with aging so long requests aren't starved).

**Why it matters:** vllm-jev uses FCFS. Under mixed load (90% short + 10% 16K-token
requests), a short request can wait behind a 16K-token request. Our scheduler runs
the short request first and chunks the long one.

**Measured on Kev:** not directly measured as a separate lever, but the architecture
is in place and benchmarkable on H100.

---

## Implementation order and timeline

```
Week 1: Ground truth on H100
├── Day 1-2: Install flash-linear-attention + causal-conv1d (Improvement 1)
│            Verify is_fast_path_available == True
│            Run smoke_test.py on Clef with real weights
├── Day 3:   Baseline benchmark: our engine vs vllm-jev vs reference, c1/c16/c64
│            Profile: backbone vs head vs encoding vs Python overhead
├── Day 4-5: Measure state cache, startup prefix, answer cache on real Clef traffic
│            Fill in Phase 0 baseline table in SYSTEMONE_PERF_PLAN.md

Week 2: Close the kernel gap
├── Day 1-2: FP8 quantization (Improvement 2)
│            Load kurcontko checkpoint, verify FP8 matmuls fire
│            Re-run parity gate and benchmark
├── Day 3-4: torch.compile (Improvement 3)
│            Compile backbone, fix any graph breaks
│            Benchmark: compile vs no-compile at c1/c16/c64
├── Day 5:   Decision gate: are we within 10% of vllm-jev on new requests?
│            If yes → proceed to Week 3
│            If no  → profile, find the remaining gap, iterate

Week 3: Pull ahead
├── Day 1-3: Packed batching (Improvement 4)
│            Implement cu_seqlens path, parity gate
│            Benchmark: packed vs padded across length distributions
├── Day 4-5: Head vectorization (Improvement 5) — only if profiling says > 10%
│            Async encoding pipeline (Improvement 6)

Week 4: Production
├── Day 1-2: Re-benchmark everything: c1/c16/c64, repeat rate 0/25/50/75/90%
│            Compare against vllm-jev at every point
├── Day 3-5: Production hardening (SYSTEMONE_PERF_PLAN.md Phase 5)
│            nginx routing, metrics, health propagation
```

---

## Decision gates

| Gate | Metric | Pass | Fail action |
|---|---|---|---|
| **G1: Kernel parity** (end of Week 1) | Our new-request p50 vs vllm-jev | Within 2x | Profile; install FlashQLA; check for silent dequant |
| **G2: Throughput parity** (end of Week 2) | Our c64 throughput vs vllm-jev | Within 10% | FP8 path B (TorchAO); packed batching early |
| **G3: Advantage** (end of Week 3) | Our 75%-repeat throughput vs vllm-jev | > 1.5x | Ship anyway — state cache is the main differentiator |
| **G4: Production** (end of Week 4) | Parity gate on 1000 records, p95 < target | Pass | Fix regressions before deploy |

---

## Projected final numbers

After all improvements, on H100 for Clef-Flash 9B:

| Scenario | vllm-jev (A800) | Us (H100, projected) | Why |
|---|---|---|---|
| New request, c1 p50 | 44 ms | **30-40 ms** | Fused kernels + FP8 + compile; H100 > A800 |
| New request, c64 | 37.2 req/s | **35-45 req/s** | Match or beat via FP8 + packing |
| Same state, new questions, c1 | 44 ms | **7-10 ms** | State cache hit |
| Same state, new questions, c64 | 37.2 req/s | **60-80 req/s** | Cache hits + merging |
| 75% repeat traffic, c16 | ~34 req/s | **80-120 req/s** | Cache + merging + scheduling |
| 16K-token state, 2nd ask | ~1,300 ms | **~8 ms** | State cache saves 16K recompute |
| Agent session (20 turns) | ~69K tokens total | **~10K tokens** | Resume from previous turn |

**The key insight:** on cold traffic we match vllm-jev. On warm traffic (the common case
for agent sessions and repeated decisions), we're 2-5x faster because we don't repeat
work they're forced to repeat.

---

## What we're NOT building

| Idea | Why skip |
|---|---|
| Custom CUDA kernels | flash-linear-attention + FlashQLA already exist |
| NVFP4 quantization | H100 has no FP4 tensor cores; Blackwell-only |
| Tensor parallelism | 9B model fits on one GPU; TP adds communication overhead |
| Replacing our stack with vllm-jev | Pins vLLM version; loses state cache, answer cache, scheduling |
| Speculative decoding | Clef has no decode step |
| Request merging (different questions) | Questions attend to each other in backbone + head |

---

## Sources

- [flash-linear-attention v0.5.2 (PyPI)](https://pypi.org/project/flash-linear-attention/)
- [FlashQLA: Qwen's fused linear attention kernels](https://github.com/QwenLM/FlashQLA)
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
- [kurcontko/clef-flash-FP8-Dynamic](https://huggingface.co/kurcontko/clef-flash-FP8-Dynamic)
- [TorchAO FP8 inference RFC](https://github.com/pytorch/ao/issues/574)
- [torch.compile for inference (HuggingFace)](https://huggingface.co/docs/transformers/en/perf_torch_compile)
- [SGLang advanced CUDA graphs (2026)](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph/)
- [Qwen3.5 cu_seqlens optimization](https://github.com/verl-project/verl/pull/7833)
- [vllm-jev PR #4: Clef-Flash benchmarks](https://github.com/mode-io/vllm-jev/pull/4)
- [clef-flash-NVFP4: prefix caching must be off](https://huggingface.co/kurcontko/clef-flash-NVFP4)
- [MLSys 2026 FlashInfer Contest: Gated DeltaNet](https://arxiv.org/pdf/2607.16831)
