# Measured results

**Everything here was run on real weights.** Nothing in this file is an estimate from the
cost model; the estimates live in [SYSTEMONE_PERF_PLAN.md](SYSTEMONE_PERF_PLAN.md) and are
labelled there. Where this file does extrapolate, the sum is shown so you can disagree with it.

**What has not been measured:** Clef itself (needs ~19 GB), any H100 or datacentre card, and
any head-to-head against vllm-jev. See [Not measured](#not-measured) before quoting anything.

- [Setup](#setup)
- [Throughput vs concurrency](#throughput-vs-concurrency)
- [State reuse](#state-reuse)
- [CPU](#cpu)
- [CUDA graphs: a negative result](#cuda-graphs-a-negative-result)
- [vllm-jev](#vllm-jev)
- [Correctness](#correctness)
- [Measurement hygiene](#measurement-hygiene)
- [Not measured](#not-measured)

## Setup

| | |
|---|---|
| Model | `jaredpalmer/kev-0.8b` — Qwen3.5-0.8B-Base, 24 layers (18 Gated DeltaNet + 6 full attention), LoRA r16, pointer head, T=2.35 |
| GPU | RTX 3050 Laptop, 4,096 MiB, sm_86, driver 555.99, CUDA 12.5 |
| torch | 2.8.0+cu126 |
| Precision | bf16 (`KEV_DTYPE=bf16`); weights occupy 3,143 of 4,096 MiB |
| Kernels | SDPA attention, reference DeltaNet layers. `flash-linear-attention`, `causal-conv1d` and `flash_attn` are **all absent**, so Kev's fused Triton path (`fused`, which requires flash-linear-attention) could not run — Kev's own `kev.serve` enables it on CUDA when available |
| Load | `bench.py`, 3-question records, ~150-token states, 10 unmeasured warmup requests per point |
| Date | 2026-10-07 |

```bash
MODEL_BACKEND=kev MODEL_PATH=jaredpalmer/kev-0.8b DEVICE=cuda KEV_DTYPE=bf16 \
KEV_MAX_STATE=2048 VLLM_API_KEY=local PORT=8001 uv run --no-sync python main.py

uv run --no-sync python bench.py --url http://127.0.0.1:8001 --key local \
  --model kev-0.8b --n 32 --concurrency 8 --state-tokens 150
```

`--no-sync` is required on this machine: `uv sync` reinstalls scikit-learn, whose compiled
`sparsefuncs_fast` is blocked by Windows Smart App Control. transformers guards that import
behind `is_sklearn_available()`, so uninstalling it is the clean fix.

## Throughput vs concurrency

0% repeat — every request a state the server has never seen. Disjoint seeds per point.

| concurrency | p50 ms | p90 ms | p99 ms | req/s |
|---|---|---|---|---|
| 1 | 197.2 | 200.2 | 248.7 | 5.0 |
| 4 | 444.9 | 566.0 | 572.5 | 8.7 |
| 8 | 735.9 | 826.3 | 826.6 | 10.9 |
| 16 | 1116.2 | 1173.1 | 1174.2 | **14.0** |

Throughput grows 2.8× from concurrency 1 to 16 while p50 grows 5.7×, which is what
batching looks like when the GPU is bandwidth-bound: the weight read is paid once per
forward pass however few tokens ride along, so extra rows are nearly free until they
aren't. The knee is between 8 and 16 — +47% latency bought +28% throughput there, against
+66% latency for +25% from 4 to 8.

## State reuse

Concurrency 8. `--repeat-rate` is the fraction of requests that reuse an earlier state with
a **different** question subset — an answer-cache miss that a state cache should still catch.

| repeat | p50 ms | p90 ms | p99 ms | req/s |
|---|---|---|---|---|
| 0% | 828.7 | 1143.2 | 1143.8 | 9.3 |
| 25% | 684.7 | 852.5 | 896.4 | 10.5 |
| 50% | 668.2 | 791.2 | 791.7 | 12.7 |
| 75% | 407.1 | 622.5 | 624.6 | **19.5** |
| 90% | 407.1 | 625.4 | 627.5 | **19.5** |

**+110% throughput and −51% p50** from 0% to 75%. Engine counters over the sweep:
`answer_cache_hits: 52`, `merged_duplicates: 13`, `records: 259`, `rejected: 0`.

Two things worth separating here:

- **This is not the state cache.** `state_cache_bytes: 0` throughout — `KevRuntime.states`
  is `None`. The gain is entirely the answer cache plus in-flight merging, i.e. exact
  repeats, not reuse across different questions. The state cache is still on the table on
  top of this (see [the prefix path](#vllm-jev)).
- **Merging needs concurrency to exist.** 13 merges here against 0 in the concurrency sweep:
  two identical requests can only overlap if two are in flight. At concurrency 1 this lever
  is structurally dead, which is why the same sweep on CPU at concurrency 1 moved p50 only 16%.

75% and 90% are identical to the tenth of a millisecond, so the bottleneck has moved off
the GPU by 75% — above that, added hits land on a server that is no longer waiting on it.

## CPU

Same model, same load, fp32 (Kev's default dtype), concurrency 1 / 4 / 8:

| | p50 ms | req/s |
|---|---|---|
| CPU fp32 | 1391.9 | 0.5 |
| GPU bf16 | 197.2 | 5.0 |

**7× lower latency, 28× more throughput**, and the shape differs more than the magnitude:
CPU throughput was **flat at ~0.5 req/s across concurrency 1, 4 and 8** (p50 1392 → 12270 →
14514 ms — pure queueing, zero throughput gain). A 0.8B model on a CPU is compute-bound, so
batching has nothing to recover; the GPU is bandwidth-bound, so it does. Every batching and
scheduling lever in this engine is a GPU lever, and measuring them on CPU reports zero.

## CUDA graphs: a negative result

`KEV_CUDA_GRAPHS=1`, same points:

| concurrency | baseline req/s | with graphs | |
|---|---|---|---|
| 1 | 5.0 | 5.0 | unchanged (199.2 ms p50) |
| 8 | 10.9 | 0.9 | **12× worse** (8,015 ms p50) |
| 16 | 14.0 | 0.8 | **17× worse** (18,317 ms p50) |

VRAM went 3,143 → 3,945 of 4,096 MiB. ~800 MiB of graph buffers left ~150 MiB, and the
allocator thrashed as soon as anything competed for it. Unchanged at concurrency 1, where
nothing does.

**The cost was unavoidable and the benefit was unreachable.** `m.graphs` is read only inside
`DecisionModel.probs_batch` (`kev/model.py:468-485`); `forward_batch` (`kev/model.py:389`),
which is the path this runtime calls, never touches it. So enabling graphs bought 800 MiB of
buffers that could not be replayed even once. Leave it off until the runtime moves to
`probs_batch` — and even then, only on a card with room.

## vllm-jev

**There is no head-to-head yet**, and the published numbers are not comparable as printed:
vllm-jev reports **Clef-Flash (9B) on an A800-SXM4-80GB at 45 ms p50, 28 req/s at 16
clients**. Different model, different hardware class, and vLLM has no Windows support — WSL2
is the official path, and 4 GB is marginal for vLLM's KV pre-allocation regardless.

What a back-of-envelope normalisation says, with the sum shown:

| axis | A800 : RTX 3050 Laptop | model size | expected |
|---|---|---|---|
| Dense BF16 compute | ~312 : ~18 TFLOPS ≈ 17× | 9B : 0.8B ≈ 11× | us ~1.5× slower |
| Memory bandwidth | ~2,039 : ~192 GB/s ≈ 11× | ~19 : ~1.6 GB ≈ 12× | roughly level |

Measured gap is **4.4×** (197 vs 45 ms), so the residual after normalising is ~3×, and on
the bandwidth axis — the one that governs short prefill-only requests — all 4.4× is residual.
Four known causes, none of them mysterious:

1. **18 of 24 DeltaNet layers on the reference PyTorch path.** `flash-linear-attention` is
   absent, so Kev's fused Triton kernels are unavailable. Kev's own server turns them on.
2. **SDPA instead of FlashAttention** for the 6 full-attention layers.
3. **No CUDA graphs** — and per the section above, not reachable from this code path anyway.
4. **The state is recomputed once per question.** Kev's own `probs()` docstring says
   `forward()` keeps the row form, which re-runs the state per question, and
   `prefix_min_tokens` prices it: **1,011 → 413 ms for 5 questions** on Kev-0.8B bf16.

(4) is the one that is ours rather than a missing wheel, and it is also the fix with the most
behind it. `kev`'s `SCORING_INTERFACE` exposes `probs_and_prefix(enc)`,
`probs_with_prefix(enc, prefix)` and `probs_batch(encs, prefixes, keep)` — a cached state
prefix goes in, a new one comes back out. That is a **cross-request state cache as a
documented API**, and `KevRuntime.prepare()` already computes the `state_key` it would be
stored under. Moving to it fixes the per-question recompute, makes `states` real instead of
`None`, and makes CUDA graphs reachable, in one change.

This corrects an earlier claim in `clef_engine/kev.py` that no such seam existed. It does.

## Correctness

`smoke_test.py --kev` compares this runtime's batched path against Kev's own per-record
`forward()`, then runs the same records through `ClefEngine`.

| | tolerance | result |
|---|---|---|
| CPU fp32 | 2e-3 | **passes** (matches to 4 decimals) |
| GPU bf16 | 1.5e-2 | **passes** |

The bf16 tolerance is sized to the dtype, not loosened to get green. bf16 keeps ~8 mantissa
bits, so a probability near 0.9 carries ~0.003 of absolute precision; the first GPU run
showed 5 deltas of 0.003–0.006 with **the chosen option never changing**, and the same two
values (0.7337 / 0.7292) appeared swapped when compared the other way round — the signature
of reassociation between a batch of four and a batch of one, not a logic error. Two things
keep this honest: `compare()` still checks the chosen option **exactly** at either tolerance,
so a flipped decision fails regardless, and fp32 on CPU still passes at the tight 2e-3.

Oversized states are rejected, not truncated: `encode(strict=True)`, with Kev's
`ContextOverflow` re-raised as `ValueError` so `main.py` returns HTTP 400. Silently dropping
the tail would answer confidently about input the model never read.

## Measurement hygiene

`bench.py --selftest` exists because three of my own benchmark bugs produced numbers I
briefly believed. Each assertion in it corresponds to one:

1. **State ids were not partitioned by sweep point**, so point 0 pre-warmed every later
   point. → cross-point disjointness assertion.
2. **`--seed` reached the reuse pattern but not the state contents**, so a second invocation
   replayed the first one's states straight out of the answer cache. This reported **827
   req/s at concurrency 4 on CPU** — a figure that should have been impossible and was
   caught by `records: 24` against `answer_cache_hits: 48`. → cross-seed disjointness
   assertion.
3. **`make_state`'s prefix contained a space**, so the generated length was off by one word.
   → exact word-count assertion (fixed the generator, not the assert).

If you are reproducing these numbers, run `--selftest` first. An inference benchmark that
accidentally measures its own cache is the easiest wrong answer to get, and it fails in the
flattering direction.

## Not measured

- **Clef**, the model this engine is named for. Needs ~19 GB; `ClefRuntime` has never run on
  real weights, only against the reference on a small random model.
- **Any datacentre GPU.** Every number above is a 4 GB laptop card, which is also what makes
  the CUDA-graphs result a memory-pressure finding rather than a general one.
- **vllm-jev head-to-head.** Needs Linux; a free Colab T4 is the cheapest path.
- **The prefix cache** with a real repeat-rate sweep. The cache is implemented and verified
  correct (same answers, `hits=1 misses=1` on the smoke test), but not yet swept under load.
  The previous repeat-rate sweep used the answer cache; the prefix cache is additive on top
  and should help on same-state/different-question traffic.
- **The new Kev path** (probs_and_prefix). The numbers in the sections above were measured with
  `forward_batch`, which recomputes the state per question. After the switch, re-run with
  `--repeat-rate 0,0.5,0.9` to measure the real prefix-cache gain separately from the
  answer-cache gain.
- **Images and video.** Clef supports them; this API is text and JSON only.
