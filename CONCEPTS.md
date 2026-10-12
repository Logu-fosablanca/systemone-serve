# Concepts

The ideas this engine is built on, why each one matters, and what the code does about it.
The README explains what the server does; this explains why it is shaped the way it is.

**Part I** is general inference — true of any serving engine, and the vocabulary the rest
of the repo assumes. **Part II** is what changes when the model makes decisions instead
of writing text, which is where this project's design actually comes from.

Entries in Part II say what is **measured**, what is **reasoned**, and what is
**unverified**, because several only become real questions on a GPU and no GPU run has
happened yet.

---

# Part I — Inference fundamentals

## 1. Tokens, prefill and decode

Models read integers. A tokenizer splits text into pieces and maps each to an id; one
token is roughly three quarters of an English word. Every cost in this codebase is
counted in tokens.

Two phases:

- **Prefill** — one forward pass over all input tokens at once.
- **Decode** — generating output one token at a time, each pass reading one new token
  plus everything cached from before.

A chat model prefills once, then decodes N times for N output tokens. Almost every
famous serving optimisation targets decode, because that is where the time goes.

## 2. The KV cache

Attention at position *i* needs the keys and values of every position before it.
Recomputing them at each decode step would be quadratic, so they are kept. The cache
grows linearly with sequence length and is normally the memory bottleneck in LLM serving
— which is why vLLM's paged attention exists, managing it in fixed blocks like virtual
memory pages.

## 3. Hybrid attention: full versus linear

Clef 27B has 32 layers: **24 DeltaNet** (linear attention) and **8 full attention**.

| | Full attention | DeltaNet / linear attention |
|---|---|---|
| State per layer | Per-token K/V, **grows** with length | A **fixed-size** recurrent summary |
| Cost at long context | Quadratic in attention | Linear |
| Exactness | Exact | Approximate |

The fixed-size part is what makes very long context affordable. It is also what drives
the snapshot arithmetic in §19: a recurrent summary costs the same whether the boundary
sits at token 256 or token 4096.

## 4. Memory-bound versus compute-bound, and why batching works

Every forward pass reads **all** the model's weights out of HBM, however few tokens it
processes. Approximate floors on an H100 at ~3 TB/s:

| Weights | Bytes | Pure weight-read time |
|---|---|---|
| Clef-flash 9B, BF16 | ~19 GB | ~6 ms |
| Clef 27B, BF16 | ~54 GB | ~18 ms |
| Clef 27B, FP8 | ~27 GB | ~9 ms |

A short request does not supply enough arithmetic to hide that read, so the GPU sits
idle waiting on memory — **memory-bound**. Run eight short requests together and you read
the weights **once** for all eight. That is the entire economics of batching.

Decode is extremely memory-bound: one token of work per full weight read. Prefill over
many tokens is compute-bound. Clef is prefill-only, which puts it on the better side of
this — but short states still waste most of the weight read, which is why batching still
matters here.

## 5. Batching: padded, packed, continuous

- **Padded** — pad every row to the longest, mask the filler. Simple; wastes compute in
  proportion to how much the lengths vary.
- **Packed** — concatenate all tokens flat and mark boundaries with `cu_seqlens`. No
  waste, but every kernel in the stack has to respect the boundaries (see §21).
- **Continuous** — add and remove requests between decode steps so the GPU never idles.
  The headline vLLM feature, and meaningless without a decode loop.

## 6. Kernel launch overhead and fused kernels

Every GPU operation is a launch with a fixed CPU-side cost, on the order of microseconds.
A layer assembled from many small operations can spend longer launching than computing
when the tensors are small. Fusing those operations into one kernel removes that floor.

Concretely here: the reference PyTorch DeltaNet path is roughly 20 launches per layer;
`flash-linear-attention`'s fused Triton kernel is 1. With 24 DeltaNet layers that is the
difference between ~480 launches and 24.

## 7. CUDA graphs

Record a sequence of kernel launches once, then replay the whole thing as a single unit,
paying the CPU-side cost once instead of per launch.

The cost is memory: capture buffers are allocated per graph and per input shape. On a
small card that allocation can starve the model itself — measured in this repo at
**12–17x worse** under concurrency on a 4 GB card, which is why the feature was cut.

## 8. torch.compile

Traces the model into a graph, fuses operations and generates kernels ahead of time.
`mode="reduce-overhead"` turns on CUDA graph trees internally, so it subsumes §7.
`dynamic=True` avoids recompiling for every new input shape. The cost is warmup: the
first pass through each shape pays for tracing and compilation, which is why the engine
warms both the short and long paths before serving.

## 9. Quantisation

Storing weights in fewer bits. FP8 halves the bytes against BF16; INT4 and NVFP4 quarter
them. Two separate wins: **capacity** (a bigger model fits) and **bandwidth** (the
weight-read floor in §4 halves with the bytes).

`W8A8` quantises weights and activations; `W4A16` only weights. Quantisation needs
hardware support — FP8 *execution* requires SM89+ (Ada) or SM90 (Hopper). Ampere cards
such as the A6000 and A40 have no FP8 tensor cores, so an FP8 checkpoint there buys
memory and nothing else, and may not run at all.

## 10. Scheduling, head-of-line blocking, admission control

**FCFS** serves in arrival order, so one long request blocks every short request behind
it — **head-of-line blocking**. **Shortest-first** fixes that but can starve long
requests forever, so **aging** grants priority in proportion to time waited.

**Admission control** is the other half. When queued work exceeds a budget, reject
immediately with HTTP 429 rather than accepting the request and making everyone wait.
Failing fast is kinder than an unbounded queue, and it gives the caller something to act
on.

## 11. Reading benchmarks

Throughput (req/s) and latency (ms) trade against each other, and **neither means
anything without stating concurrency**. p50 is the typical request; p95 and p99 are what
people actually complain about. Nearest-rank percentiles report a request that really
happened; interpolated ones report a number nobody experienced.

---

# Part II — What changes for a decision model

## 12. Clef has no decode loop

Clef runs **one** forward pass over the whole input, then a small scoring step reads the
result. Nothing is generated.

**Consequence:** most of §5's continuous batching, §2's paged attention and speculative
decoding have nothing to act on. The only way to make a second request cheaper is to not
redo work the first one already did.

## 13. Hidden states, and why they are the constraint

As tokens pass through the backbone, each becomes a vector describing what it means in
context — 4,096 numbers for Clef. The last layer's vectors are the **final hidden
states**.

Clef's joint schema head reads the hidden state at **every position**, not just the last
one. vllm-jev describes its own implementation the same way: it reads the head "over
vLLM's per-token hidden states."

This collides with prefix caching. A prefix cache earns its speedup by **skipping the
forward pass** over tokens it has already seen — but the forward pass is what produces
the hidden states. Skip the computation and its output does not exist.

vLLM's prefix cache stores keys and values, not hidden states, and vLLM's own tracker
carried this as a bug: with prefix caching on, extracted hidden states came back shorter
than the token ids, because only freshly scheduled tokens were included.

**What this engine does:** saves the hidden states alongside the K/V and the DeltaNet
recurrent state, so a resumed state still has everything the head needs.

**Not novel.** vLLM-Omni merged hidden-state prefix caching in April 2026 (v0.20.0) and
reported ~8.8x on text prefills, with the cache on CPU. The gap this engine fills is
narrower: vllm-jev has not adopted that approach for Clef.

## 14. The three regions

```
[ system prompt ][ state .............. ][ questions + options ][ close ]
    ~36 tokens      any length              varies                fixed
```

| Region | Cost |
|---|---|
| System prompt | Computed once at warmup (`_compute_startup_prefix`), effectively free |
| State | Cacheable — the whole point |
| Questions | Recomputed every request |

Because the backbone is **causal** — each token sees only tokens before it — the state's
hidden states do not depend on the questions that follow. That is the entire reason a
state can be computed once and reused with different questions.

**Placement rule:** anything that varies per request must go in `questions`, never in
`state`. The cache keys on a hash of the state's token ids, so a timestamp or request id
slipped into the state JSON makes every request unique and the cache silently never
fires.

## 15. Three different caches, often confused

| Mechanism | Fires when | Saves |
|---|---|---|
| **Answer cache** | Identical state *and* questions seen before | Everything; no GPU work |
| **In-flight merging** | Identical request already running | One of the two GPU jobs |
| **State cache** | Same state, *different* questions | The state pass only |

Only the third is specific to this design. The first two are generic — which matters,
because the one benchmark run so far (Kev-0.8B on an RTX 3050, +110% throughput at a 75%
repeat rate) measured the first two. The state cache never engaged: the generated states
were shorter than `CLEF_LONG_STATE_TOKENS`, so every request took the short path.

**Check `state_cache_bytes` in `/health` before attributing any result to the state
cache.**

## 16. Chunk invariance

**The question:** does computing a state in chunks give the same answer as computing it
in one pass?

**Why it might not:** DeltaNet (§3) carries a recurrent state updated once per chunk, so
the reduction order follows the chunk boundaries — and floating-point addition is not
associative: `(a+b)+c ≠ a+(b+c)` in finite precision.

**Why it is load-bearing:** if chunk size changed the answer, saving and resuming could
never be sound. A resume replays a *different* boundary pattern than the request that
wrote the entry, so the two could not be made to agree by any amount of tuning. The whole
saved-state design would be unsalvageable.

**Measured:** `check_chunk_invariance` runs the same state at chunk 64, 128, 512 and 8192
(the last exceeding the state, so a single pass) and compares all four against
Cloudflare's reference. On the tiny fp32 CPU model all four are **bit-identical**.

**Status:** the algorithm is chunk-order invariant. Anything seen at bf16 would be
rounding rather than structure — a tolerance question, not a correctness one.
**Unverified at bf16.**

## 17. Batch invariance

**The question:** does the same record answer the same way regardless of what else shares
its batch?

**Why it might not:** batching changes the GEMM shapes, so cuBLAS picks a different
kernel for `[1, seq, hidden]` than for `[8, seq, hidden]` — different tiling, different
reduction order, different rounding. **Split-K** is the usual culprit: a strategy that
splits the reduction across thread blocks which each compute a partial sum. cuBLAS
chooses it when batches are small and it needs parallelism, and does not when they are
large. The kernel flips based on how many requests happened to arrive together.

**Why decision models are uniquely exposed:** an LLM picks from a ~150,000-token
vocabulary, where a 1e-3 logit wobble essentially never changes the argmax — and if it
did, the text is robust. A decision model picks from perhaps three options. If the top
two sit at 0.51 and 0.49, that same wobble flips the decision. **The argmax is the entire
output**, with no downstream context to absorb it.

The failure is **load-dependent**: stable at concurrency 1 in testing, unstable at
concurrency 32 in production. Nothing errors. The answer just changes.

vllm-jev documents this as open: "Kev-0.8B can change its selected label" in concurrent
batches, and as of 2026-10-08, "Score probabilities and near-tied levels can still vary
with batching / cache state." They ship `VLLM_JEV_BF16_MATMUL=no_splitk` as a partial
mitigation and note label changes can remain.

**Measured:** `check_batch_invariance` runs one record at batch 1, 2, 4, 8, 16 and 32 and
fails on any argmax flip regardless of the delta, because a flip is a wrong answer rather
than rounding. On the tiny model: `0.00e+00`, zero flips — **which proves only that the
harness runs.** CPU fp32 is deterministic.

**Mitigations, cheapest first:**

1. **Fixed batch shapes** — pad to 1/2/4/8/16/32 so cuBLAS only ever sees a small fixed
   set of shapes. Costs some wasted compute, no precision.
2. **`CLEF_FP32_HEAD=1`** — runs the joint head in fp32 while the backbone keeps its own
   precision. The head decides the answer and is small, so fp32 there is cheap. Holds an
   fp32 copy of the output embedding. Does not cover the padded short-batch path, which
   goes through the vendor forward, unless `CLEF_PACKED=1` is also set.
3. Deterministic kernels — heavy-handed, costs throughput.

**Status: unverified at bf16.** This is the measurement that would say whether a
stability guarantee is available here that vllm-jev currently cannot offer.

## 18. Exact-match reuse versus prefix reuse

Exact-match reuse fires only when a whole state repeats byte for byte, which misses the
two shapes most likely to occur:

- **A growing transcript.** Turn N's state is turn N−1's plus more. Never an exact match.
- **A fixed preamble.** A RAG request is a constant document plus a varying tail. Never an
  exact match.

**Prefix reuse** matches on the longest shared prefix instead. `prepare()` chains a
SHA-256 per chunk boundary across the state; `StateCache.longest()` walks those
boundaries backwards and resumes from the deepest one held.

Two details that bite:

- Every boundary is a **lookup** point, but only every
  `CLEF_PREFIX_SNAPSHOT_TOKENS`-th is **written back**. See §19.
- `long_step` stops chunks at boundaries, so a save lands on the same grid a later request
  looks up. Without this the grid shifts by `prefix_len` depending on whether the startup
  prefix was available to resume from — which is exactly how the first version failed,
  with zero hits and zero saves.

**Measured:** resumes at token 548 of 843 — 65% of the state skipped — answers identical
to the reference, on the tiny fp32 model.

## 19. The snapshot interval tradeoff

Why not snapshot at every chunk and maximise the hit rate?

Because the **DeltaNet recurrent snapshot is a fixed size per boundary** (§3), regardless
of block size. Hidden states cost ~8 KB/token (4,096 dims at bf16) and scale with length.
The recurrent state does not — it is the same size whether the boundary sits at token 256
or 4096.

Rough figures for 27B, 24 linear-attention layers, recurrent state on the order of 0.5 MB
per layer, so **R ≈ 12–16 MB per snapshot** — an estimate, not measured:

| Spacing | Recurrent overhead | Against ~8 KB/token of hidden state |
|---|---|---|
| 4096 (default) | ~3–4 KB/token | ~40% |
| 1024 | ~14 KB/token | ~175% |
| 256 | ~56 KB/token | ~700% |

Fine granularity is unaffordable in VRAM. That is why the default interval is coarse, and
why vLLM-Omni put their cache on the host.

**`CLEF_STATE_CACHE_OFFLOAD=1`** holds entries in host memory. A state is tens of
megabytes, so a PCIe round trip costs single-digit milliseconds against the hundreds
needed to recompute a long prefill. The budget then applies to RAM rather than VRAM,
removing the ceiling that forces coarse snapshots. With offload on,
`CLEF_PREFIX_SNAPSHOT_TOKENS=512` becomes reasonable.

## 20. The in-place mutation hazard

DeltaNet layers update their cache tensors **in place** rather than returning new ones.
Resume directly from a stored entry and the resume overwrites what was stored.

The failure mode is nasty:

1. **Request 1** computes the state and stores its own copy. Correct, and the copy is clean.
2. **Request 2** takes the stored copy and runs its questions, mutating it. **Still
   correct** — it ran against a clean copy.
3. **Request 3** resumes from the now-polluted copy. **Wrong**, and plausible-looking.

Nothing crashes. The first reuse works. The damage shows up on whichever request touches
that state next. `StateCache.restore()` owns the deepcopy so the invariant lives in one
place instead of at every call site.

## 21. Padding is safe; packing is not

**Padding** puts filler at the end and masks it. The model is causal and only looks
backwards, so real tokens never see the filler. Safe.

**Packing** concatenates records flat and marks boundaries with `cu_seqlens`. It removes
the waste in §5, but the reference PyTorch DeltaNet path **does not reset its recurrent
state at those boundaries**, so one record's state bleeds into the next.

`CLEF_PACKED=1` is therefore ignored unless `flash-linear-attention` is installed, and the
engine logs why when it refuses.

## 22. Why FP8 needs checking twice

Two ways to get FP8 weights, needing opposite handling:

- **Path A** — the checkpoint already carries `float8_e4m3fn` weights. Nothing to do.
- **Path B** — a BF16 checkpoint, quantised at load with TorchAO.

The trap: HF transformers can silently dequantise an FP8 checkpoint to BF16 while loading.
You then believe you are on Path A, you are actually on Path B, and applying TorchAO to
already-FP8 weights would double-quantise.

`_is_fp8_checkpoint()` samples an actual weight dtype rather than trusting the checkpoint
name, and the code picks the path from what it finds. See also §9 on why FP8 execution
needs SM89+.

---

## What is actually verified

| Concept | Verified on | Open |
|---|---|---|
| Chunk invariance (§16) | tiny model, CPU fp32 — bit-identical | bf16 on GPU |
| Batch invariance (§17) | tiny model, CPU fp32 — harness only | bf16 on GPU |
| Prefix reuse (§18) | tiny model, CPU fp32 — 65% skipped, answers match | bf16 on GPU |
| Host offload (§19) | tiny model — stored on CPU, restored, reused | throughput cost |
| Short/long/state-reuse parity | tiny model vs vendor reference | real Clef weights |
| Kev backend | **real weights**, CPU fp32 and GPU bf16 | — |
| FP8, torch.compile, packed batching | coded only | everything |

One command on a GPU closes most of this column.
