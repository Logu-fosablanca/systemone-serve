# Concepts

The ideas this engine is built on, why each one matters for a decision model
specifically, and what the code does about it. The README explains what the server
does; this explains why it is shaped the way it is.

Each entry says what is **measured**, what is **reasoned**, and what is **unverified**,
because several of these only become real questions on a GPU and no GPU run has happened
yet.

---

## 1. Clef has no decode loop

A chat model writes one token at a time, feeding each back in. N tokens means N forward
passes, and essentially every optimisation in a modern serving engine — continuous
batching, paged attention, speculative decoding — exists to make that loop cheaper.

Clef runs **one** forward pass over the whole input, then a small scoring step reads the
result. Nothing is generated. So most of that machinery has nothing to act on.

**Consequence:** the usual serving playbook does not transfer. The only way to make a
second request cheaper is to not redo work the first one already did.

---

## 2. Hidden states, and why they are the constraint

As tokens pass through the backbone, each one becomes a vector describing what it means
in context — 4,096 numbers for Clef. The last layer's vectors are the **final hidden
states**.

Clef's joint schema head reads the hidden state at **every position**, not just the last
one. vllm-jev describes its own implementation the same way: it reads the head "over
vLLM's per-token hidden states."

This collides with how prefix caching works. A prefix cache earns its speedup by
**skipping the forward pass** over tokens it has already seen. But the forward pass is
what produces the hidden states. Skip the computation and its output does not exist.

vLLM's prefix cache stores attention keys and values, not hidden states — and vLLM's own
tracker carried this as a bug: with prefix caching enabled, extracted hidden states came
back shorter than the token ids, because only freshly scheduled tokens were included.

**What this engine does:** saves the hidden states alongside the K/V and the DeltaNet
recurrent state, so a resumed state still has everything the head needs.

**Not novel.** vLLM-Omni merged hidden-state prefix caching in April 2026 (v0.20.0) and
reported ~8.8x on text prefills. Their cache lives on CPU. The gap this engine fills is
narrower: vllm-jev has not adopted that approach for Clef.

---

## 3. The three regions

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

---

## 4. Three different caches, often confused

| Mechanism | Fires when | Saves |
|---|---|---|
| **Answer cache** | Identical state *and* questions seen before | Everything; no GPU work |
| **In-flight merging** | Identical request already running | One of the two GPU jobs |
| **State cache** | Same state, *different* questions | The state pass only |

Only the third is specific to this design. The first two are generic and any server can
implement them — which matters, because the one benchmark run so far (Kev-0.8B on an RTX
3050, +110% throughput at a 75% repeat rate) measured the first two. The state cache
never engaged in that run: the generated states were shorter than
`CLEF_LONG_STATE_TOKENS`, so every request took the short path.

**Check `state_cache_bytes` in `/health` before attributing any result to the state
cache.**

---

## 5. Chunk invariance

**The question:** does computing a state in chunks give the same answer as computing it
in one pass?

**Why it might not:** 24 of Clef's 32 layers are DeltaNet, which carries a recurrent
state updated once per chunk. The reduction order therefore follows the chunk
boundaries, and floating-point addition is not associative — `(a+b)+c ≠ a+(b+c)` in
finite precision.

**Why it is load-bearing:** if chunk size changed the answer, saving and resuming could
never be made sound. A resume replays a *different* boundary pattern than the request
that wrote the entry, so the two could not agree no matter how the code was tuned. The
whole saved-state design would be unsalvageable.

**Measured:** `check_chunk_invariance` runs the same state at chunk 64, 128, 512 and 8192
(the last exceeding the state, so a single pass) and compares all four against
Cloudflare's reference. On the tiny fp32 CPU model all four are **bit-identical**.

**Status:** the algorithm is chunk-order invariant. Anything seen at bf16 on a GPU would
be rounding rather than structure — a tolerance question, not a correctness one.
**Unverified at bf16.**

---

## 6. Batch invariance

**The question:** does the same record answer the same way regardless of what else
shares its batch?

**Why it might not:** batching changes the GEMM shapes. cuBLAS picks a different kernel
for `[1, seq, hidden]` than for `[8, seq, hidden]` — different tiling, different
reduction order, so different rounding. **Split-K** is the usual culprit: a strategy that
splits the reduction across thread blocks which each compute a partial sum. cuBLAS
chooses it when batches are small and it needs more parallelism, and does not when they
are large. The kernel flips based on how many requests happened to arrive together.

**Why decision models are uniquely exposed:** an LLM picks from a ~150,000-token
vocabulary, where a 1e-3 logit wobble essentially never changes the argmax — and if it
did, the text is robust. A decision model picks from perhaps three options. If the top
two sit at 0.51 and 0.49, that same wobble flips the decision. **The argmax is the entire
output.** There is no downstream context to absorb it.

The failure is **load-dependent**: stable at concurrency 1 in testing, unstable at
concurrency 32 in production. Nothing errors. The answer just changes.

vllm-jev documents this as open: "Kev-0.8B can change its selected label" in concurrent
batches, and as of 2026-10-08, "Score probabilities and near-tied levels can still vary
with batching / cache state." They ship `VLLM_JEV_BF16_MATMUL=no_splitk` as a partial
mitigation and note label changes can remain.

**Measured:** `check_batch_invariance` runs one record at batch 1, 2, 4, 8, 16 and 32 and
fails on any argmax flip regardless of how small the delta is, because a flip is a wrong
answer rather than rounding. On the tiny model: `0.00e+00`, zero flips — **which proves
only that the harness runs.** CPU fp32 is deterministic.

**Mitigations available, cheapest first:**

1. **Fixed batch shapes** — pad to 1/2/4/8/16/32 so cuBLAS only ever sees a small fixed
   set of shapes. Costs some wasted compute, no precision.
2. **`CLEF_FP32_HEAD=1`** — runs the joint head in fp32 while the backbone keeps its own
   precision. The head decides the answer and is small, so fp32 there is cheap. Holds an
   fp32 copy of the output embedding. Does not cover the padded short-batch path, which
   goes through the vendor forward, unless `CLEF_PACKED=1` is also set.
3. Deterministic kernels — heavy-handed, costs throughput.

**Status: unverified at bf16.** This is the measurement that would say whether a
stability guarantee is available here that vllm-jev currently cannot offer.

---

## 7. Exact-match reuse vs prefix reuse

Exact-match reuse fires only when a whole state repeats byte for byte. That misses the
two shapes most likely to occur in practice:

- **A growing transcript.** Turn N's state is turn N−1's plus more. Never an exact match.
- **A fixed preamble.** A RAG request is a constant document plus a varying tail. Never
  an exact match.

**Prefix reuse** matches on the longest shared prefix instead. `prepare()` chains a
SHA-256 per chunk boundary across the state; `StateCache.longest()` walks those
boundaries backwards and resumes from the deepest one held.

Two details that bite:

- Every boundary is a **lookup** point, but only every
  `CLEF_PREFIX_SNAPSHOT_TOKENS`-th is **written back**. See §8.
- `long_step` stops chunks at boundaries, so a save lands on the same grid a later
  request looks up. Without this the grid shifts by `prefix_len` depending on whether
  the startup prefix was available to resume from — which is exactly how the first
  version failed, with zero hits and zero saves.

**Measured:** `check_prefix_reuse` builds a state extending another and confirms the
resume. Resumes at token 548 of 843 — 65% of the state skipped — answers identical to the
reference. On the tiny fp32 model.

---

## 8. The snapshot interval tradeoff

Why not snapshot at every chunk and get maximum hit rate?

Because the **DeltaNet recurrent snapshot is a fixed size per boundary**, independent of
block size. Hidden states cost ~8 KB/token (4,096 dims at bf16) and scale with length.
The recurrent state does not — it is the same size whether the boundary sits at token 256
or token 4096.

Rough figures for 27B, 24 linear-attention layers, recurrent state on the order of 0.5 MB
per layer, so **R ≈ 12–16 MB per snapshot** — an estimate, not measured:

| Spacing | Recurrent overhead | Against ~8 KB/token of hidden state |
|---|---|---|
| 4096 (default) | ~3–4 KB/token | ~40% |
| 1024 | ~14 KB/token | ~175% |
| 256 | ~56 KB/token | ~700% |

Fine granularity is unaffordable in VRAM. That is why the default interval is coarse —
and why vLLM-Omni put their cache on the host.

**`CLEF_STATE_CACHE_OFFLOAD=1`** holds entries in host memory. A state is tens of
megabytes, so a PCIe round trip costs single-digit milliseconds against the hundreds
needed to recompute a long prefill. The budget then applies to RAM rather than VRAM,
which removes the ceiling that forces coarse snapshots. With offload on,
`CLEF_PREFIX_SNAPSHOT_TOKENS=512` becomes reasonable.

---

## 9. The in-place mutation hazard

DeltaNet layers update their cache tensors **in place** rather than returning new ones.
Resume directly from a stored entry and the resume overwrites what was stored.

The failure mode is nasty:

1. **Request 1** computes the state and stores its own copy. Correct, and the copy is clean.
2. **Request 2** takes the stored copy, runs its questions, mutating it. **Still correct** —
   it ran against a clean copy.
3. **Request 3** resumes from the now-polluted copy. **Wrong**, and plausible-looking.

Nothing crashes. The first reuse works. The damage shows up on whichever request touches
that state next.

`StateCache.restore()` owns the deepcopy, so the invariant lives in one place instead of
at every call site.

---

## 10. Padding is safe; packing is not

**Padding** puts filler tokens at the end and masks them. The model is causal and only
looks backwards, so real tokens never see the filler. Safe.

**Packing** concatenates records flat and uses `cu_seqlens` to mark boundaries. It
removes padding waste, but the reference PyTorch DeltaNet path **does not reset its
recurrent state at those boundaries**, so one record's state bleeds into the next.

`CLEF_PACKED=1` is therefore ignored unless `flash-linear-attention` is installed, and
the engine logs why when it refuses.

---

## 11. Why FP8 needs checking twice

Two ways to get FP8 weights, and they need opposite handling:

- **Path A** — the checkpoint already carries `float8_e4m3fn` weights. Nothing to do.
- **Path B** — a BF16 checkpoint, quantised at load with TorchAO.

The trap: HF transformers can silently dequantise an FP8 checkpoint to BF16 while
loading. Then you believe you are on Path A, you are actually on Path B, and applying
TorchAO to already-FP8 weights would double-quantise.

`_is_fp8_checkpoint()` samples an actual weight dtype rather than trusting the checkpoint
name, and the code picks the path from what it finds.

Separately: FP8 *execution* needs SM89+ (Ada) or SM90 (Hopper). Ampere cards — A6000,
A40 — have no FP8 tensor cores, so an FP8 checkpoint there will not run faster and may
not run at all.

---

## What is actually verified

| Concept | Verified on | Open |
|---|---|---|
| Chunk invariance | tiny model, CPU fp32 — bit-identical | bf16 on GPU |
| Batch invariance | tiny model, CPU fp32 — harness only | bf16 on GPU |
| Prefix reuse | tiny model, CPU fp32 — 65% skipped, answers match | bf16 on GPU |
| Host offload | tiny model — stored on CPU, restored, reused | throughput cost |
| Short/long/state-reuse parity | tiny model vs vendor reference | real Clef weights |
| Kev backend | **real weights**, CPU fp32 and GPU bf16 | — |
| FP8, torch.compile, packed batching | coded only | everything |

One command on a GPU closes most of this column.
