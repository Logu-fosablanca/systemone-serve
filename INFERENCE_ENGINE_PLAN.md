<div align="center">

# Building an Inference Engine That Beats vLLM

**What a top-tier LLM inference engine is made of, where vLLM falls short, and where this project fits**

Research notes · last updated 2026-10-06

</div>

---

## At a glance

| | |
|---|---|
| **Why vLLM leads** | PagedAttention and continuous batching, plus the widest support for models and hardware |
| **Where it hurts** | Memory fragmentation under long mixed traffic, prefill-heavy workloads, weaker non-NVIDIA backends |
| **Biggest levers** | Smarter prefix reuse, disaggregated prefill and decode, more quantization formats, hardware portability |
| **This project** | A specialised System One engine for Cloudflare Clef first; general LLM serving later |

> **Reading guide.** Sections 1–4 cover inference engines in general. Section 5 is where this repo fits. The detailed Clef plan is in [SYSTEMONE_PERF_PLAN.md](SYSTEMONE_PERF_PLAN.md), and how the code works is in the [README](README.md).

## Contents

1. [Why vLLM leads, and where it breaks](#1-why-vllm-leads-and-where-it-breaks)
2. [Anatomy of an inference engine](#2-anatomy-of-an-inference-engine)
3. [The competitive landscape](#3-the-competitive-landscape)
4. [Roadmap for a general engine](#4-roadmap-for-a-general-engine)
5. [Where this project fits: System One serving](#5-where-this-project-fits-system-one-serving)
6. [Status of this repo](#6-status-of-this-repo)
7. [References](#7-references)

---

## 1. Why vLLM leads, and where it breaks

### What made vLLM the default

| Innovation | What it does | Why it matters |
|---|---|---|
| **PagedAttention** | Stores each request's attention cache (K/V) in fixed blocks of 16 tokens, like pages of virtual memory | Little GPU memory is wasted, so more requests fit at once |
| **Continuous batching** | Adds new requests between decode steps instead of waiting for a whole batch to finish | Roughly 2–3× the throughput of static batching |

### Where it hurts in production

| Problem | What happens |
|---|---|
| Memory fragmentation | Out-of-memory errors after hours of mixed-length traffic, usually fixed by restarting |
| Prefill-heavy traffic | Long prompts (RAG, agents) crowd out decode steps, and throughput drops |
| Quantization formats | GGUF support is experimental; EXL2 isn't supported |
| Non-NVIDIA hardware | AMD, Intel and Apple (vllm-metal) backends exist but trail CUDA |
| Multi-GPU stability | Reports of tensor-parallel ranks falling out of sync under heavy load |
| Public hosting | API-key auth only; no rate limiting or per-user quotas |

The production failures above are from [vLLM in production: five failure patterns](https://perun.au/insights/vllm-production/).

---

## 2. Anatomy of an inference engine

An inference engine can be described as seven layers. vLLM, SGLang and TensorRT-LLM differ in how well they do each one.

```
┌───────────────────────────────────────────────────────────────────────┐
│ 7  Hardware portability   CUDA · ROCm · Apple · Gaudi · CPU           │
│ 6  API and serving        OpenAI API · /v1/systemone · auth · metrics │
│ 5  Quantization           FP8 · FP4 · AWQ/GPTQ · GGUF · EXL2          │
│ 4  Model execution        parallelism · CUDA graphs · speculation     │
│ 3  Scheduler              continuous batching · chunked prefill       │
│ 2  Memory management      paged K/V · prefix cache · eviction         │
│ 1  Kernels                attention · fused ops · matrix multiply     │
└───────────────────────────────────────────────────────────────────────┘
   requests enter at the top · the GPU does the work at the bottom
```

### Layer 1 · Kernels

The hand-tuned GPU code that everything else calls.

| Component | What it does | State of the art |
|---|---|---|
| Attention | Computes attention in tiles in fast on-chip memory, without storing the full attention matrix | FlashAttention 2 and 3; FlashInfer (29–69% lower inter-token latency than a Triton backend, per its paper) |
| Fused operations | Combines small steps (RMSNorm, RoPE, activations) into one kernel to cut memory traffic | Triton, CUTLASS |
| Matrix multiply | The feed-forward layers, where most of the compute goes | cuBLAS, CUTLASS |
| CUDA graphs | Records a step once and replays it without Python overhead | Standard for low-latency decode |

> **Tip.** Build on existing kernel libraries (FlashInfer's templates, flash-linear-attention) instead of writing CUDA from scratch.

### Layer 2 · Memory management

| Component | vLLM today | Better approach |
|---|---|---|
| K/V layout | Paged blocks of 16 tokens | Paging is now standard |
| Prefix reuse | A hash per block; reuses any shared, block-aligned prefix | A radix tree (SGLang) that finds the longest shared prefix across branching conversations |
| Eviction | Least recently used block | Cost-aware: keep what is most expensive to recompute |
| K/V precision | FP16/BF16, or FP8 | FP8 or INT8 fits about 2× more sequences |

> **Biggest single win.** SGLang's RadixAttention showed 37% lower median time-to-first-token than vLLM on prefix-heavy RAG and agent traffic.

### Layer 3 · Scheduler

| Component | What it does |
|---|---|
| Continuous batching | New requests join between steps; the baseline every engine needs |
| Chunked prefill | Splits long prompts into chunks so they don't block other requests |
| Priority queues | First-come, priority, or deadline ordering for different service levels |
| Preemption | Pauses or recomputes a request to free memory for a more important one |
| Disaggregated prefill and decode | Reads prompts and generates tokens on separate GPUs |

> **Why disaggregation matters.** Reading a prompt is limited by compute; generating tokens is limited by memory speed. Mixing them on one GPU is a main cause of throughput drops under mixed traffic. vLLM has experimental support, and it is the frontier for 2026.

### Layer 4 · Model execution

| Component | What it does |
|---|---|
| Tensor parallelism | Splits each layer across GPUs; scales well for prefill but needs fast links such as NVLink |
| Pipeline parallelism | Puts different layers on different GPUs, for very large models |
| Expert parallelism | For mixture-of-experts models, sends each token to the GPUs that hold its experts |
| Compilation and CUDA graphs | Remove Python overhead from every step |
| Speculative decoding | A small draft model proposes tokens and the big model checks them in one pass: 2–3× faster decode with the same output |

### Layer 5 · Quantization

| Format | vLLM today | Notes |
|---|---|---|
| FP8 | Supported | Native on H100 and H200; up to about 2× the compute of BF16 |
| FP4 (NVFP4) | Supported on Blackwell | Native only on Blackwell GPUs (B200, GB200, RTX 50 series) |
| AWQ / GPTQ | Supported | The standard 4-bit weight formats |
| GGUF | Experimental | The main local format, from llama.cpp |
| EXL2 | Not supported | Strong quality per bit on consumer GPUs |
| FP8 K/V cache | Supported | Fits about twice as many sequences |

### Layer 6 · API and serving

| Component | Notes |
|---|---|
| OpenAI-compatible API | Required for adoption |
| `/v1/systemone` | Decision models such as Jev and Clef. Served by llama.cpp, Ollama and vllm-jev, not by upstream vLLM |
| Streaming | Token-by-token output (server-sent events) |
| Structured output | JSON mode and grammar-constrained sampling |
| Multimodal | Image and video inputs |
| Offline batch API | Large dataset jobs |
| Auth and rate limits | vLLM has an API key but no rate limits |
| Metrics | Latency histograms, queue depth, GPU use |

### Layer 7 · Hardware portability

| Platform | Why it matters | Effort |
|---|---|---|
| NVIDIA CUDA | Where vLLM is strongest | Baseline |
| AMD ROCm | Growing data-center share | Medium |
| Apple Silicon (Metal, MLX) | Large developer base | Medium |
| Intel Gaudi | Enterprise buyers | High |
| CPU | llama.cpp's home ground | Low: delegate to llama.cpp |

---

## 3. The competitive landscape

| Engine | Best at | Weak at | Pick it for |
|---|---|---|---|
| **vLLM** | Ecosystem; model and hardware coverage | Fragmentation under long mixed loads; non-NVIDIA backends trail | General-purpose serving |
| **SGLang** | Prefix reuse (RadixAttention); structured generation | Mostly NVIDIA | RAG, agents, shared prompts |
| **TensorRT-LLM** | Raw NVIDIA throughput | Long engine builds (about 28 minutes cold start in one 2026 benchmark); NVIDIA only | Fixed models at maximum speed |
| **llama.cpp** | GGUF; runs almost anywhere, including CPU | Lower throughput under heavy concurrency than GPU-first engines | Local and edge |
| **MLC-LLM** | Cross-platform: Metal, WebGPU, CUDA | Smaller community | Browser and mobile |

---

## 4. Roadmap for a general engine

If this repo grows into general LLM serving, this order moves the needle most.

| # | Component | Impact | Effort | Status here |
|---|---|---|---|---|
| 1 | Prefix reuse | Large for RAG, agents, multi-turn | Medium | Done for Clef states (exact matches) |
| 2 | Disaggregated prefill and decode | Fixes throughput drops on mixed traffic | High | Not started |
| 3 | FlashInfer attention backend | Lower decode latency; a library swap | Low | Not started |
| 4 | FP8 weights and K/V cache | About 2× compute and 2× sequences | Medium | Not started |
| 5 | Speculative decoding | 2–3× on latency-bound decode | Medium | Not relevant to Clef, which never decodes |
| 6 | GGUF and EXL2 | Opens the consumer-GPU market | Medium | Not started |
| 7 | AMD and Apple backends | Large untapped market | High | Not started |
| 8 | Auth and rate limiting | Needed for public hosting | Low | API key done; rate limits not started |

---

## 5. Where this project fits: System One serving

System One models such as TypeSafe's Jev and Cloudflare's Clef don't write text. They read a state and a set of typed questions, then return a probability for every allowed option in a single pass. Most of a general engine (decode scheduling, sampling, speculation) doesn't apply.

```
   state + questions
            │  tokens
            ▼
  ┌────────────────────┐  hidden    ┌─────────────────┐  score per  ┌───────────────┐
  │ backbone           │  states    │ decision head   │  option     │ probabilities │
  │ Qwen3.5 · one pass │──────────► │ from Cloudflare │───────────► │ via softmax   │
  └────────────────────┘            └─────────────────┘             └───────────────┘
```

Several servers already speak `/v1/systemone`: llama.cpp, Ollama, and the vLLM-based vllm-jev. The vLLM-based servers must run with **prefix caching off**: Clef's decision head reads every token's hidden state, and vLLM's prefix cache doesn't keep those. So they recompute every input on every request.

This repo's engine is aimed at that gap:

| | vLLM-based Clef | This engine |
|---|---|---|
| Same state, new questions | Recomputes the whole input | Resumes from the saved state and computes only the questions |
| Exact repeat of a request | Recomputes | Answer cache, no GPU work |
| Identical requests at once | Each one runs | Merged into one |
| Long inputs | Chunked prefill | Chunked prefill, taking turns with short batches |
| Kernels | vLLM's, which are mature | transformers plus flash-linear-attention, not yet measured |

> **Caveat.** Whether this beats vLLM-based Clef depends on how often states repeat in real traffic, and how close our kernels get to vLLM's. The decision rule and benchmarks are in [SYSTEMONE_PERF_PLAN.md](SYSTEMONE_PERF_PLAN.md).

---

## 6. Status of this repo

**Built**

- [x] `/v1/systemone` for Cloudflare Clef (choice, score, noul), with API-key auth and Prometheus metrics
- [x] Batched short path through Cloudflare's own forward pass
- [x] Chunked long path with saved states, giving exact reuse when a state repeats
- [x] Answer cache, and merging of identical in-flight requests
- [x] Scheduler that alternates short batches and long chunks, with admission control (HTTP 429)
- [x] vLLM endpoint plugin that forwards to the engine under one API key
- [x] Parity test against Cloudflare's reference, passing on a small random model

**Next**

- [ ] Run the parity test on the real weights on the H100
- [ ] Benchmark against vllm-jev on our own traffic
- [ ] FP8 weights, CUDA graphs, and batches without padding
- [ ] Reuse for growing transcripts (partial state matches)
- [ ] Images and video

---

## 7. References

**vLLM internals**
- [Inside vLLM: anatomy of a high-throughput inference system](https://www.aleksagordic.com/blog/vllm)
- [vLLM in production: five failure patterns](https://perun.au/insights/vllm-production/)

**Engine comparisons**
- [vLLM vs SGLang 2026: RadixAttention benchmarks](https://www.spheron.network/blog/vllm-vs-sglang-2026/)
- [Best LLM inference engines 2026](https://gigagpu.com/best-llm-inference-engines-2026/)
- [SGLang: the complete guide](https://inference.net/content/sglang-complete-guide/)

**Kernels and optimisation**
- [FlashInfer paper](https://arxiv.org/pdf/2501.01005)
- [LLM inference optimisation, 2026](https://www.morphllm.com/llm-inference-optimization)
- [TD-Pipe: disaggregated pipeline parallelism](https://arxiv.org/pdf/2506.10470)

**System One**
- [Cloudflare: Introducing Clef](https://blog.cloudflare.com/clef-decision-models/)
- [vllm-jev](https://github.com/mode-io/vllm-jev)
- [llama.cpp ships `/v1/systemone`](https://aicoder.com/news/news-20261003-llamacpp-systemone-decision-models)
