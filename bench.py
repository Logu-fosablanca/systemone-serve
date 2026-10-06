"""Benchmark any /v1/systemone server: this engine, vllm-jev, or the reference.

Both speak the same API, so the same load generator compares them directly.

    python bench.py --url http://127.0.0.1:8001 --key $VLLM_API_KEY --model clef-flash
    python bench.py --url http://127.0.0.1:8795 --key local --model kev-0.8b
    python bench.py --selftest            # validates load generation and stats, no server

The knob that matters is --repeat-rate: the fraction of requests reusing an earlier state
with fresh questions. A server that recomputes every input is flat across it; one that
caches state should improve as it rises. Sweep it to find where the approaches cross.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from typing import Any

QUESTIONS = {
    "dept": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "charges and refunds", "shipping": "deliveries", "returns": "exchanges"},
    },
    "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
    "mood": {"type": "score", "instructions": "How frustrated is the customer?",
             "criteria": ["Calm", "Annoyed", "Furious"]},
}
# Asking a different subset of the same state is what exercises a cross-request state cache.
VARIANTS = [["dept"], ["urgent"], ["mood"], ["dept", "urgent"], ["dept", "urgent", "mood"]]


def make_state(seed: int, approx_tokens: int) -> str:
    """Deterministic filler of roughly the requested token length (~4 chars/token)."""
    rng = random.Random(seed)
    words = ["order", "refund", "invoice", "delayed", "charged", "twice", "shipment", "support"]
    out, target = [f"ticket-{seed}"], max(1, approx_tokens)  # no space: keeps the word count exact
    while len(out) < target:
        out.append(rng.choice(words))
    return " ".join(out)


def build_load(n: int, repeat_rate: float, state_tokens: int, seed: int = 0,
               namespace: int = 0) -> list[dict[str, Any]]:
    """n requests where ~repeat_rate of them reuse an already-seen state.

    namespace shifts the state ids so separate points in a sweep never share a state.
    Without it the first point warms the server's cache for every later one, and the
    sweep measures nothing.
    """
    rng = random.Random(seed)
    requests: list[dict[str, Any]] = []
    seen: list[int] = []
    base = namespace * (n + 1)
    for i in range(n):
        if seen and rng.random() < repeat_rate:
            state_id = rng.choice(seen)
        else:
            state_id = base + i
            seen.append(state_id)
        qids = VARIANTS[rng.randrange(len(VARIANTS))]
        requests.append({
            "state": make_state(state_id, state_tokens),
            "questions": {q: QUESTIONS[q] for q in qids},
        })
    return requests


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(p: float) -> float:
        # Nearest-rank: no interpolation, so a reported figure is an observed request.
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "p50": at(0.50), "p90": at(0.90), "p99": at(0.99),
        "min": ordered[0], "max": ordered[-1], "mean": statistics.fmean(ordered),
    }


async def run(url: str, key: str, model: str, requests: list[dict[str, Any]],
              concurrency: int) -> tuple[list[float], int, dict[str, int]]:
    import aiohttp

    endpoint = url.rstrip("/") + "/v1/systemone"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    latencies: list[float] = []
    errors: dict[str, int] = {}
    gate = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as session:
        async def one(body: dict[str, Any]) -> None:
            async with gate:
                start = time.perf_counter()
                try:
                    async with session.post(endpoint, json={"model": model, **body}) as resp:
                        await resp.read()
                        if resp.status != 200:
                            errors[f"http_{resp.status}"] = errors.get(f"http_{resp.status}", 0) + 1
                            return
                except Exception as exc:
                    name = type(exc).__name__
                    errors[name] = errors.get(name, 0) + 1
                    return
                latencies.append((time.perf_counter() - start) * 1000)

        wall_start = time.perf_counter()
        await asyncio.gather(*(one(b) for b in requests))
        wall = time.perf_counter() - wall_start

    return latencies, len(requests), errors | {"_wall_ms": int(wall * 1000)}


async def sweep(args: argparse.Namespace) -> int:
    rates = [float(r) for r in args.repeat_rate.split(",")]
    print(f"\n{args.url}  model={args.model}  n={args.n}  concurrency={args.concurrency}  "
          f"state~{args.state_tokens} tokens\n")
    print(f"{'repeat':>7} {'ok':>5} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'req/s':>8}  errors")
    print("-" * 64)
    for point, rate in enumerate(rates):
        # Each point gets its own state namespace so an earlier point cannot pre-warm it.
        requests = build_load(args.n, rate, args.state_tokens, seed=args.seed, namespace=point + 1)
        if args.warmup:
            # Warm the weights and kernels on throwaway states, not on measured ones.
            await run(args.url, args.key, args.model,
                      build_load(args.warmup, 0.0, args.state_tokens, namespace=0), args.concurrency)
        latencies, total, errors = await run(args.url, args.key, args.model, requests, args.concurrency)
        wall_ms = errors.pop("_wall_ms")
        if not latencies:
            print(f"{rate:>7.0%} {0:>5} {'-':>8} {'-':>8} {'-':>8} {'-':>8}  {errors}")
            continue
        p = percentiles(latencies)
        rps = len(latencies) / (wall_ms / 1000) if wall_ms else 0.0
        print(f"{rate:>7.0%} {len(latencies):>5} {p['p50']:>8.1f} {p['p90']:>8.1f} "
              f"{p['p99']:>8.1f} {rps:>8.1f}  {errors or ''}")
    print()
    return 0


def selftest() -> int:
    load = build_load(200, 0.75, 64, seed=1)
    assert len(load) == 200
    distinct = len({r["state"] for r in load})
    # 75% reuse should collapse 200 requests onto far fewer distinct states.
    assert distinct < 80, distinct
    assert all(r["questions"] for r in load)
    assert all(q in QUESTIONS for r in load for q in r["questions"])

    none_repeated = build_load(50, 0.0, 32, seed=1)
    assert len({r["state"] for r in none_repeated}) == 50, "repeat-rate 0 must give unique states"

    # Points in a sweep must not share states, or point 0 pre-warms every later point.
    a = {r["state"] for r in build_load(40, 0.5, 32, seed=1, namespace=1)}
    b = {r["state"] for r in build_load(40, 0.5, 32, seed=1, namespace=2)}
    assert not (a & b), f"{len(a & b)} states leaked between sweep points"

    # make_state is deterministic, so a repeated state_id is byte-identical and cacheable.
    assert make_state(7, 50) == make_state(7, 50)
    assert make_state(7, 50) != make_state(8, 50)
    assert len(make_state(5, 300).split()) == 300

    p = percentiles([float(i) for i in range(1, 101)])
    assert p["p50"] == 51 and p["min"] == 1 and p["max"] == 100, p
    assert percentiles([5.0]) == {"p50": 5.0, "p90": 5.0, "p99": 5.0,
                                 "min": 5.0, "max": 5.0, "mean": 5.0}
    print("selftest ok: load generation, repeat collapsing, determinism, percentiles")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--key", default="local")
    ap.add_argument("--model", default="clef-flash")
    ap.add_argument("--n", type=int, default=200, help="requests per repeat-rate point")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--state-tokens", type=int, default=300, help="approximate state length")
    ap.add_argument("--repeat-rate", default="0.0,0.25,0.5,0.75,0.9",
                    help="comma-separated fractions of requests reusing an earlier state")
    ap.add_argument("--warmup", type=int, default=10, help="unmeasured requests before each point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    return selftest() if args.selftest else asyncio.run(sweep(args))


if __name__ == "__main__":
    sys.exit(main())
