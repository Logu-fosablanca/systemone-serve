"""Check the Clef engine against Cloudflare's reference systemone() on the same records.

    python smoke_test.py          # real model (SYSTEMONE_MODEL, default Cloudflare/clef-flash) on the GPU
    python smoke_test.py --tiny   # small random model on the CPU: checks engine logic, not weights

Covers the batched short path, the chunked long path, a saved state reused with new
questions, merging of duplicate in-flight requests, and the answer cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from clef_engine import ClefEngine, ClefRuntime
from clef_engine.runtime import load_reference

MODEL = os.getenv("SYSTEMONE_MODEL", "Cloudflare/clef-flash")

DEPT = {
    "type": "choice",
    "instructions": "Which department should handle this?",
    "criteria": {"shipping": "delivery issues", "billing": "payment issues", "returns": "return requests"},
}
URGENT = {"type": "noul", "instructions": "Does this need urgent human attention?"}
MOOD = {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Annoyed", "Furious"]}
LOG = {"events": [{"t": i, "msg": f"user opened order {1000 + i} and reported a delay"} for i in range(40)]}

RECORDS = [
    {"state": "My order #1234 hasn't arrived in two weeks and I'm very upset.", "questions": {"dept": DEPT, "urgent": URGENT, "mood": MOOD}},
    {"state": "Thanks, the invoice looks correct.", "questions": {"dept": DEPT, "urgent": URGENT}},
    {"state": LOG, "questions": {"urgent": URGENT, "mood": MOOD}},
    {"state": LOG, "questions": {"dept": DEPT}},  # same state, new questions: must reuse the saved state
]


def build_tiny() -> tuple:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForConditionalGeneration

    path = Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.py", "*.jinja"]))
    jsm = load_reference(path)
    cfg = json.loads((path / "config.json").read_text())
    cfg["text_config"].update(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
    )
    cfg["text_config"]["rope_parameters"]["mrope_section"] = [3, 3, 2]
    cfg["vision_config"].update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    torch.manual_seed(0)
    backbone = Qwen3_5ForConditionalGeneration(Qwen3_5Config.from_dict(cfg)).float().eval()
    head = jsm.JointSchemaHead(hidden_size=64, width=32, routing_layers=1, layers=1, heads=2, feedforward=64).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    return jsm.ClefModel(backbone, head).eval(), SimpleNamespace(tokenizer=tokenizer), jsm


def compare(name: str, ours: dict, ref: dict, tol: float) -> int:
    bad = 0
    for qid, r in ref.items():
        o = ours[qid]
        if r["type"] == "choice":
            ok = o["choice"] == r["choice"] and all(abs(o["probabilities"][k] - v) <= tol for k, v in r["probabilities"].items())
        elif r["type"] == "score":
            ok = abs(o["score"] - r["score"]) <= tol * len(r["legend"])
        else:
            ok = abs(o["noul"] - r["noul"]) <= tol
        print(f"{name:12s} {qid:7s} {'ok' if ok else 'MISMATCH'}  ours={o}  ref={r}")
        bad += not ok
    return bad


async def check_engine(rt: ClefRuntime, ref: dict, tol: float) -> int:
    engine = ClefEngine(rt)
    engine.start()
    outs = await asyncio.gather(*(engine.submit(RECORDS[0]) for _ in range(3)))
    outs.append(await engine.submit(RECORDS[0]))
    stats = engine.stats_snapshot()
    assert stats["merged_duplicates"] == 2 and stats["answer_cache_hits"] == 1, stats
    return sum(compare("engine", out["answers"], ref, tol) for out in outs)


class FakePackagedRuntime:
    """Stands in for a vendor package so the packaged path is testable without installing one."""

    def __init__(self) -> None:
        self.calls = 0

    def _call(self, state, questions):
        self.calls += 1
        return {"answers": {qid: {"type": "noul", "noul": 0.5} for qid in questions}}


async def check_packaged() -> int:
    from clef_engine import PackageRuntime

    rt = PackageRuntime(agent=None, max_length=256)
    agent = FakePackagedRuntime()
    rt._call = agent._call  # the one method every vendor subclass overrides
    job = rt.prepare(RECORDS[0])
    assert not job.long, "packaged records must never take the chunked path"
    assert job.cost > 0 and rt.states is None, (job.cost, rt.states)

    engine = ClefEngine(rt)
    engine.start()
    outs = await asyncio.gather(*(engine.submit(RECORDS[1]) for _ in range(3)))
    outs.append(await engine.submit(RECORDS[1]))
    stats = engine.stats_snapshot()
    # Four identical requests: three merged in flight, one served from the answer cache,
    # so the model must have been called exactly once.
    assert agent.calls == 1, agent.calls
    assert stats["merged_duplicates"] == 2 and stats["answer_cache_hits"] == 1, stats
    assert all(o["answers"].keys() == RECORDS[1]["questions"].keys() for o in outs)
    print(f"packaged     shared scheduler ok  model_calls={agent.calls}  stats={stats}")
    return 0


def main() -> int:
    if "--packaged" in sys.argv:
        return asyncio.run(check_packaged())
    tiny = "--tiny" in sys.argv
    settings = {"chunk_tokens": 128, "long_state_tokens": 64}  # force several chunks on the LOG state
    if tiny:
        rt = ClefRuntime(*build_tiny(), **settings)
    else:
        rt = ClefRuntime.load(MODEL, os.getenv("SYSTEMONE_DEVICE", "cuda"), revision=os.getenv("CLEF_REVISION") or None, **settings)
    tol = 1e-3 if tiny else 0.02  # fp32 on CPU vs bf16 on GPU
    refs = [rt.jsm.systemone(rt.model, rt.processor, {"model": "ref", **r})["answers"] for r in RECORDS]

    bad = 0
    jobs = [rt.prepare(r) for r in RECORDS[:2]]
    assert not any(job.long for job in jobs)
    for job, out, ref in zip(jobs, rt.run_short(jobs), refs):
        bad += compare("short-batch", out["answers"], ref, tol)

    for i, name in ((2, "long"), (3, "long-reused")):
        job = rt.prepare(RECORDS[i])
        assert job.long and job.state_end > rt.prefix_len + settings["chunk_tokens"], (job.state_end, rt.prefix_len)
        while (out := rt.long_step(job)) is None:
            pass
        bad += compare(name, out["answers"], refs[i], tol)
    assert rt.stats["state_hits"] == 1 and rt.stats["state_misses"] == 1, rt.stats

    bad += asyncio.run(check_engine(rt, refs[0], tol))
    print(f"prefix tokens: {rt.prefix_len}, stats: {rt.stats}")
    print("PASS" if not bad else f"FAIL: {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
