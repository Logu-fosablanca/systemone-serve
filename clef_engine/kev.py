"""Runtime for jaredpalmer/kev-* decision models (Qwen3.5 backbone, LoRA, pointer head).

Kev ships readable code, so this uses its own encode(), forward_batch(), to_record() and
to_answers() rather than reimplementing any of them. That matters: encode() packs a state
prefix followed by one branch per question, and for a hybrid backbone forward_batch()
routes to a row-based pass where each branch reads the state's cache independently. A flat
causal pass over the packed sequence would instead let question 2 attend to question 1 and
return confidently wrong probabilities with no error raised.

Batching is real here, not a loop: forward_batch() takes a list of encodings.

This path is the slow one, and knowingly so for now. Two measured reasons to move off it:

1. forward_batch re-runs the state once per question. kev's own probs() docstring says so,
   and prefix_min_tokens puts a number on it: Kev-0.8B bf16, 5 questions, 1011 -> 413 ms.
2. kev's SCORING_INTERFACE exposes probs_and_prefix(enc), probs_with_prefix(enc, prefix)
   and probs_batch(encs, prefixes, keep) -- a cached state prefix goes in, a new one comes
   back out. That is a cross-request state cache as a first-class, documented API, and
   prepare() already computes the state_key it would be stored under.

So the seam this file originally claimed did not exist does exist; switching to the prefix
path should also make KEV_CUDA_GRAPHS useful, since kev wires graphs into probs_batch.

Measured caveat on a 4 GB card: KEV_CUDA_GRAPHS=1 costs ~800 MiB of graph buffers, taking
VRAM to 96% and collapsing throughput under concurrency (10.9 -> 0.9 req/s at 8 concurrent
on an RTX 3050). Harmless at 1 concurrent. Leave it off unless the card has room.
"""

from __future__ import annotations

import hashlib
import logging
from array import array
from dataclasses import dataclass
from typing import Any

import torch

from .runtime import Job

log = logging.getLogger("clef.kev")


@dataclass
class Encoded:
    enc: dict[str, Any]  # kev.model.encode output
    meta: list[dict[str, Any]]  # kev.api.to_record metadata, ordered as the logits are


class KevRuntime:
    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        api: Any,
        encode: Any,
        overflow: type[Exception] = ValueError,
        *,
        max_state: int | None = None,
        max_branch: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.api = api
        self.encode = encode
        self.overflow = overflow
        self.max_state = max_state
        self.max_branch = max_branch
        self.states = None  # see module docstring: no cross-request reuse yet
        self.stats = {"records": 0, "state_tokens_seen": 0, "truncated_states": 0}

    @classmethod
    def load(cls, model_id: str, device: str = "cuda", **kwargs: Any) -> "KevRuntime":
        try:
            from kev import api
            from kev.checkpoint import LoadOptions
            from kev.checkpoint import load as load_checkpoint
            from kev.model import ContextOverflow, encode
        except ModuleNotFoundError as exc:
            # Only claim kev is missing when it actually is. Anything else -- a blocked or
            # broken native extension in its dependency tree -- must surface its own error
            # rather than be reported as an install problem.
            if (exc.name or "").partition(".")[0] != "kev":
                raise
            raise RuntimeError(
                "kev backend needs the kev package, which is not on PyPI: "
                "uv add git+https://github.com/jaredpalmer/kev"
            ) from exc
        # from_env rather than a bare LoadOptions: this is a server entry point, and it
        # gives operators kev's own documented knobs instead of inventing parallel ones --
        # KEV_DTYPE, KEV_ATTN, KEV_CUDA_GRAPHS, KEV_FUSED. The default is fp32, which is
        # the path kev's published numbers use but twice the memory of bf16.
        opts = LoadOptions.from_env()
        tokenizer, model = load_checkpoint(model_id, device, opts)
        temperature = getattr(getattr(model, "head", None), "temperature", None)
        log.info(
            "kev loaded %s on %s (dtype: %s, head temperature: %s)",
            model_id, device, opts.dtype or "fp32 (default)", temperature,
        )
        return cls(tokenizer, model, api, encode, ContextOverflow, **kwargs)

    def _encode(self, request: dict[str, Any]) -> Encoded:
        req = self.api.SystemOneRequest(
            state=request["state"], questions=request["questions"], model="kev"
        )
        record, meta = self.api.to_record(req)
        limits = {k: v for k, v in (("max_state", self.max_state), ("max_branch", self.max_branch)) if v}
        try:
            # strict=True on purpose: silently truncating a state would answer confidently
            # about input the model never saw. Too long is the caller's problem to fix.
            enc = self.encode(self.tokenizer, record, strict=True, **limits)
        except self.overflow as exc:
            raise ValueError(f"input too long for this model: {exc}") from exc
        return Encoded(enc, meta)

    def prepare(self, request: dict[str, Any]) -> Job:
        """Tokenize on the CPU; safe to call off the GPU thread."""
        encoded = self._encode(request)
        enc = encoded.enc
        state_end = enc["state_tokens"]
        if enc.get("state_truncated"):
            self.stats["truncated_states"] += 1
        # Recorded for a future state cache even though nothing reads it yet, so the
        # key is defined by the same bytes the backbone actually sees.
        state_key = hashlib.sha256(array("q", enc["ids"][:state_end]).tobytes()).hexdigest()
        return Job(
            request=request,
            enc=encoded,
            state_end=state_end,
            state_key=state_key,
            long=False,  # forward_batch owns the backbone pass; no chunked path
            cost=len(enc["ids"]),
        )

    @torch.inference_mode()
    def run_short(self, jobs: list[Job]) -> list[dict[str, Any]]:
        batch = self.model.forward_batch([job.enc.enc for job in jobs])
        return [self._result(job, logits) for job, logits in zip(jobs, batch)]

    def _result(self, job: Job, logits: list[torch.Tensor]) -> dict[str, Any]:
        probs = [z.float().softmax(-1).tolist() for z in logits]
        self.stats["records"] += 1
        self.stats["state_tokens_seen"] += job.state_end
        return {
            "answers": self.api.to_answers(probs, job.enc.meta),
            "usage": {"input_tokens": job.cost, "output_tokens": 0},
        }

    def long_step(self, job: Job) -> dict[str, Any] | None:
        raise RuntimeError("kev records have no chunked path; prepare() never sets long")

    def warmup(self) -> None:
        self.run_short([self.prepare({
            "state": "warmup",
            "questions": {"q": {"type": "noul", "instructions": "warmup"}},
        })])
        self.stats = dict.fromkeys(self.stats, 0)
