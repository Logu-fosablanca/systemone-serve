"""Encoder (BERT-family) runtime for System One decision models, e.g. Laya.

Bidirectional attention means every token's representation depends on every other token
in the input. There is no causal prefix whose hidden states stay valid once the questions
change, so the decoder path's state cache and chunked prefill do not apply: a saved state
would simply be wrong. Every request is one full forward pass.

That is fine here. These models are small — Laya is 644 MB against clef-flash's 19 GB — so
a pass costs milliseconds and there is no prefill worth skipping.

The scheduler is shared with the decoder path unchanged: answer cache, merging of identical
in-flight requests, admission control and token-budget batching all still apply.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import torch

from .runtime import Job

log = logging.getLogger("clef.encoder")


class EncoderRuntime:
    def __init__(self, agent: Any, *, max_length: int = 1024, model_id: str = "") -> None:
        self.agent = agent
        self.max_length = max_length
        self.model_id = model_id
        self.states = None  # bidirectional: nothing survives a change of questions
        self.stats = {"records": 0}

    @classmethod
    def load(cls, model_id: str, device: str = "cuda", **kwargs: Any) -> "EncoderRuntime":
        try:
            import laya
        except ImportError as exc:
            raise RuntimeError("encoder backend needs the laya package: uv add laya") from exc
        # laya.load owns device placement; use CUDA_VISIBLE_DEVICES to pin a GPU.
        agent = laya.load(model_id)
        log.info("encoder model loaded: %s (requested device: %s)", model_id, device)
        return cls(agent, model_id=model_id, **kwargs)

    def _tokens(self, request: dict[str, Any]) -> int:
        """Token count for admission control and batch sizing — never for correctness."""
        text = json.dumps(request, ensure_ascii=False)
        tokenizer = getattr(self.agent, "tokenizer", None)
        if tokenizer is not None:
            return len(tokenizer.encode(text))
        # ponytail: byte heuristic, since scripts differ wildly in bytes per token.
        # Swap for the real tokenizer once laya exposes one.
        return max(1, len(text.encode()) // 4)

    def prepare(self, request: dict[str, Any]) -> Job:
        """Cost estimation only; runs on the encode thread. No tokenization to reuse later."""
        cost = min(self._tokens(request), self.max_length)
        return Job(
            request=request,
            enc=SimpleNamespace(input_ids=()),
            state_end=0,
            state_key="",
            long=False,  # no chunked path: the model reads the whole input at once
            cost=cost,
        )

    @torch.inference_mode()
    def run_short(self, jobs: list[Job]) -> list[dict[str, Any]]:
        # ponytail: one predict() per record, because laya's public API is single-record.
        # Batching is the upgrade if a profile shows the GPU idling between calls.
        return [self._predict(job) for job in jobs]

    def _predict(self, job: Job) -> dict[str, Any]:
        out = self.agent.predict(
            job.request["state"], job.request["questions"], max_len=self.max_length
        )
        self.stats["records"] += 1
        usage = out.get("usage") or {"input_tokens": job.cost, "output_tokens": 0}
        return {"answers": out["answers"], "usage": usage}

    def long_step(self, job: Job) -> dict[str, Any] | None:
        raise RuntimeError("encoder models have no chunked path; prepare() never sets long")

    def warmup(self) -> None:
        self.run_short([self.prepare({
            "state": "warmup",
            "questions": {"q": {"type": "noul", "instructions": "warmup"}},
        })])
        self.stats = dict.fromkeys(self.stats, 0)
