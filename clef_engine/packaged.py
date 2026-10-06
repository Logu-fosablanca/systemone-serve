"""Runtimes for decision models that ship their own package with an opaque predict().

Laya, Strands and friends hide the backbone behind a module-level load() and an
agent.predict(). Without a way to call model(input_ids, past_key_values=...) there is no
cache to save and no way to run prefill in chunks, so the decoder path's state reuse
cannot apply: every request is one call into the package.

For Laya there is a second, independent reason — its mmBERT backbone is bidirectional, so
a token's representation depends on which questions were asked and no saved prefix would
stay valid anyway.

What does still apply is the whole scheduler: answer cache, merging of identical in-flight
requests, admission control and token-budget batching. That is most of the production
value, and adding a model costs one subclass with two methods.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import torch

from .runtime import Job

log = logging.getLogger("clef.packaged")


class PackageRuntime:
    package = ""  # pip name, shown in the install hint when the import fails

    def __init__(self, agent: Any, *, max_length: int = 1024, model_id: str = "") -> None:
        self.agent = agent
        self.max_length = max_length
        self.model_id = model_id
        self.states = None  # no reusable cache: the package owns the backbone
        self.stats = {"records": 0}

    @classmethod
    def load(cls, model_id: str, device: str = "cuda", **kwargs: Any) -> "PackageRuntime":
        try:
            agent = cls._load_agent(model_id, device)
        except ImportError as exc:
            raise RuntimeError(f"{cls.__name__} needs the {cls.package} package: uv add {cls.package}") from exc
        log.info("%s loaded %s (requested device: %s)", cls.__name__, model_id, device)
        return cls(agent, model_id=model_id, **kwargs)

    @staticmethod
    def _load_agent(model_id: str, device: str) -> Any:
        """Import the package and return its loaded agent. Packages own device placement."""
        raise NotImplementedError

    def _call(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Run one record through the package and return its /v1/systemone response body."""
        raise NotImplementedError

    def _tokens(self, request: dict[str, Any]) -> int:
        """Token count for admission control and batch sizing — never for correctness."""
        text = json.dumps(request, ensure_ascii=False)
        tokenizer = getattr(self.agent, "tokenizer", None)
        if tokenizer is not None:
            return len(tokenizer.encode(text))
        # ponytail: byte heuristic, since scripts differ wildly in bytes per token.
        # Swap for the real tokenizer on any package that exposes one.
        return max(1, len(text.encode()) // 4)

    def prepare(self, request: dict[str, Any]) -> Job:
        """Cost estimation only; runs on the encode thread. Nothing here is reused later."""
        return Job(
            request=request,
            enc=SimpleNamespace(input_ids=()),
            state_end=0,
            state_key="",
            long=False,  # no chunked path: the package reads the whole input at once
            cost=min(self._tokens(request), self.max_length),
        )

    @torch.inference_mode()
    def run_short(self, jobs: list[Job]) -> list[dict[str, Any]]:
        # ponytail: one call per record, because these packages expose no batch API.
        # Batch properly if a profile shows the GPU idling between calls.
        return [self._predict(job) for job in jobs]

    def _predict(self, job: Job) -> dict[str, Any]:
        out = self._call(job.request["state"], job.request["questions"])
        self.stats["records"] += 1
        return {
            "answers": out["answers"],
            "usage": out.get("usage") or {"input_tokens": job.cost, "output_tokens": 0},
        }

    def long_step(self, job: Job) -> dict[str, Any] | None:
        raise RuntimeError("packaged models have no chunked path; prepare() never sets long")

    def warmup(self) -> None:
        self.run_short([self.prepare({
            "state": "warmup",
            "questions": {"q": {"type": "noul", "instructions": "warmup"}},
        })])
        self.stats = dict.fromkeys(self.stats, 0)


class LayaRuntime(PackageRuntime):
    """convaiinnovations/laya-multilingual and other mmBERT-backbone decision models."""

    package = "laya"

    @staticmethod
    def _load_agent(model_id: str, device: str) -> Any:
        import laya

        return laya.load(model_id)

    def _call(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return self.agent.predict(state, questions, max_len=self.max_length)


class StrandsRuntime(PackageRuntime):
    """StrandsAgents/strands-decider-* — Qwen3.5 backbone with a LoRA adapter and readout head.

    Causal, so state reuse would be valid in principle; the package not exposing the
    backbone is what rules it out, not the architecture.
    """

    package = "strands-decider"

    @staticmethod
    def _load_agent(model_id: str, device: str) -> Any:
        from strands_decider.modeling import StrandsDeciderModel

        return StrandsDeciderModel.load(model_id)

    def _call(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        # Unverified: the model card documents the CLI and the loader, not predict's
        # signature. Confirm against the installed package before trusting answers.
        return self.agent.predict(state, questions)


RUNTIMES: dict[str, type[PackageRuntime]] = {"laya": LayaRuntime, "strands": StrandsRuntime}
