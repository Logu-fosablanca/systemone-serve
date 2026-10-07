"""Runtime for jaredpalmer/kev-* decision models (Qwen3.5 backbone, LoRA, pointer head).

Kev ships readable code, so this uses its own encode(), probs_and_prefix() and
probs_with_prefix() rather than reimplementing any of them. That matters for two reasons:

1. encode() packs a state prefix followed by one branch per question, and for a hybrid
   backbone (Kev-0.8B, Kev-27B) the backbone runs in row form where each branch reads
   the state's cache independently. A flat causal pass would let question 2 attend to
   question 1 and return confidently wrong probabilities with no error raised.

2. probs_and_prefix(enc) runs the state ONCE and returns both the answer probabilities
   and a reusable state prefix. forward_batch() runs the state per question instead:
   kev's own figure is 1011 -> 413 ms for 5 questions on Kev-0.8B bf16. probs_and_prefix
   is the right call regardless of whether there is a cache.

probs_with_prefix(enc, prefix) skips the state pass entirely for a cached state.
_rows_hidden() (called internally) creates a replica of the cache before each branch pass,
so the stored prefix is never modified -- no deep copy needed on reads.

Cross-request state cache lives in KevPrefixStore (byte-bounded LRU). Enable with
KEV_PREFIX_CACHE_GB (default 0 = off; the state prefix is on GPU so it eats VRAM).

KEV_CUDA_GRAPHS=1: model.graphs is only read inside probs_batch; probs_and_prefix and
probs_with_prefix never touch it. On this path graphs are pure cost: on a 4 GB card
~800 MiB of buffers collapse throughput 12-17x under concurrency. Leave it off unless
the card has room AND the runtime is migrated to probs_batch. See BENCHMARKS.md.
"""

from __future__ import annotations

import hashlib
import logging
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch

from .runtime import Job

log = logging.getLogger("clef.kev")


def _prefix_nbytes(prefix: Any) -> int:
    """Bytes used by a kev state prefix.

    cache.layers has two layer types:
      LinearAttentionLayer  (DeltaNet): .conv_states, .recurrent_states
      DynamicLayer          (full attn): .keys, .values
    h_state is None for hybrid backbones (Kev-0.8B, Kev-27B).
    """
    _, cache, h_state = prefix
    tensors: list[Any] = []
    for layer in cache.layers:
        tensors += [getattr(layer, name, None) for name in ("keys", "values", "conv_states", "recurrent_states")]
    if h_state is not None:
        tensors.append(h_state)
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


class KevPrefixStore:
    """LRU cache of kev state prefixes (Ls, cache, h_state), bounded by bytes.

    probs_with_prefix() creates an internal replica of the cache, so the stored prefix
    is never modified between calls -- no deep copy needed on reads or writes.
    """

    def __init__(self, budget_bytes: int) -> None:
        self.budget = budget_bytes
        self.used = 0
        self._items: OrderedDict[str, tuple[Any, int]] = OrderedDict()  # key -> (prefix, nbytes)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        pair = self._items.get(key)
        if pair is not None:
            self._items.move_to_end(key)
            self.hits += 1
            return pair[0]
        self.misses += 1
        return None

    def put(self, key: str, prefix: Any) -> None:
        if key in self._items:
            self._items.move_to_end(key)
            return
        nbytes = _prefix_nbytes(prefix)
        if nbytes > self.budget:
            return
        while self.used + nbytes > self.budget:
            _, (_, evicted) = self._items.popitem(last=False)
            self.used -= evicted
        self._items[key] = (prefix, nbytes)
        self.used += nbytes


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
        prefix_cache_bytes: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.api = api
        self.encode = encode
        self.overflow = overflow
        self.max_state = max_state
        self.max_branch = max_branch
        # states exposed so scheduler.stats_snapshot() can report state_cache_bytes.
        self.states = KevPrefixStore(prefix_cache_bytes) if prefix_cache_bytes > 0 else None
        self.stats = {"records": 0, "state_tokens_seen": 0, "truncated_states": 0,
                      "prefix_hits": 0, "prefix_misses": 0}

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
        state_key = hashlib.sha256(array("q", enc["ids"][:state_end]).tobytes()).hexdigest()
        return Job(
            request=request,
            enc=encoded,
            state_end=state_end,
            state_key=state_key,
            long=False,
            cost=len(enc["ids"]),
        )

    @torch.inference_mode()
    def run_short(self, jobs: list[Job]) -> list[dict[str, Any]]:
        results = []
        for job in jobs:
            prefix = self.states.get(job.state_key) if self.states else None
            if prefix is not None:
                # State already computed: only the branches run.
                probs = self.model.probs_with_prefix(job.enc.enc, prefix)
                self.stats["prefix_hits"] += 1
            else:
                # Full pass; returns both answers and the state prefix in one forward.
                probs, new_prefix = self.model.probs_and_prefix(job.enc.enc)
                self.stats["prefix_misses"] += 1
                if self.states is not None:
                    self.states.put(job.state_key, new_prefix)
            results.append(self._result(job, probs))
        return results

    def _result(self, job: Job, probs: list[torch.Tensor]) -> dict[str, Any]:
        # probs_and_prefix and probs_with_prefix already return softmax'd probabilities;
        # .float().tolist() is all that's needed, not .softmax(-1).
        probs_list = [z.float().tolist() for z in probs]
        self.stats["records"] += 1
        self.stats["state_tokens_seen"] += job.state_end
        return {
            "answers": self.api.to_answers(probs_list, job.enc.meta),
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
        if self.states is not None:
            # Clear warmup entries so they don't pollute production cache metrics.
            self.states = KevPrefixStore(self.states.budget)
