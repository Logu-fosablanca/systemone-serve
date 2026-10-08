"""GPU side of the Clef engine. Every method that touches the model runs on one worker thread.

Records with short states run as one padded batch through Cloudflare's own forward pass.
Records with long states prefill the system prompt and state in chunks, save that backbone
cache (attention K/V, DeltaNet states) with its final hidden states, then run only the
questions. A later record with the same state skips to the questions. The backbone is causal
and the state comes before the questions, so a reused state gives the same answer as a full pass.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import logging
import os
import sys
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("clef.runtime")


@dataclass(eq=False)
class Job:
    request: dict[str, Any]
    enc: Any  # joint_schema_model.EncodedRecord
    state_end: int  # index of the first schema token
    state_key: str
    long: bool
    cost: int  # tokens still to compute
    admitted: int = 0
    queued_at: float = 0.0
    future: Any = None
    pos: int = 0  # state tokens prefilled so far (long path)
    cache: Any = None
    hidden: list[torch.Tensor] = field(default_factory=list)
    # (position, key, snapshot_here) at every chunk boundary in the state, ascending.
    # Lookup matches any of them; only the flagged ones are written back.
    prefix_keys: list[tuple[int, str, bool]] = field(default_factory=list)

    def boundary_at(self, pos: int) -> tuple[str, bool] | None:
        for p, key, snap in self.prefix_keys:
            if p == pos:
                return key, snap
        return None


@dataclass
class _Saved:
    cache: Any
    hidden: torch.Tensor
    nbytes: int


def _nbytes(cache: Any, hidden: torch.Tensor) -> int:
    tensors = [hidden]
    for layer in cache.layers:
        tensors += [getattr(layer, name, None) for name in ("keys", "values", "conv_states", "recurrent_states")]
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


def _prefix_keys(
    ids: Any, start: int, end: int, chunk: int, snapshot_every: int
) -> list[tuple[int, str, bool]]:
    """Chained hashes at every chunk boundary in [start, end], plus `end` itself.

    Chained rather than hashing each prefix from scratch, so building the list is linear
    in the state length. Every boundary is a lookup point, which maximises the chance of
    matching a shared prefix; only every `snapshot_every`-th one (and `end`) is written
    back, which bounds how much a single long state can occupy.
    """
    keys: list[tuple[int, str, bool]] = []
    digest = hashlib.sha256(array("q", ids[:start]).tobytes()).digest()
    pos, i = start, 0
    while pos < end:
        nxt = min(pos + chunk, end)
        digest = hashlib.sha256(digest + array("q", ids[pos:nxt]).tobytes()).digest()
        pos, i = nxt, i + 1
        keys.append((pos, digest.hex(), i % snapshot_every == 0 or pos == end))
    return keys


class _Fp32Head(torch.nn.Module):
    """Run the joint schema head in fp32 while the backbone stays bf16 or FP8.

    The head decides the answer, so a rounding difference there can flip the argmax
    between two close options. The backbone is where the parameters are, so it keeps its
    own precision; the head is small enough that fp32 costs little. The fp32 copy of the
    output embedding is materialised once and held.
    """

    def __init__(self, head: Any) -> None:
        super().__init__()
        self.head = head.float()
        self._w32: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor, input_ids: Any, mask: Any, records: Any,
                lm_head_weight: torch.Tensor) -> Any:
        if self._w32 is None:
            self._w32 = lm_head_weight.float()
            log.info("fp32 output embedding materialised for the head (%.1f GB)",
                     self._w32.numel() * 4 / 1e9)
        return self.head(hidden.float(), input_ids, mask, records, self._w32)


class StateCache:
    """Saved states, bounded by bytes; least recently used goes first."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget = budget_bytes
        self.used = 0
        self.too_large = 0
        self._items: OrderedDict[str, _Saved] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def get(self, key: str) -> _Saved | None:
        saved = self._items.get(key)
        if saved is not None:
            self._items.move_to_end(key)
        return saved

    def longest(self, keys: list[tuple[int, str, bool]], touch: bool = True) -> tuple[int, _Saved | None]:
        """Deepest cached boundary among `keys`, or (0, None).

        Keys are ascending by position, so walking backwards returns the longest
        shared prefix: a transcript that grew, or a request sharing a fixed preamble,
        resumes from as far in as we have rather than recomputing from the start.

        `touch=False` skips the LRU reorder. prepare() runs on the encode thread while
        the GPU thread may be in long_step, and only one of them may mutate the ordering.
        """
        for pos, key, _ in reversed(keys):
            saved = self._items.get(key)
            if saved is not None:
                if touch:
                    self._items.move_to_end(key)
                return pos, saved
        return 0, None

    def put(self, key: str, cache: Any, hidden: torch.Tensor) -> None:
        nbytes = _nbytes(cache, hidden)
        if key in self._items:
            return
        if nbytes > self.budget:
            log.warning("state %s… too large for cache (%.1f MB > %.1f MB budget); not saving",
                        key[:8], nbytes / 1e6, self.budget / 1e6)
            self.too_large += 1
            return
        while self.used + nbytes > self.budget:
            self.used -= self._items.popitem(last=False)[1].nbytes
        # DeltaNet layers update their cache tensors in place, so store a private copy.
        self._items[key] = _Saved(copy.deepcopy(cache), hidden, nbytes)
        self.used += nbytes


def load_reference(path: Path) -> Any:
    """Import Cloudflare's joint_schema_model.py from a model snapshot."""
    spec = importlib.util.spec_from_file_location("joint_schema_model", path / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_fused_kernels() -> bool:
    try:
        import fla  # noqa: F401
        return True
    except ImportError:
        return False


def _check_kernels(device: str) -> None:
    if _has_fused_kernels() or not str(device).startswith("cuda"):
        return
    if os.getenv("CLEF_ALLOW_SLOW_KERNELS") == "1":
        log.warning("DeltaNet fast kernels missing; using the much slower PyTorch fallback")
        return
    raise RuntimeError(
        "flash-linear-attention and causal-conv1d are not installed, so DeltaNet layers would run the much "
        "slower PyTorch fallback. Install them, or set CLEF_ALLOW_SLOW_KERNELS=1 to run anyway."
    )


def _is_fp8_checkpoint(model: Any) -> bool:
    """Return True if backbone linear weights are already float8_e4m3fn.

    HF transformers can silently convert FP8 weights to BF16 during loading.
    This detects that and lets the caller decide whether to apply TorchAO instead.
    """
    for module in model.language_model.modules():
        w = getattr(module, "weight", None)
        if w is not None:
            return w.dtype == torch.float8_e4m3fn
    return False


def _apply_fp8(model: Any) -> Any:
    """Ensure backbone linear layers run FP8 W8A8.

    Path A: checkpoint already carries float8_e4m3fn weights (e.g.
    kurcontko/clef-flash-FP8-Dynamic loaded without silent dequant). No TorchAO needed.

    Path B: BF16 checkpoint (or HF silently dequanted an FP8 one). Apply TorchAO
    dynamic quantization. The joint head, vision tower, embeddings and norms stay BF16,
    matching what kurcontko's checkpoint preserves.
    """
    if _is_fp8_checkpoint(model):
        log.info("FP8 checkpoint active: backbone weights are float8_e4m3fn")
        return model
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_
    except ImportError:
        log.warning("CLEF_FP8=1 but torchao is not installed; running in original precision")
        return model
    config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    backbone = model.language_model.model.language_model
    quantize_(backbone, config)
    log.info("FP8 dynamic quantization applied to backbone via TorchAO (%d layers)",
             len(list(backbone.layers)))
    return model


class ClefRuntime:
    def __init__(
        self,
        model: Any,
        processor: Any,
        jsm: Any,
        *,
        max_length: int = 16384,
        chunk_tokens: int = 2048,
        long_state_tokens: int = 1024,
        state_cache_bytes: int = 16 << 30,
        compile_backbone: bool = False,
        packed: bool = False,
        prefix_snapshot_tokens: int = 4096,
        fp32_head: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.jsm = jsm
        self.max_length = max_length
        self.chunk_tokens = chunk_tokens
        self.long_state_tokens = long_state_tokens
        self.device = next(model.parameters()).device
        backbone = model.language_model
        self.text_model = backbone.model.language_model
        self.lm_head_weight = backbone.get_output_embeddings().weight
        self._compile_backbone = compile_backbone
        if packed and not _has_fused_kernels():
            log.warning("CLEF_PACKED=1 ignored: flash-linear-attention is required for "
                        "packed batching (the reference DeltaNet path doesn't reset "
                        "recurrent state at cu_seqlens boundaries)")
            packed = False
        self._packed = packed
        # Snapshot spacing is independent of chunk size: every chunk boundary is a lookup
        # point, but writing one back every chunk would make a long state cost O(n^2/chunk).
        self._snapshot_every = max(1, prefix_snapshot_tokens // max(1, chunk_tokens))
        if fp32_head:
            model.head = _Fp32Head(model.head)
            log.info("joint head running in fp32; the padded short-batch path is unaffected "
                     "unless CLEF_PACKED=1 is also set")
        self._fp32_head = fp32_head
        self.states = StateCache(state_cache_bytes)
        self.stats = {"state_hits": 0, "prefix_hits": 0, "state_misses": 0, "state_tokens_reused": 0}
        self.prefix_len = self._prefix_len()
        self._startup_cache: Any = None      # precomputed after warmup()
        self._startup_hidden: Any = None     # [prefix_len, d_model] fp32

    @classmethod
    def load(cls, model_id: str, device: str = "cuda", revision: str | None = None, **kwargs: Any) -> ClefRuntime:
        from huggingface_hub import snapshot_download

        _check_kernels(device)
        path = Path(model_id) if Path(model_id).is_dir() else Path(snapshot_download(model_id, revision=revision))
        jsm = load_reference(path)
        attn = os.getenv("CLEF_ATTN_IMPL")
        model, processor = jsm.load_release_model(path, device=device, **({"attn_implementation": attn} if attn else {}))
        if os.getenv("CLEF_FP8") == "1" and str(device).startswith("cuda"):
            model = _apply_fp8(model)
        log.info("Clef loaded from %s (attention override: %s)", path, attn or "none")
        return cls(model, processor, jsm, **kwargs)

    def _compute_startup_prefix(self) -> None:
        """Run the fixed system-prompt tokens once and cache the result.

        Every long request now starts here instead of from scratch. The state still
        covers tokens [prefix_len, state_end), so the saved hidden states and K/V cache
        from this call prepend correctly when computing or loading a state.

        DeltaNet layers mutate their cache in place, so long_step() copies this before
        resuming -- same rule as StateCache. warmup() resets states but keeps this intact.
        """
        questions = {"q": {"type": "noul", "instructions": "x"}}
        enc = self.jsm.encode_record(self.tokenizer, {"state": "a", "questions": questions})
        prefix_ids = tuple(enc.input_ids[:self.prefix_len])
        hidden, cache = self._prefill(prefix_ids, None)
        self._startup_cache = cache
        self._startup_hidden = hidden  # [prefix_len, d_model]
        log.info("startup prefix precomputed: %d tokens", self.prefix_len)

    def _prefix_len(self) -> int:
        # Two records that differ only in their state diverge where the state starts.
        questions = {"q": {"type": "noul", "instructions": "x"}}
        a = self.jsm.encode_record(self.tokenizer, {"state": "a", "questions": questions}).input_ids
        b = self.jsm.encode_record(self.tokenizer, {"state": "b", "questions": questions}).input_ids
        return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)

    def prepare(self, request: dict[str, Any]) -> Job:
        """Tokenize on the CPU; safe to call off the GPU thread."""
        enc = self.jsm.encode_record(self.tokenizer, request, max_length=self.max_length, processor=self.processor)
        # The schema and suffix don't depend on the state, so the same questions with an
        # empty state tell us where the state ends.
        empty = self.jsm.encode_record(self.tokenizer, {**request, "state": ""}, max_length=self.max_length)
        state_end = len(enc.input_ids) - (len(empty.input_ids) - self.prefix_len)
        long = state_end - self.prefix_len >= self.long_state_tokens
        keys = (
            _prefix_keys(enc.input_ids, self.prefix_len, state_end, self.chunk_tokens, self._snapshot_every)
            if long else []
        )
        state_key = keys[-1][1] if keys else hashlib.sha256(
            array("q", enc.input_ids[:state_end]).tobytes()
        ).hexdigest()
        # Cost is what is left after the deepest prefix we already hold, so the scheduler
        # prices a request that shares a preamble like the small job it actually is.
        reuse = self.states.longest(keys, touch=False)[0] if long else 0
        cost = len(enc.input_ids) - reuse
        return Job(request, enc, state_end, state_key, long, cost, prefix_keys=keys)

    @torch.inference_mode()
    def run_short(self, jobs: list[Job]) -> list[dict[str, Any]]:
        if self._packed and len(jobs) > 1:
            return self._run_short_packed(jobs)
        batch = self.jsm.collate_records([job.enc for job in jobs], self.tokenizer.pad_token_id, self.device)
        return [self._result(job, logits) for job, logits in zip(jobs, self.model(batch))]

    def _run_short_packed(self, jobs: list[Job]) -> list[dict[str, Any]]:
        """Run short records without padding by concatenating into one packed sequence.

        Each record's tokens are concatenated flat; cu_seqlens tells the backbone and
        attention kernels where each record starts.  The joint head receives per-record
        hidden states split back out of the packed output.
        """
        lengths = [len(job.enc.input_ids) for job in jobs]
        flat_ids = []
        for job in jobs:
            flat_ids.extend(job.enc.input_ids)
        input_ids = torch.tensor([flat_ids], device=self.device)
        cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(torch.tensor(lengths, dtype=torch.int32, device=self.device), dim=0, out=cu[1:])

        # cu_seq_lens_q is the kwarg name both DeltaNet (via kwargs.pop) and full-attention
        # (via FlashAttentionKwargs) layers read from **kwargs in transformers' Qwen3.5.
        out = self.text_model(input_ids=input_ids, cu_seq_lens_q=cu, use_cache=False)
        hidden = out.last_hidden_state[0]  # [total_tokens, d_model]

        results = []
        for i, job in enumerate(jobs):
            start, end = int(cu[i]), int(cu[i + 1])
            h = hidden[start:end].unsqueeze(0)  # [1, seq_len, d_model]
            rec_ids = torch.tensor([job.enc.input_ids], device=self.device)
            logits = self.model.head(h, rec_ids, torch.ones_like(rec_ids), [job.enc], self.lm_head_weight)[0]
            results.append(self._result(job, logits))
        return results

    @torch.inference_mode()
    def long_step(self, job: Job) -> dict[str, Any] | None:
        """Advance a long record by one chunk. Returns its result once it is done, else None."""
        ids = job.enc.input_ids
        if job.pos == 0:
            hit, saved = self.states.longest(job.prefix_keys)
            if saved is not None:
                job.cache, job.hidden, job.pos = copy.deepcopy(saved.cache), [saved.hidden], hit
                self.stats["state_hits" if hit == job.state_end else "prefix_hits"] += 1
                self.stats["state_tokens_reused"] += hit
            else:
                self.stats["state_misses"] += 1
                # Reuse the precomputed startup prefix so every long request avoids
                # recomputing the fixed system-prompt tokens. The startup hidden states
                # are prepended to the accumulated state hidden states below.
                if self._startup_cache is not None:
                    job.cache = copy.deepcopy(self._startup_cache)
                    job.hidden = [self._startup_hidden]
                    job.pos = self.prefix_len
        if job.pos < job.state_end:
            # Stop at the next boundary prepare() hashed, so a save lands on the same grid
            # a later request will look up. Without this the grid shifts by prefix_len
            # depending on whether the startup prefix was available to resume from.
            nxt = next((p for p, _, _ in job.prefix_keys if p > job.pos), job.state_end)
            end = min(job.pos + self.chunk_tokens, nxt, job.state_end)
            hidden, job.cache = self._prefill(ids[job.pos : end], job.cache)
            job.hidden.append(hidden)
            job.pos = end
            job.cost = len(ids) - end
            mark = job.boundary_at(end)
            if mark is not None and mark[1]:
                # put() deep-copies, so the job keeps advancing on its own cache.
                job.hidden = [torch.cat(job.hidden)]
                self.states.put(mark[0], job.cache, job.hidden[0])
            if end < job.state_end:
                return None
            if len(job.hidden) > 1:
                job.hidden = [torch.cat(job.hidden)]
        hidden, _ = self._prefill(ids[job.state_end :], job.cache)
        full = torch.cat([job.hidden[0], hidden])[None]
        job.cache, job.hidden = None, []
        input_ids = torch.tensor([ids], device=self.device)
        logits = self.model.head(full, input_ids, torch.ones_like(input_ids), [job.enc], self.lm_head_weight)[0]
        return self._result(job, logits)

    def _prefill(self, ids: tuple[int, ...], cache: Any) -> tuple[torch.Tensor, Any]:
        out = self.text_model(
            input_ids=torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True
        )
        return out.last_hidden_state[0], out.past_key_values

    def _result(self, job: Job, logits: list[torch.Tensor]) -> dict[str, Any]:
        questions = job.request["questions"]
        answers = {}
        for q, l in zip(job.enc.questions, logits):
            ans = self.jsm.systemone_answer(
                questions[q.question_id], dict(zip(q.option_ids, l.float().softmax(-1).tolist()))
            )
            if ans.get("type") == "noul" and "confidence" not in ans:
                n = ans["noul"]
                ans = {**ans, "confidence": max(n, 1.0 - n)}
            answers[q.question_id] = ans
        return {"answers": answers, "usage": {"input_tokens": len(job.enc.input_ids), "output_tokens": 0}}

    def warmup(self) -> None:
        """Run both paths once so kernels compile before the first real request."""
        self._compute_startup_prefix()
        questions = {"q": {"type": "noul", "instructions": "warmup"}}
        self.run_short([self.prepare({"state": "warmup", "questions": questions})])
        job = self.prepare({"state": "warmup " * 64, "questions": questions})
        job.long = True
        while self.long_step(job) is None:
            pass
        if self._compile_backbone:
            self.text_model = torch.compile(self.text_model, mode="reduce-overhead", dynamic=True)
            log.info("torch.compile applied to backbone; running compiled warmup pass")
            self.run_short([self.prepare({"state": "compile warmup", "questions": questions})])
            # Also trace the long path so torch.compile doesn't trigger JIT on the first real long request.
            compile_long = self.prepare({"state": "warmup " * 64, "questions": questions})
            compile_long.long = True
            while self.long_step(compile_long) is None:
                pass
        # Clear warmup entries from the state cache but keep the startup prefix.
        self.states = StateCache(self.states.budget)
        self.stats = dict.fromkeys(self.stats, 0)
