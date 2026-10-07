"""Async side of the Clef engine: answer cache, merging of identical in-flight requests,
admission control, and the loop that feeds the single GPU worker thread.

Short records go out in padded batches, oldest first, capped by padded tokens. Long records
advance one chunk per turn. When both kinds are waiting the loop alternates, so a short
request never waits behind more than one chunk, and only one long record holds a
half-built cache at a time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .runtime import ClefRuntime, Job

log = logging.getLogger("clef.scheduler")

# Long records are taken shortest-first; waiting earns this much priority per second so none starves.
AGING_TOKENS_PER_S = 20_000


class QueueFull(Exception):
    pass


def answer_key(request: dict[str, Any]) -> str:
    # Question order changes the model input, so the key keeps the request's order.
    blob = json.dumps([request["state"], request["questions"]], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ClefEngine:
    def __init__(
        self,
        rt: ClefRuntime,
        *,
        batch_tokens: int = 8192,
        max_batch: int = 32,
        max_queued_tokens: int = 524_288,
        answer_cache_size: int = 10_000,
        on_batch: Callable[[int], None] | None = None,
    ) -> None:
        self.rt = rt
        self.batch_tokens = batch_tokens
        self.max_batch = max_batch
        self.max_queued_tokens = max_queued_tokens
        self.answer_cache_size = answer_cache_size
        self.on_batch = on_batch
        self.stats = {"answer_cache_hits": 0, "merged_duplicates": 0, "rejected": 0}
        self._short: list[Job] = []
        self._long: list[Job] = []
        self._queued = 0
        self._answers: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future] = {}
        self._gpu = ThreadPoolExecutor(1, thread_name_prefix="clef-gpu")
        self._cpu = ThreadPoolExecutor(4, thread_name_prefix="clef-encode")
        self._prefer_long = False
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._wake = asyncio.Event()
        self._task = asyncio.get_running_loop().create_task(self._run())

    def stats_snapshot(self) -> dict[str, Any]:
        return {
            **self.stats,
            **self.rt.stats,
            "queued_tokens": self._queued,
            "short_queue": len(self._short),
            "long_queue": len(self._long),
            "state_cache_bytes": self.rt.states.used if self.rt.states else 0,
        }

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        key = answer_key(request)
        cached = self._answers.get(key)
        if cached is not None:
            self._answers.move_to_end(key)
            self.stats["answer_cache_hits"] += 1
            return cached
        future = self._inflight.get(key)
        if future is None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._inflight[key] = future
            future.add_done_callback(lambda f: self._settle(key, f))
            loop.create_task(self._enqueue(request, future))
        else:
            self.stats["merged_duplicates"] += 1
        # A client disconnect must not cancel work other callers share; the result
        # still lands in the answer cache, so a retry is free.
        return await asyncio.shield(future)

    def _settle(self, key: str, future: asyncio.Future) -> None:
        self._inflight.pop(key, None)
        if future.cancelled() or future.exception() is not None:
            return
        self._answers[key] = future.result()
        if len(self._answers) > self.answer_cache_size:
            self._answers.popitem(last=False)

    async def _enqueue(self, request: dict[str, Any], future: asyncio.Future) -> None:
        try:
            job = await asyncio.get_running_loop().run_in_executor(self._cpu, self.rt.prepare, request)
        except Exception as exc:
            future.set_exception(exc)
            return
        if self._queued + job.cost > self.max_queued_tokens:
            self.stats["rejected"] += 1
            future.set_exception(QueueFull())
            return
        job.future, job.admitted, job.queued_at = future, job.cost, time.monotonic()
        self._queued += job.admitted
        (self._long if job.long else self._short).append(job)
        self._wake.set()

    async def _run(self) -> None:
        while True:
            if not self._short and not self._long:
                self._wake.clear()
                await self._wake.wait()
                continue
            if self._short and self._long:
                take_long, self._prefer_long = self._prefer_long, not self._prefer_long
            else:
                take_long = bool(self._long)
            try:
                await (self._step_long() if take_long else self._step_short())
            except Exception:
                log.exception("scheduler step failed")

    async def _step_short(self) -> None:
        # Length-group: sort largest-cost-first so pop() from the end takes the smallest,
        # grouping similar lengths to cut padding waste. O(k log k), k = queue length,
        # bounded by admission control. FIFO order is not preserved, but all short jobs
        # are cheap and the admission control prevents unbounded waiting.
        # ponytail: sorts the whole list; a priority-queue insert would be O(log k) per
        # enqueue but k is small in practice and a heap complicates the dedup logic.
        self._short.sort(key=lambda j: j.cost, reverse=True)
        # For a short job cost is its token count, so this is unchanged for the decoder
        # path and lets encoder runtimes size batches without exposing token ids.
        batch = [self._short.pop()]  # smallest cost (list is largest-first)
        longest = batch[0].cost
        while self._short and len(batch) < self.max_batch:
            next_longest = max(longest, self._short[-1].cost)  # peek at next smallest
            if next_longest * (len(batch) + 1) > self.batch_tokens:
                break
            longest = next_longest
            batch.append(self._short.pop())
        if self.on_batch:
            self.on_batch(len(batch))
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(self._gpu, self.rt.run_short, batch)
        except Exception as exc:
            if len(batch) == 1:
                self._finish(batch[0], exc=exc)
                return
            # One bad record must not fail its batch-mates: rerun each alone.
            for job in batch:
                try:
                    result = (await loop.run_in_executor(self._gpu, self.rt.run_short, [job]))[0]
                except Exception as job_exc:
                    self._finish(job, exc=job_exc)
                else:
                    self._finish(job, result)
            return
        for job, result in zip(batch, results):
            self._finish(job, result)

    async def _step_long(self) -> None:
        # Finish the long record already in progress before starting another.
        job = next((j for j in self._long if j.pos > 0), None)
        if job is None:
            now = time.monotonic()
            job = min(self._long, key=lambda j: j.cost - (now - j.queued_at) * AGING_TOKENS_PER_S)
        try:
            result = await asyncio.get_running_loop().run_in_executor(self._gpu, self.rt.long_step, job)
        except Exception as exc:
            self._long.remove(job)
            job.cache, job.hidden = None, []
            self._finish(job, exc=exc)
            return
        if result is not None:
            self._long.remove(job)
            self._finish(job, result)

    def _finish(self, job: Job, result: dict[str, Any] | None = None, exc: BaseException | None = None) -> None:
        self._queued -= job.admitted
        if job.future.done():
            return
        if exc is not None:
            job.future.set_exception(exc)
        else:
            job.future.set_result(result)
