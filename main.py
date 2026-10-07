from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from clef_engine import RUNTIMES, ClefEngine, ClefRuntime, KevRuntime, QueueFull
from config import config
from engine import get_engine
from schema import SystemOneRequest, SystemOneResponse, Usage

# Backends served by ClefEngine; anything else falls through to the generic engine.
SYSTEMONE_BACKENDS = {"clef", "kev", *RUNTIMES}
API_KEY = os.getenv("VLLM_API_KEY", "")
_bearer = HTTPBearer(auto_error=False)
_batch_size = Histogram("clef_batch_size", "Records per short-batch forward pass", buckets=(1, 2, 4, 8, 16, 32))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


async def require_key(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if creds is None:
        raise HTTPException(401, "missing bearer token")
    if not hmac.compare_digest(creds.credentials, API_KEY):
        raise HTTPException(401, "invalid bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("VLLM_API_KEY must be set; refusing to serve unauthenticated")
    if config.model_backend in SYSTEMONE_BACKENDS and (os.getenv("NOUL_YES_TOKEN") or os.getenv("NOUL_NO_TOKEN")):
        logging.getLogger("jev.config").warning(
            "NOUL_YES_TOKEN/NOUL_NO_TOKEN have no effect on the %s backend", config.model_backend
        )
    if config.model_backend in SYSTEMONE_BACKENDS:
        if config.model_backend in RUNTIMES:
            rt = await asyncio.to_thread(
                RUNTIMES[config.model_backend].load,
                config.model_path,
                config.device,
                max_length=_env_int("PACKAGE_MAX_LENGTH", 1024),
            )
        elif config.model_backend == "kev":
            rt = await asyncio.to_thread(
                KevRuntime.load,
                config.model_path,
                config.device,
                max_state=int(os.getenv("KEV_MAX_STATE") or 0) or None,
                max_branch=int(os.getenv("KEV_MAX_BRANCH") or 0) or None,
                prefix_cache_bytes=int(float(os.getenv("KEV_PREFIX_CACHE_GB", "0")) * 2**30),
            )
        else:
            rt = await asyncio.to_thread(
                ClefRuntime.load,
                config.model_path,
                config.device,
                revision=os.getenv("CLEF_REVISION") or None,
                chunk_tokens=_env_int("CLEF_CHUNK_TOKENS", 2048),
                long_state_tokens=_env_int("CLEF_LONG_STATE_TOKENS", 1024),
                state_cache_bytes=int(float(os.getenv("CLEF_STATE_CACHE_GB", "16")) * 2**30),
                compile_backbone=os.getenv("CLEF_COMPILE") == "1",
                packed=os.getenv("CLEF_PACKED") == "1",
            )
        await asyncio.to_thread(rt.warmup)
        app.state.clef = ClefEngine(
            rt,
            batch_tokens=_env_int("CLEF_BATCH_TOKENS", 8192),
            max_batch=_env_int("CLEF_BATCH_MAX", 32),
            max_queued_tokens=_env_int("CLEF_MAX_QUEUED_TOKENS", 524288),
            answer_cache_size=_env_int("CLEF_ANSWER_CACHE_SIZE", 10000),
            on_batch=_batch_size.observe,
        )
        app.state.clef.start()
        app.state.model_name = config.model_path.rstrip("/").split("/")[-1]
    else:
        app.state.engine = get_engine()
        app.state.sem = asyncio.Semaphore(config.max_concurrent)
        app.state.model_name = app.state.engine.model_name
    yield


app = FastAPI(title="jev-inference", version="0.3.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


@app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(require_key)])
async def systemone(req: SystemOneRequest):
    if not req.questions:
        raise HTTPException(400, "questions must not be empty")
    for qid, q in req.questions.items():
        if q.type == "score" and not (2 <= len(q.criteria) <= 10):
            raise HTTPException(400, f"question '{qid}': score needs 2-10 levels")
        if q.type == "choice" and not (1 <= len(q.criteria) <= 255):
            raise HTTPException(400, f"question '{qid}': choice needs 1-255 options")

    if req.scoring == "multi" and config.model_backend in SYSTEMONE_BACKENDS:
        raise HTTPException(400, f"'scoring: multi' has no effect on the {config.model_backend!r} backend; "
                                  "omit scoring or use 'single'")
    if config.model_backend in SYSTEMONE_BACKENDS:
        request = {
            "state": req.state,
            "questions": {qid: q.model_dump(exclude_none=True) for qid, q in req.questions.items()},
        }
        try:
            out = await app.state.clef.submit(request)
        except QueueFull:
            raise HTTPException(429, "queue full, retry later", headers={"Retry-After": "2"})
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        answers, tokens = out["answers"], out["usage"]["input_tokens"]
    else:
        state = req.state if isinstance(req.state, str) else json.dumps(req.state, sort_keys=True)
        loop = asyncio.get_running_loop()
        async with app.state.sem:
            answers, tokens = await loop.run_in_executor(
                None, app.state.engine.run, state, req.questions, req.scoring
            )

    return SystemOneResponse(
        model=app.state.model_name,
        answers=answers,
        usage=Usage(input_tokens=tokens),
    )


@app.get("/health")
async def health():
    clef = getattr(app.state, "clef", None)
    return {"status": "ok", "model": app.state.model_name, **({"stats": clef.stats_snapshot()} if clef else {})}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": app.state.model_name, "object": "model"}]}


if __name__ == "__main__":
    import uvicorn

    # The app object, not "main:app": an import string makes uvicorn import this module a
    # second time (it is already __main__), re-running module-level code and tripping
    # prometheus' DuplicateTimeseries on the histogram below.
    uvicorn.run(app, host=config.host, port=config.port)
