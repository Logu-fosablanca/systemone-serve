"""vLLM endpoint plugin: serves POST /v1/systemone on vLLM's port by forwarding to the Clef
engine, which runs as its own process (main.py with MODEL_BACKEND=clef).

vLLM's API-key middleware checks the caller first, and the same Authorization header is
passed on, so one VLLM_API_KEY covers both servers. No model is loaded in vLLM's process.
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

import aiohttp
from fastapi import FastAPI, Request, Response


class JevSystemOnePlugin:
    name = "jev_systemone"
    required_tasks = None

    def attach_router(self, app: FastAPI) -> None:
        @app.post("/v1/systemone")
        async def systemone(raw_request: Request) -> Response:
            state = raw_request.app.state
            try:
                async with state.systemone_session.post(
                    state.systemone_url,
                    data=await raw_request.body(),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": raw_request.headers.get("authorization", ""),
                    },
                ) as upstream:
                    return Response(await upstream.read(), status_code=upstream.status, media_type="application/json")
            except aiohttp.ClientError as exc:
                return Response(f'{{"detail": "clef engine unreachable: {type(exc).__name__}"}}', status_code=502,
                                media_type="application/json")

    async def init_state(self, engine_client: Any, state: Any, args: Namespace) -> None:
        state.systemone_url = os.getenv("CLEF_ENGINE_URL", "http://127.0.0.1:8001").rstrip("/") + "/v1/systemone"
        state.systemone_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=float(os.getenv("CLEF_PROXY_TIMEOUT_S", "300")))
        )
