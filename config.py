import os
from dataclasses import dataclass, field


@dataclass
class Config:
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda"))
    dtype: str = field(default_factory=lambda: os.getenv("DTYPE", "bfloat16"))
    # ponytail: semaphore cap; replace with async batching queue if p95 latency matters
    max_concurrent: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT", "4")))
    # "clef": Cloudflare/clef joint-head loader (causal, state cache, chunked prefill)
    # "laya" / "strands": models shipping their own package — see clef_engine.packaged.RUNTIMES
    # "generic": AutoModelForCausalLM with prompt scoring — an approximation, no trained head
    model_backend: str = field(default_factory=lambda: os.getenv("MODEL_BACKEND", "generic"))
    # Tokens the model uses for noul yes/no; override if your tokenizer splits these differently
    noul_yes_token: str = field(default_factory=lambda: os.getenv("NOUL_YES_TOKEN", "yes"))
    noul_no_token: str = field(default_factory=lambda: os.getenv("NOUL_NO_TOKEN", "no"))


config = Config()
