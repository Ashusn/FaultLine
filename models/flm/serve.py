import os
import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .claims import rule_check
from .render import render_window
from .signals import rule_rank

app = FastAPI(title="FaultLine model API", version="1.0")


def env(name, default=""):
    return os.environ.get(name, default)


class EmbedIn(BaseModel):
    texts: list[str]
    kind: str = Field("doc", pattern="^(query|doc)$")


class RerankIn(BaseModel):
    query: str
    docs: list[str]
    top_k: int = 10


class ClassifyIn(BaseModel):
    signals: dict
    services: list[str]


class ClaimIn(BaseModel):
    claim: str
    evidence: str


class VerifyIn(BaseModel):
    claims: list[ClaimIn]
    threshold: float = 0.5


class CompressIn(BaseModel):
    incident: dict | None = None
    text: str | None = None
    max_chars: int = 12000


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(env("FLM_EMBEDDER"))


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(env("FLM_RERANKER"), num_labels=1)


@lru_cache(maxsize=1)
def classifier():
    from huggingface_hub import snapshot_download

    from .models.classifier import Classifier

    ref = env("FLM_CLASSIFIER")
    path = ref if os.path.isdir(ref) else snapshot_download(ref)
    return Classifier(path)


@lru_cache(maxsize=1)
def verifier():
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(env("FLM_VERIFIER"), num_labels=1)


@app.get("/v1/health")
def health():
    return {"ok": True, "time": int(time.time())}


@app.get("/v1/models")
def models():
    return {k: env(f"FLM_{k.upper()}") or None for k in ("embedder", "reranker", "classifier", "verifier", "compressor")}


@app.post("/v1/embed")
def embed(x: EmbedIn):
    if not env("FLM_EMBEDDER"):
        raise HTTPException(503, "FLM_EMBEDDER is not set")
    prefix = env("FLM_QUERY_PREFIX") if x.kind == "query" else env("FLM_DOC_PREFIX")
    v = embedder().encode([prefix + t for t in x.texts], normalize_embeddings=True)
    return {"vectors": v.tolist(), "dim": int(v.shape[1])}


@app.post("/v1/rerank")
def rerank(x: RerankIn):
    if not env("FLM_RERANKER"):
        raise HTTPException(503, "FLM_RERANKER is not set")
    s = reranker().predict([(x.query, d) for d in x.docs], show_progress_bar=False).tolist()
    order = sorted(range(len(x.docs)), key=lambda i: -s[i])[: x.top_k]
    return {"results": [{"index": i, "score": float(s[i])} for i in order]}


@app.post("/v1/classify")
def classify(x: ClassifyIn):
    if not env("FLM_CLASSIFIER"):
        return {"services": rule_rank(x.signals, x.services), "faults": [], "source": "rule"}
    return {**classifier().predict(x.signals, x.services), "source": "model"}


@app.post("/v1/verify")
def verify(x: VerifyIn):
    pairs = [(c.evidence, c.claim) for c in x.claims]
    if not env("FLM_VERIFIER"):
        return {"results": [{"supported": bool(rule_check(c.claim, c.evidence)), "score": float(rule_check(c.claim, c.evidence)), "source": "rule"} for c in x.claims]}
    import torch

    s = verifier().predict(pairs, activation_fn=torch.nn.Sigmoid(), show_progress_bar=False).tolist()
    return {"results": [{"supported": v >= x.threshold, "score": float(v), "source": "model"} for v in s]}


@app.post("/v1/compress")
def compress(x: CompressIn):
    url = env("FLM_COMPRESSOR_URL")
    model = env("FLM_COMPRESSOR")
    if not url or not model:
        raise HTTPException(503, "FLM_COMPRESSOR_URL and FLM_COMPRESSOR must point at a vLLM OpenAI server")
    text = x.text or (render_window(x.incident, x.max_chars) if x.incident else None)
    if not text:
        raise HTTPException(400, "send incident or text")
    import httpx

    from .models.compressor import parse, system_prompt

    cfg = {"signals": {"top_k": int(env("FLM_TOP_K", "8"))}}
    r = httpx.post(f"{url.rstrip('/')}/chat/completions", json={"model": model, "temperature": 0, "max_tokens": 512, "messages": [{"role": "system", "content": system_prompt(cfg)}, {"role": "user", "content": text}], "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    sig = parse(out)
    return {"signals": sig, "valid": sig is not None, "raw": None if sig else out}
