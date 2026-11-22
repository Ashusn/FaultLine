from importlib import import_module

TYPES = {
    "embedder": "embedder",
    "reranker": "reranker",
    "compressor": "compressor",
    "classifier": "classifier",
    "verifier": "verifier",
}


def module(cfg, name):
    t = cfg["models"][name]["type"]
    if t not in TYPES:
        raise SystemExit(f"unknown model type {t!r} for {name}")
    return import_module(f".{TYPES[t]}", __name__)
