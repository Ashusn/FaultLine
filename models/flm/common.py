import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path=None):
    p = Path(path or os.environ.get("FLM_CONFIG", "configs/config.yaml"))
    if not p.is_absolute():
        p = ROOT / p
    cfg = yaml.safe_load(p.read_text())
    cfg["_path"] = str(p)
    work = os.environ.get("FLM_WORK") or cfg.get("work") or str(ROOT / "work" / p.stem)
    cfg["_work"] = str(Path(work).resolve())
    return cfg


def wpath(cfg, *parts):
    p = Path(cfg["_work"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def wdir(cfg, *parts):
    p = Path(cfg["_work"]).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def namespace(cfg):
    ns = cfg["hub"].get("namespace", "auto")
    if ns != "auto":
        return ns
    from huggingface_hub import HfApi

    return HfApi().whoami()["name"]


def repo_id(cfg, name):
    return f"{namespace(cfg)}/{cfg['hub']['prefix']}{cfg['models'][name]['repo']}"


def dataset_repo(cfg, key):
    return f"{namespace(cfg)}/{cfg['hub']['prefix']}{cfg['hub']['datasets'][key]}"


def model_cfg(cfg, name):
    return cfg["models"][name]


def read_json(p):
    return json.loads(Path(p).read_text())


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p


def read_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return p


def sha256_files(paths):
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def data_hash(cfg):
    files = sorted(wdir(cfg, "data", "splits").glob("*.jsonl"))
    return sha256_files(files) if files else "none"


def meta(cfg, extra=None):
    m = {
        "commit": git_commit(),
        "config": Path(cfg["_path"]).name,
        "seed": cfg.get("seed", 0),
        "data_hash": data_hash(cfg),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        m.update(extra)
    return m


def results_path(cfg, name, kind):
    return wpath(cfg, "results", f"{name}_{kind}.json")


def log(*a):
    print(*a, flush=True)
