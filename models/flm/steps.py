import os

from .common import read_json, wpath, write_json

DATA_STEPS = ["data:build", "data:compressor", "data:retrieval", "data:claims", "data:push"]
MODEL_STAGES = ["baseline", "train", "push", "verify", "eval", "publish"]
BENCH_STEPS = ["bench:freeze", "bench:run", "bench:publish"]


def model_names(cfg):
    names = [n for n, m in cfg["models"].items() if m.get("enabled", True)]
    only = os.environ.get("ONLY_MODELS", "").strip()
    if only:
        want = [x.strip() for x in only.split(",") if x.strip()]
        unknown = [x for x in want if x not in cfg["models"]]
        if unknown:
            raise SystemExit(f"ONLY_MODELS has unknown names: {unknown}")
        names = [n for n in names if n in want]
    return names


def step_list(cfg):
    steps = ["check"] + DATA_STEPS
    for n in model_names(cfg):
        steps += [f"{n}:{s}" for s in MODEL_STAGES]
    return steps + BENCH_STEPS + ["report"]


def state_path(cfg):
    return wpath(cfg, "state.json")


def load_state(cfg):
    p = state_path(cfg)
    return read_json(p) if p.exists() else {"done": [], "durations": {}}


def save_state(cfg, s):
    write_json(state_path(cfg), s)


def mark_done(cfg, step, seconds):
    s = load_state(cfg)
    if step not in s["done"]:
        s["done"].append(step)
    s["durations"][step] = round(seconds, 1)
    write_json(state_path(cfg), s)


def reset(cfg, step):
    s = load_state(cfg)
    targets = [x for x in s["done"] if x == step or (step.endswith(":*") and x.startswith(step[:-1]))]
    s["done"] = [x for x in s["done"] if x not in targets]
    write_json(state_path(cfg), s)
    return targets
