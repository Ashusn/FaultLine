import random
import time

from ..claims import PERTURB, rule_check
from ..common import read_jsonl, set_seed, wdir, wpath
from ..metrics import summarize
from ..train_utils import common_args, device, save_history

KEYS = ["correct", "catch", "false_reject"]


def rows(cfg, split, n=None):
    p = wpath(cfg, "data", "bench", "claims.jsonl") if split == "bench" else wpath(cfg, "data", "claims", f"{split}.jsonl")
    r = read_jsonl(p) if p.exists() else []
    if n and len(r) > n:
        r = random.Random(cfg.get("seed", 0)).sample(r, n)
    return r


def load(ref, m):
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(ref, num_labels=1, max_length=int(m.get("max_length", 384)), device=device())


def evaluate(cfg, name, ref, split, n=None):
    m = cfg["models"][name]
    data = rows(cfg, split, n)
    if not data:
        return None
    t = time.time()
    if ref == "rule":
        preds = [rule_check(r["claim"], r["evidence"]) for r in data]
    else:
        import torch

        model = load(ref, m)
        scores = model.predict([(r["evidence"], r["claim"]) for r in data], batch_size=int(m.get("eval_batch", 64)), activation_fn=torch.nn.Sigmoid(), show_progress_bar=False)
        th = float(m.get("threshold", 0.5))
        preds = [1 if float(s) >= th else 0 for s in scores]
    ms = (time.time() - t) * 1000 / len(data)
    items = []
    for r, p in zip(data, preds):
        it = {"id": r["id"], "kind": r["kind"], "correct": 1.0 if p == r["label"] else 0.0, "catch": None, "false_reject": None}
        if r["label"] == 0:
            it["catch"] = 1.0 if p == 0 else 0.0
        else:
            it["false_reject"] = 1.0 if p == 0 else 0.0
        items.append(it)
    s = summarize(items, KEYS, seed=cfg.get("seed", 0))
    s["ms_per_claim"] = round(ms, 3)
    s["catch_by_kind"] = {}
    for k in PERTURB:
        v = [it["catch"] for it in items if it["kind"] == k]
        if v:
            s["catch_by_kind"][k] = round(sum(v) / len(v), 4)
    return {"items": items, "summary": s}


def train(cfg, name):
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    m = cfg["models"][name]
    set_seed(cfg.get("seed", 0))
    data = [{"evidence": r["evidence"], "claim": r["claim"], "label": float(r["label"])} for r in rows(cfg, "train")]
    random.Random(cfg.get("seed", 0)).shuffle(data)
    model = load(m["base"], m)
    loss = BinaryCrossEntropyLoss(model)
    args = CrossEncoderTrainingArguments(**common_args(m, wdir(cfg, "runs", name), cfg.get("seed", 0)))
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=Dataset.from_list(data), loss=loss)
    trainer.train()
    out = wdir(cfg, "models", name)
    model.save(str(out))
    save_history(trainer, wdir(cfg, "models", f"{name}_training"))
    return {"local": str(out), "training": str(wdir(cfg, "models", f"{name}_training"))}
