import random
import time

from ..common import read_jsonl, set_seed, wdir, wpath, write_jsonl
from ..metrics import ndcg_at_k, recall_at_k, summarize
from ..train_utils import common_args, device, save_history
from . import embedder

KEYS = ["ndcg@10", "recall@10", "candidate_recall"]


def load(ref, m):
    from sentence_transformers.cross_encoder import CrossEncoder

    return CrossEncoder(ref, num_labels=1, max_length=int(m.get("max_length", 512)), device=device())


def candidates(cfg, name, split, n=None):
    m = cfg["models"][name]
    p = wpath(cfg, "data", "retrieval", f"{name}_cands_{split}_{n or 'all'}.jsonl")
    if p.exists():
        return read_jsonl(p)
    em = cfg["models"][m["first_stage"]]
    ref = em["base"]
    qs = embedder.queries(cfg, split, n)
    if not qs:
        return []
    docs = embedder.corpus(cfg)
    model = embedder.load(ref, em)
    ranked, _, _ = embedder.rank(model, qs, docs, *embedder.prefixes(em, ref), top=int(m.get("candidates", 50)))
    rows = [{"id": q["id"], "text": q["text"], "relevant": q["relevant"], "cands": r} for q, r in zip(qs, ranked)]
    write_jsonl(p, rows)
    return rows


def evaluate(cfg, name, ref, split, n=None):
    m = cfg["models"][name]
    rows = candidates(cfg, name, split, n)
    if not rows:
        return None
    by = {d["id"]: d["text"] for d in embedder.corpus(cfg)}
    items = []
    t = time.time()
    if ref == "first_stage":
        for r in rows:
            items.append({"id": r["id"], "ndcg@10": ndcg_at_k(r["cands"], r["relevant"], 10), "recall@10": recall_at_k(r["cands"], r["relevant"], 10), "candidate_recall": recall_at_k(r["cands"], r["relevant"], len(r["cands"]))})
    else:
        model = load(ref, m)
        for r in rows:
            scores = model.predict([(r["text"], by[c]) for c in r["cands"]], batch_size=int(m.get("eval_batch", 64)), show_progress_bar=False)
            order = [c for _, c in sorted(zip(scores.tolist(), r["cands"]), key=lambda x: -x[0])]
            items.append({"id": r["id"], "ndcg@10": ndcg_at_k(order, r["relevant"], 10), "recall@10": recall_at_k(order, r["relevant"], 10), "candidate_recall": recall_at_k(r["cands"], r["relevant"], len(r["cands"])), "top5": order[:5]})
    ms = (time.time() - t) * 1000 / len(rows)
    s = summarize(items, KEYS, seed=cfg.get("seed", 0))
    s["ms_per_query"] = round(ms, 2)
    return {"items": items, "summary": s}


def train(cfg, name):
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    import torch

    m = cfg["models"][name]
    set_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    rows = candidates(cfg, name, "train")
    by = {d["id"]: d["text"] for d in embedder.corpus(cfg)}
    k = int(m.get("negatives_per_query", 4))
    data = []
    for r in rows:
        rel = set(r["relevant"])
        for d in r["relevant"]:
            data.append({"query": r["text"], "doc": by[d], "label": 1.0})
        hard = [c for c in r["cands"] if c not in rel][: k * 3]
        for d in rng.sample(hard, min(k, len(hard))):
            data.append({"query": r["text"], "doc": by[d], "label": 0.0})
    rng.shuffle(data)
    model = load(m["base"], m)
    pos = sum(1 for d in data if d["label"] == 1.0) or 1
    neg = len(data) - pos
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(max(1.0, neg / pos)))
    args = CrossEncoderTrainingArguments(**common_args(m, wdir(cfg, "runs", name), cfg.get("seed", 0)))
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=Dataset.from_list(data), loss=loss)
    trainer.train()
    out = wdir(cfg, "models", name)
    model.save(str(out))
    save_history(trainer, wdir(cfg, "models", f"{name}_training"))
    return {"local": str(out), "training": str(wdir(cfg, "models", f"{name}_training"))}
