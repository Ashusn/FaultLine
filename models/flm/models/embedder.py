import time

import numpy as np

from ..common import log, read_jsonl, set_seed, wdir, wpath, write_jsonl
from ..metrics import mrr_at_k, ndcg_at_k, recall_at_k, summarize
from ..train_utils import common_args, device, save_history

KEYS = ["ndcg@10", "recall@10", "mrr@10", "recall@50"]


def corpus(cfg):
    return read_jsonl(wpath(cfg, "data", "retrieval", "corpus.jsonl"))


def queries(cfg, split, n=None):
    p = wpath(cfg, "data", "retrieval", f"queries_{split}.jsonl")
    rows = read_jsonl(p) if p.exists() else []
    return rows[:n] if n else rows


def load(ref, m=None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(ref, device=device())
    if m and m.get("max_seq_length"):
        model.max_seq_length = int(m["max_seq_length"])
    return model


def encode(model, texts, prefix="", batch=64):
    return np.asarray(model.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)


def rank(model, qs, docs, qp="", dp="", top=100, doc_emb=None):
    d = doc_emb if doc_emb is not None else encode(model, [x["text"] for x in docs], dp)
    q = encode(model, [x["text"] for x in qs], qp)
    sims = q @ d.T
    top = min(top, len(docs))
    idx = np.argsort(-sims, axis=1)[:, :top]
    return [[docs[j]["id"] for j in row] for row in idx], sims, d


def prefixes(m, ref):
    for b in m.get("baselines", []):
        if b["id"] == ref:
            return b.get("query_prefix", ""), b.get("doc_prefix", "")
    return m.get("query_prefix", ""), m.get("doc_prefix", "")


def evaluate(cfg, name, ref, split, n=None):
    m = cfg["models"][name]
    qs = queries(cfg, split, n)
    if not qs:
        return None
    docs = corpus(cfg)
    model = load(ref, m)
    qp, dp = prefixes(m, ref)
    t = time.time()
    ranked, _, _ = rank(model, qs, docs, qp, dp)
    ms = (time.time() - t) * 1000 / len(qs)
    items = []
    for q, r in zip(qs, ranked):
        items.append({"id": q["id"], "ndcg@10": ndcg_at_k(r, q["relevant"], 10), "recall@10": recall_at_k(r, q["relevant"], 10), "mrr@10": mrr_at_k(r, q["relevant"], 10), "recall@50": recall_at_k(r, q["relevant"], 50), "top5": r[:5]})
    s = summarize(items, KEYS, seed=cfg.get("seed", 0))
    s["ms_per_query"] = round(ms, 2)
    return {"items": items, "summary": s}


def mine_negatives(model, pairs, qrel, docs, m):
    qp, dp = m.get("query_prefix", ""), m.get("doc_prefix", "")
    text_to_ids = {}
    for d in docs:
        text_to_ids.setdefault(d["text"], set()).add(d["id"])
    dtexts = [d["text"] for d in docs]
    demb = encode(model, dtexts, dp)
    anchors = [p["anchor"] for p in pairs]
    qemb = encode(model, anchors, qp)
    pemb = encode(model, [p["positive"] for p in pairs], dp)
    sims = qemb @ demb.T
    pos_s = (qemb * pemb).sum(1)
    k = int(m.get("num_negatives", 3))
    lo, hi = int(m.get("neg_range_min", 5)), int(m.get("neg_range_max", 50))
    margin = float(m.get("neg_relative_margin", 0.05))
    out, filled = [], 0
    for i, p in enumerate(pairs):
        bad = set(qrel.get(p["qid"], [])) | {p["did"]}
        order = [j for r, j in enumerate(np.argsort(-sims[i])) if r >= lo and docs[j]["id"] not in bad and dtexts[j] != p["positive"]]
        negs = [dtexts[j] for j in order[: hi - lo] if sims[i, j] <= pos_s[i] * (1 - margin)][:k]
        if len(negs) < k:
            filled += 1
            negs += [dtexts[j] for j in order if dtexts[j] not in negs][: k - len(negs)]
        if len(negs) == k:
            out.append({"anchor": p["anchor"], "positive": p["positive"], **{f"negative_{n + 1}": t for n, t in enumerate(negs)}})
    log("hard negatives mined for", len(out), "of", len(pairs), "pairs,", filled, "filled without the margin rule")
    return out


def train(cfg, name):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss

    m = cfg["models"][name]
    set_seed(cfg.get("seed", 0))
    pairs = read_jsonl(wpath(cfg, "data", "retrieval", "pairs_train.jsonl"))
    qrel = {q["id"]: q["relevant"] for q in queries(cfg, "train")}
    docs = corpus(cfg)
    model = load(m["base"], m)
    rows = mine_negatives(model, pairs, qrel, docs, m) if m.get("hard_negatives", True) else [{"anchor": p["anchor"], "positive": p["positive"]} for p in pairs]
    if not rows:
        raise SystemExit("no training rows after mining")
    qp, dp = m.get("query_prefix", ""), m.get("doc_prefix", "")
    for r in rows:
        r["anchor"] = qp + r["anchor"]
        for k in list(r):
            if k != "anchor":
                r[k] = dp + r[k]
    write_jsonl(wpath(cfg, "data", "retrieval", f"{name}_train_rows.jsonl"), rows)
    ds = Dataset.from_list(rows)
    if m.get("cached_loss", True):
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=int(m.get("mini_batch_size", 32)))
    else:
        loss = MultipleNegativesRankingLoss(model)
    out = wdir(cfg, "models", name)
    args = SentenceTransformerTrainingArguments(**common_args(m, wdir(cfg, "runs", name), cfg.get("seed", 0)), batch_sampler="no_duplicates")
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save(str(out))
    save_history(trainer, wdir(cfg, "models", f"{name}_training"))
    return {"local": str(out), "training": str(wdir(cfg, "models", f"{name}_training"))}
