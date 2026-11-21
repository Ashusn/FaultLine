from collections import Counter

from . import claims, retrieval
from .common import log, read_jsonl, wdir, wpath, write_json, write_jsonl
from .render import render_signals, render_window
from .signals import extract
from .sources import collect
from .split import SPLITS, leak_report, split_incidents


def split_path(cfg, name):
    return wpath(cfg, "data", "splits", f"{name}.jsonl")


def load_split(cfg, name, n=None):
    rows = read_jsonl(split_path(cfg, name))
    return rows[:n] if n else rows


def step_build(cfg):
    incidents = collect(cfg)
    k = int(cfg["signals"]["top_k"])
    for inc in incidents:
        inc["signals"] = extract(inc, top_k=k, cfg=cfg)
    splits = split_incidents(incidents, cfg)
    rep = leak_report(splits, cfg)
    rep["faults"] = {n: dict(Counter(i["label"]["fault"] for i in v)) for n, v in splits.items()}
    rep["apps"] = {n: dict(Counter(i["app"] for i in v)) for n, v in splits.items()}
    write_json(wpath(cfg, "results", "data_report.json"), rep)
    if not rep["ok"]:
        raise SystemExit("leak check FAILED: " + "; ".join(rep["problems"]))
    for n in SPLITS:
        write_jsonl(split_path(cfg, n), splits[n])
    empty = [n for n in ("train", "val", "test") if not splits[n]]
    if empty:
        raise SystemExit(f"empty splits: {empty}")
    log("splits", rep["sizes"])
    log("leak check PASSED")


def step_compressor_data(cfg):
    mx = int(cfg["models"].get("compressor_sft", cfg["models"].get("compressor_grpo", {})).get("max_input_chars", 12000))
    for n in SPLITS:
        rows = [{"id": i["incident_id"], "input": render_window(i, mx), "target": render_signals(i["signals"])} for i in load_split(cfg, n)]
        write_jsonl(wpath(cfg, "data", "compressor", f"{n}.jsonl"), rows)
    log("compressor data written")


def step_retrieval(cfg):
    splits = {n: load_split(cfg, n) for n in SPLITS}
    corpus, queries, pairs = retrieval.build(cfg, splits)
    missing = retrieval.check_runbooks(corpus)
    if missing:
        log("runbooks missing fault sections:", ", ".join(missing))
    s = cfg["retrieval"].get("synthetic", {})
    if s.get("enabled"):
        from .synthetic import paraphrase

        services = sorted({x for v in splits.values() for i in v for x in i["services"]})
        extra = paraphrase(cfg, queries["train"], services)
        queries["train"] += extra
        by_id = {d["id"]: d for d in corpus}
        for q in extra:
            for did in q["relevant"][: cfg["retrieval"].get("max_positives_per_query", 4)]:
                pairs.append({"anchor": q["text"], "positive": by_id[did]["text"], "qid": q["id"], "did": did})
    d = wdir(cfg, "data", "retrieval")
    write_jsonl(d / "corpus.jsonl", corpus)
    for n, rows in queries.items():
        write_jsonl(d / f"queries_{n}.jsonl", rows)
    write_jsonl(d / "pairs_train.jsonl", pairs)
    write_json(wpath(cfg, "results", "retrieval_report.json"), {"corpus": len(corpus), "kinds": dict(Counter(x["kind"] for x in corpus)), "queries": {n: len(v) for n, v in queries.items()}, "pairs": len(pairs), "missing_runbook_faults": missing})
    log("retrieval data", len(corpus), "docs", len(pairs), "pairs")


def step_claims(cfg):
    splits = {n: load_split(cfg, n) for n in SPLITS}
    out = claims.build(cfg, splits)
    for n, rows in out.items():
        write_jsonl(wpath(cfg, "data", "claims", f"{n}.jsonl"), rows)
    write_json(wpath(cfg, "results", "claims_report.json"), {n: dict(Counter(r["kind"] for r in v)) for n, v in out.items()})
    log("claims data", {n: len(v) for n, v in out.items()})


def step_push(cfg):
    from .hub import push_dataset

    push_dataset(cfg, "incidents", wdir(cfg, "data", "splits"))
    push_dataset(cfg, "compressor", wdir(cfg, "data", "compressor"))
    push_dataset(cfg, "retrieval", wdir(cfg, "data", "retrieval"))
    push_dataset(cfg, "claims", wdir(cfg, "data", "claims"))
