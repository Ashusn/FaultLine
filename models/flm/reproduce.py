import os
import shutil

from .common import dataset_repo, log, read_json, repo_id, wdir, wpath, write_json
from .models import module
from .proof import claim_value

DATA = {"incidents": "splits", "compressor": "compressor", "retrieval": "retrieval", "claims": "claims"}


def fetch_data(cfg):
    from huggingface_hub import snapshot_download

    for key, sub in DATA.items():
        d = wdir(cfg, "data", sub)
        snapshot_download(dataset_repo(cfg, key), repo_type="dataset", local_dir=str(d), allow_patterns=["*.jsonl"], ignore_patterns=["report/*"])
    snapshot_download(dataset_repo(cfg, "bench"), repo_type="dataset", local_dir=str(wdir(cfg, "published")), allow_patterns=["proof.json", "results/*", "labels.jsonl"])
    log("published data downloaded")


def fetch_model(cfg, name):
    from huggingface_hub import snapshot_download

    d = wdir(cfg, "repro_models", name)
    snapshot_download(repo_id(cfg, name), local_dir=str(d), ignore_patterns=["adapter/*", "training/*", "eval/*", "logs/*"])
    return str(d)


def run(cfg):
    fetch_data(cfg)
    pub = read_json(wpath(cfg, "published", "proof.json"))
    want = {x.strip() for x in os.environ.get("CLAIMS", "").split(",") if x.strip()}
    tol = float(cfg.get("proof", {}).get("tolerance", 0.5))
    defs = {c["id"]: c for c in cfg["proof"]["claims"]}
    out = []
    for pc in pub["claims"]:
        c = defs.get(pc["id"])
        if not c or (want and pc["id"] not in want):
            continue
        if c.get("source") == "bench" and pc.get("result") and c["key"] in ("top1", "top3"):
            from .bench import score
            from .common import read_jsonl

            res_files = sorted(wdir(cfg, "published", "results").rglob("bench_results.json"))
            sysname = read_json(res_files[-1])["headline"]["system"]
            pred = read_jsonl(sorted(wdir(cfg, "published", "results").rglob(f"pred_{sysname}.jsonl"))[-1])
            sc = score(pred, read_jsonl(wpath(cfg, "published", "labels.jsonl")))
            a, b = pc["result"]["value"], round(100 * sc[c["key"]], 1)
            ok = abs(float(a) - float(b)) <= tol
            log(f"{pc['id']}: published {a}, rescored from public predictions {b}, {'MATCH' if ok else 'DIFF'}")
            out.append({"id": pc["id"], "published": a, "reproduced": b, "match": ok, "split": "bench"})
            continue
        if not c.get("model") or not pc.get("result"):
            log(f"{pc['id']}: skipped, comes from an external file or was not measured")
            continue
        split = pc["result"]["split"]
        path = fetch_model(cfg, c["model"])
        mod = module(cfg, c["model"])
        res = mod.evaluate(cfg, c["model"], path, split, cfg["eval"].get("full_n"))
        full = {"splits": {split: res["summary"]}}
        base_full = None
        if c.get("kind") == "relative_gain_percent":
            from .pipeline import base_ref

            b = mod.evaluate(cfg, c["model"], base_ref(cfg, c["model"]), split, cfg["eval"].get("full_n"))
            base_full = {"splits": {split: b["summary"]}}
        write_json(wpath(cfg, "results", f"{c['model']}_full.json"), full)
        if base_full:
            write_json(wpath(cfg, "results", f"{c['model']}_base_full.json"), base_full)
        mine = claim_value(cfg, {**c, "split": split, "fallback_splits": []})
        a, b = pc["result"]["value"], mine["value"] if mine else None
        ok = b is not None and abs(float(a) - float(b)) <= tol
        log(f"{pc['id']}: published {a}, reproduced {b}, {'MATCH' if ok else 'DIFF'}")
        out.append({"id": pc["id"], "published": a, "reproduced": b, "match": ok, "split": split})
        shutil.rmtree(path, ignore_errors=True)
    write_json(wpath(cfg, "results", "reproduce.json"), out)
    if not all(x["match"] for x in out):
        raise SystemExit("reproduce found differences")
    log("reproduce: all MATCH")
