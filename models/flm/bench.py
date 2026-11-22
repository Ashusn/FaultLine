import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from . import claims, retrieval
from .common import dataset_repo, log, meta, namespace, read_json, read_jsonl, repo_id, results_path, sha256_files, wdir, wpath, write_json, write_jsonl
from .metrics import bootstrap_ci, mean
from .render import render_signals, render_window, signals_query
from .schema import group_key, public

POOL = ["test", "test_ood_fault", "test_ood_app"]
PUBLIC_FILES = ["incidents.jsonl", "labels.jsonl", "manifest.json"]


def bdir(cfg):
    return wdir(cfg, "data", "bench")


def bench_url(cfg, path="BENCH.md"):
    return f"https://huggingface.co/datasets/{dataset_repo(cfg, 'bench')}/blob/main/{path}"


def _tag_exists(cfg, tag):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        refs = HfApi().list_repo_refs(dataset_repo(cfg, "bench"), repo_type="dataset")
    except RepositoryNotFoundError:
        return False
    return any(t.name == tag for t in refs.tags)


def select(cfg, pool):
    size = int(cfg["bench"]["size"])
    rng = random.Random(f"bench:{cfg.get('seed', 0)}:{cfg['bench']['version']}")
    buckets = defaultdict(list)
    for split, inc in pool:
        buckets[(inc["source"], inc["app"], inc["label"]["fault"])].append((split, inc))
    keys = sorted(buckets)
    for k in keys:
        rng.shuffle(buckets[k])
    picked = []
    while len(picked) < size and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(picked) < size:
                picked.append(buckets[k].pop())
    return picked


def _derived(cfg, full):
    d = bdir(cfg)
    write_jsonl(d / "incidents_full.jsonl", full)
    mx = int(cfg["models"].get("compressor_sft", {}).get("max_input_chars", 12000))
    write_jsonl(d / "compressor.jsonl", [{"id": i["incident_id"], "input": render_window(i, mx), "target": render_signals(i["signals"])} for i in full])
    corpus = read_jsonl(wpath(cfg, "data", "retrieval", "corpus.jsonl"))
    qs = []
    for inc in full:
        rel = retrieval.relevant(inc, corpus, cfg["retrieval"].get("max_code_positives", 3))
        if rel:
            qs.append({"id": f"q:{inc['incident_id']}", "incident_id": inc["incident_id"], "text": signals_query(inc["signals"], k=cfg["retrieval"].get("query_anomalies", 5)), "relevant": rel, "fault": inc["label"]["fault"], "service": inc["label"]["service"]})
    write_jsonl(d / "queries.jsonl", qs)
    rng = random.Random(f"bench_claims:{cfg.get('seed', 0)}")
    rows = [r for inc in full for r in claims.build_for_incident(inc, rng, int(cfg["claims"].get("false_per_item", 2)))]
    write_jsonl(d / "claims.jsonl", rows)


def step_freeze(cfg):
    from .data import load_split
    from .hub import api, push_folder

    b = cfg["bench"]
    tag = b["version"]
    repo = dataset_repo(cfg, "bench")
    pub = wdir(cfg, "bench_public")
    if b.get("reuse", True) and _tag_exists(cfg, tag):
        from huggingface_hub import hf_hub_download

        for f in PUBLIC_FILES:
            shutil.copy(hf_hub_download(repo, f, repo_type="dataset", revision=tag), pub / f)
        labels = {r["incident_id"]: r for r in read_jsonl(pub / "labels.jsonl")}
        full = []
        for inc in read_jsonl(pub / "incidents.jsonl"):
            lab = labels[inc["incident_id"]]
            full.append({**inc, "label": {"service": lab["service"], "fault": lab["fault"], "detail": lab.get("detail", lab["fault"])}})
        _derived(cfg, full)
        log(f"bench {tag} reused from {repo}: {len(full)} incidents")
        return
    if _tag_exists(cfg, tag):
        raise SystemExit(f"{repo} already has tag {tag}. Bump bench.version to publish a new benchmark.")
    pool = [(s, inc) for s in POOL for inc in load_split(cfg, s)]
    picked = select(cfg, pool)
    full = [inc for _, inc in picked]
    write_jsonl(pub / "incidents.jsonl", [public(i) for i in full])
    write_jsonl(pub / "labels.jsonl", [{"incident_id": i["incident_id"], "service": i["label"]["service"], "fault": i["label"]["fault"], "detail": i["label"].get("detail", ""), "app": i["app"], "source": i["source"], "origin_split": s} for s, i in picked])
    man = {
        "name": repo,
        "version": tag,
        "size": len(full),
        "requested_size": int(b["size"]),
        "by_source": dict(Counter(i["source"] for i in full)),
        "by_app": dict(Counter(i["app"] for i in full)),
        "by_fault": dict(Counter(i["label"]["fault"] for i in full)),
        "by_origin_split": dict(Counter(s for s, _ in picked)),
        "rule": "drawn only from test, test_ood_fault and test_ood_app; no incident group of the benchmark is in train or val",
        "files_sha256": {f: sha256_files([pub / f]) for f in ("incidents.jsonl", "labels.jsonl")},
        "meta": meta(cfg),
    }
    write_json(pub / "manifest.json", man)
    (pub / "README.md").write_text(card(cfg, man))
    push_folder(repo, pub, repo_type="dataset", private=bool(cfg["hub"].get("private", False)), message=f"Freeze benchmark {tag}")
    api().create_tag(repo, tag=tag, repo_type="dataset", tag_message=f"FaultLine Bench {tag}")
    _derived(cfg, full)
    log(f"bench {tag} frozen and tagged: {len(full)} incidents, {man['by_origin_split']}")


def leak_guard(cfg):
    from .data import load_split

    bench = load_split(cfg, "bench")
    ids = {i["incident_id"] for i in bench}
    groups = {group_key(i) for i in bench}
    for s in ("train", "val"):
        rows = load_split(cfg, s)
        if ids & {i["incident_id"] for i in rows}:
            raise SystemExit(f"bench incidents found in {s}")
        shared = groups & {group_key(i) for i in rows}
        if shared:
            raise SystemExit(f"bench incident groups found in {s}: {sorted(shared)[:3]}")
    log("bench leak guard PASSED")
    return len(bench)


def _trained(cfg, name):
    return name in cfg["models"] and (results_path(cfg, name, "train").exists() or results_path(cfg, name, "push").exists())


def _ref(cfg, name, local_dir=False):
    t = results_path(cfg, name, "train")
    if t.exists():
        local = read_json(t)["local"]
        if Path(local).exists():
            return local
    rid = repo_id(cfg, name)
    if local_dir:
        from huggingface_hub import snapshot_download

        return snapshot_download(rid)
    return rid


def _pick(res, keys):
    if not res:
        return None
    s = res["summary"]
    return {k: s.get(k) for k in keys if k in s}


def _preds(cfg, system, items):
    write_jsonl(wpath(cfg, "results", "bench", f"pred_{system}.jsonl"), [{"incident_id": it["id"], "services": it.get("ranked", []), "fault": it.get("pred_fault")} for it in items])


def step_run(cfg):
    from .models import classifier, compressor, embedder, verifier

    n = leak_guard(cfg)
    out = {"version": cfg["bench"]["version"], "repo": dataset_repo(cfg, "bench"), "n": n, "systems": {}, "meta": meta(cfg)}
    cname = next((c for c in cfg["bench"]["compressor"] if _trained(cfg, c)), None)
    comp = None
    if cname:
        comp = compressor.evaluate(cfg, cname, _ref(cfg, cname), "bench")
        out["systems"][f"{cname}_signals"] = _pick(comp, ["signal_f1", "signal_f1_ci95", "valid_json", "token_ratio", "token_ratio_ci95", "n"])
    runs = [("rule_on_rule_signals", "rule", "reference")]
    if _trained(cfg, "classifier"):
        clf = _ref(cfg, "classifier", local_dir=True)
        runs.append(("classifier_on_rule_signals", clf, "reference"))
        if comp:
            runs.append(("end_to_end", clf, cname))
    if comp:
        runs.append(("rule_on_compressor_signals", "rule", cname))
    for system, ref, src in runs:
        r = classifier.evaluate(cfg, "classifier", ref, "bench", source=src)
        out["systems"][system] = _pick(r, ["top1", "top1_ci95", "top3", "top3_ci95", "fault_acc", "fault_acc_ci95", "n", "ms_per_incident"])
        _preds(cfg, system, r["items"])
    if "embedder" in cfg["models"]:
        out["systems"]["embedder_base"] = _pick(embedder.evaluate(cfg, "embedder", cfg["models"]["embedder"]["base"], "bench"), ["recall@10", "recall@10_ci95", "ndcg@10", "n"])
        if _trained(cfg, "embedder"):
            out["systems"]["embedder_trained"] = _pick(embedder.evaluate(cfg, "embedder", _ref(cfg, "embedder"), "bench"), ["recall@10", "recall@10_ci95", "ndcg@10", "n"])
    out["systems"]["verifier_rule"] = _pick(verifier.evaluate(cfg, "verifier", "rule", "bench"), ["correct", "catch", "catch_ci95", "false_reject", "n"])
    if _trained(cfg, "verifier"):
        out["systems"]["verifier_trained"] = _pick(verifier.evaluate(cfg, "verifier", _ref(cfg, "verifier"), "bench"), ["correct", "catch", "catch_ci95", "false_reject", "n"])
    out["headline"] = headline_numbers(out, float(cfg["bench"].get("max_end_to_end_drop", 0.05)))
    write_json(wpath(cfg, "results", "bench", "bench_results.json"), out)
    log("bench results", json.dumps(out["headline"]))


def _pct(x):
    return None if x is None else round(100 * float(x), 1)


def headline_numbers(out, max_drop=0.05):
    s = out["systems"]
    main = s.get("end_to_end") or s.get("classifier_on_rule_signals") or s.get("rule_on_rule_signals") or {}
    comp = next((v for k, v in s.items() if k.endswith("_signals") and k.startswith("compressor") and v), None)
    e2e, ref = s.get("end_to_end"), s.get("classifier_on_rule_signals")
    comp_ok = bool(comp and e2e and ref and e2e.get("top1") is not None and e2e["top1"] >= ref["top1"] - max_drop)
    if comp and not comp_ok:
        comp = None
    return {
        "compressor_keeps_accuracy": comp_ok,
        "system": "end_to_end" if s.get("end_to_end") else ("classifier_on_rule_signals" if s.get("classifier_on_rule_signals") else "rule_on_rule_signals"),
        "n": out["n"],
        "top1": _pct(main.get("top1")),
        "top3": _pct(main.get("top3")),
        "token_reduction": _pct(1 - comp["token_ratio"]) if comp and comp.get("token_ratio") is not None else None,
        "recall10_base": _pct((s.get("embedder_base") or {}).get("recall@10")),
        "recall10_trained": _pct((s.get("embedder_trained") or {}).get("recall@10")),
        "verifier_catch": _pct((s.get("verifier_trained") or s.get("verifier_rule") or {}).get("catch")),
        "uncited_claims": None,
    }


def card(cfg, man):
    repo = dataset_repo(cfg, "bench")
    return "\n".join([
        "---",
        "license: mit",
        "tags: [faultline, benchmark, root-cause-analysis, sre]",
        "---",
        f"# {repo.split('/')[1]} {man['version']}",
        "",
        f"A frozen benchmark of {man['size']} labelled microservice incidents for root cause analysis.",
        "Each incident has metrics, logs and spans from a time window around a fault, and the true root cause service and fault kind.",
        "",
        "Every incident comes from the test splits of the FaultLine model lab. No incident group (app, service, fault) of this benchmark is in any training or validation data.",
        "",
        "Files:",
        "",
        "* `incidents.jsonl`: model inputs, no labels",
        "* `labels.jsonl`: the answer key",
        "* `manifest.json`: counts, file hashes, commit, config and seed",
        "",
        f"Load this exact version with `revision=\"{man['version']}\"`.",
        "",
        "Score your own predictions:",
        "",
        "```bash",
        "git clone https://github.com/amareshhebbar/FaultLine && cd FaultLine/models",
        f"python -m flm.bench score --pred my_preds.jsonl --repo {repo} --revision {man['version']}",
        "```",
        "",
        "Each prediction line: `{\"incident_id\": \"...\", \"services\": [\"best\", \"second\", \"third\"], \"fault\": \"latency\"}`.",
        "",
        f"Counts by source: {man['by_source']}",
        f"Counts by app: {man['by_app']}",
        f"Counts by fault: {man['by_fault']}",
        "",
        "Sources: a seeded incident simulator from the FaultLine lab, and RCAEval (MIT license, https://github.com/phamquiluan/RCAEval).",
        "",
    ])


def _row(name, v, keys):
    cells = []
    for k in keys:
        x = v.get(k)
        ci = v.get(f"{k}_ci95")
        cells.append("" if x is None else (f"{x} ({ci[0]} to {ci[1]})" if ci else f"{x}"))
    return f"| {name} | " + " | ".join(cells) + f" | {v.get('n', '')} |"


def links(cfg):
    from .common import repo_id as rid

    ns = namespace(cfg)
    coll_p = wpath(cfg, "results", "collection.json")
    return {
        "bench": f"https://huggingface.co/datasets/{dataset_repo(cfg, 'bench')}",
        "proof": bench_url(cfg, "PROOF.md"),
        "runs": f"https://huggingface.co/datasets/{dataset_repo(cfg, 'runs')}",
        "collection": read_json(coll_p)["url"] if coll_p.exists() else f"https://huggingface.co/{ns}",
        "github": cfg["bench"]["resume"]["github"],
        "models": {n: f"https://huggingface.co/{rid(cfg, n)}" for n, m in cfg["models"].items() if m.get("enabled", True)},
        "datasets": {k: f"https://huggingface.co/datasets/{dataset_repo(cfg, k)}" for k in cfg["hub"]["datasets"] if k not in ("bench", "runs")},
    }


def full_card(cfg, res, man, latex, run_link):
    L = links(cfg)
    s, h = res["systems"], res["headline"]
    head = [
        "---",
        "license: mit",
        f"pretty_name: FaultLine Bench {res['version']}",
        "tags: [faultline, benchmark, root-cause-analysis, sre, observability, microservices]",
        "size_categories: [n<1K]",
        "configs:",
        "- config_name: incidents",
        "  data_files: incidents.jsonl",
        "- config_name: labels",
        "  data_files: labels.jsonl",
        "---",
        "",
    ]
    body = card(cfg, man).split("---\n", 2)[-1].splitlines()
    body = [x for x in body if not x.startswith("# ")]
    lines = head + [
        f"# FaultLine Bench {res['version']}",
        "",
        f"[Proof for every number]({L['proof']}) · [Run logs]({run_link}) · [Models]({L['collection']}) · [Code]({L['github']})",
        "",
        "## Headline results",
        "",
        "| result | value |",
        "| --- | --- |",
        f"| failing service named correctly (top 1) | **{h['top1']}%** of {h['n']} incidents |",
        f"| failing service in top 3 | **{h['top3']}%** |",
    ]
    if h.get("token_reduction") is not None:
        lines.append(f"| telemetry window shrink by the compressor | **{h['token_reduction']}%** fewer tokens |")
    if h.get("recall10_trained") is not None:
        lines.append(f"| embedder Recall@10, base to fine tuned | **{h['recall10_base']}%** to **{h['recall10_trained']}%** |")
    if h.get("verifier_catch") is not None:
        lines.append(f"| broken claims caught by the verifier | **{h['verifier_catch']}%** |")
    lines += ["| uncited claims in agent reports | not measured yet, needs the FaultLine agent |", ""]
    lines += ["## Leaderboard", "", "| system | top1 | top3 | fault_acc | n |", "| --- | --- | --- | --- | --- |"]
    for k in ("end_to_end", "classifier_on_rule_signals", "rule_on_compressor_signals", "rule_on_rule_signals"):
        if s.get(k):
            lines.append(_row(k, s[k], ["top1", "top3", "fault_acc"]))
    lines += ["", "end_to_end means the fine tuned compressor reads the raw window and the classifier ranks services from its output. Numbers in brackets are 95% bootstrap intervals.", ""]
    lines += ["## Models", ""] + [f"* [{n}]({u})" for n, u in L["models"].items()] + [""]
    lines += ["## Datasets", ""] + [f"* [{k}]({u})" for k, u in L["datasets"].items()] + [f"* [run logs and results]({L['runs']})", ""]
    lines += body
    lines += ["## Resume line", "", "```latex", latex, "```", ""]
    return "\n".join(lines)


def step_publish(cfg):
    from .headline import build_headline
    from .hub import push_folder, push_text, runs_url
    from .proof import build_proof

    res = read_json(wpath(cfg, "results", "bench", "bench_results.json"))
    rep = build_headline(cfg)
    build_proof(cfg, rep)
    latex = read_json(rep / "proof.json")["latex"]
    man_p = wpath(cfg, "bench_public", "manifest.json")
    man = read_json(man_p) if man_p.exists() else {"version": res["version"], "size": res["n"], "by_source": {}, "by_app": {}, "by_fault": {}}
    readme = full_card(cfg, res, man, latex, runs_url(cfg))
    (rep / "BENCH_README.md").write_text(readme)
    (rep / "resume_line.tex").write_text(latex + "\n")
    up = wdir(cfg, "bench_upload")
    shutil.rmtree(up, ignore_errors=True)
    shutil.copytree(wdir(cfg, "results", "bench"), up / "results" / res["version"])
    for f in ("PROOF.md", "proof.json", "HEADLINE.md", "headline.json"):
        if (rep / f).exists():
            shutil.copy(rep / f, up / f)
    repo = dataset_repo(cfg, "bench")
    push_folder(repo, up, repo_type="dataset", message=f"Results and proof for {res['version']}")
    push_text(repo, "README.md", readme, repo_type="dataset", message="Benchmark card with results")
    shutil.rmtree(up, ignore_errors=True)
    log("benchmark card", links(cfg)["bench"])
    log("proof page", links(cfg)["proof"])
    log("RESUME LINE:")
    log(latex)


def score(preds, labels):
    lab = {r["incident_id"]: r for r in labels}
    p = {r["incident_id"]: r for r in preds}
    missing = [i for i in lab if i not in p]
    t1, t3, fa = [], [], []
    for i, l in lab.items():
        r = p.get(i, {"services": [], "fault": None})
        svcs = r.get("services") or []
        t1.append(1.0 if svcs[:1] == [l["service"]] else 0.0)
        t3.append(1.0 if l["service"] in svcs[:3] else 0.0)
        if r.get("fault") is not None:
            fa.append(1.0 if r["fault"] == l["fault"] else 0.0)
    return {
        "n": len(lab),
        "missing_predictions": len(missing),
        "top1": round(mean(t1), 4),
        "top1_ci95": bootstrap_ci(t1),
        "top3": round(mean(t3), 4),
        "top3_ci95": bootstrap_ci(t3),
        "fault_acc": round(mean(fa), 4) if fa else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m flm.bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--pred", required=True)
    sc.add_argument("--labels")
    sc.add_argument("--repo", default="AmareshHebbar/FaultLine_Bench")
    sc.add_argument("--revision", default="v1.0")
    a = ap.parse_args(argv)
    if a.labels:
        labels = read_jsonl(a.labels)
    else:
        from huggingface_hub import hf_hub_download

        labels = read_jsonl(hf_hub_download(a.repo, "labels.jsonl", repo_type="dataset", revision=a.revision))
    print(json.dumps(score(read_jsonl(a.pred), labels), indent=2))


if __name__ == "__main__":
    main()
