from pathlib import Path

from .common import dataset_repo, model_cfg, read_json, repo_id, results_path

DATASET_TEXT = {
    "incidents": "Labelled incident windows (metrics, logs, spans) with ground truth root cause service and fault kind, plus extracted signals. Splits are made by incident group (app, service, fault), so no group appears in two splits. test_ood_fault and test_ood_app hold out a whole fault kind and a whole app.",
    "compressor": "Pairs of a long rendered telemetry window (input) and its structured incident signals as JSON (target).",
    "retrieval": "Corpus of runbook sections, code chunks and past incidents, with symptom queries and their relevant documents.",
    "claims": "Evidence and claim pairs for claim verification. True claims are built from evidence; false claims change the service, the number, the direction, the citation, invent an id, or add a cause that is not in the evidence.",
}


def dataset_card(cfg, key, folder):
    files = sorted(p.name for p in Path(folder).glob("*.jsonl"))
    rows = []
    for f in files:
        with open(Path(folder) / f) as fh:
            rows.append(f"| {f} | {sum(1 for _ in fh)} |")
    return "\n".join([
        "---",
        "license: mit",
        "tags: [faultline, root-cause-analysis, sre, observability]",
        "---",
        f"# {dataset_repo(cfg, key).split('/')[1]}",
        "",
        DATASET_TEXT.get(key, ""),
        "",
        "Part of the FaultLine project. Sources: a seeded incident simulator, and RCAEval when enabled (MIT license, https://github.com/phamquiluan/RCAEval).",
        "",
        "| file | rows |",
        "| --- | --- |",
        *rows,
        "",
    ])


def _fmt_ci(v, ci):
    return f"{v} ({ci[0]} to {ci[1]})" if ci else f"{v}"


def model_card(cfg, name):
    m = model_cfg(cfg, name)
    res_p = results_path(cfg, name, "full")
    base_p = results_path(cfg, name, "base_full")
    res = read_json(res_p) if res_p.exists() else {}
    base = read_json(base_p) if base_p.exists() else {}
    key = m["main_metric"]
    lines = [
        "---",
        "license: apache-2.0",
        f"base_model: {m['base']}" if "/" in str(m.get("base", "")) else "",
        "tags: [faultline, root-cause-analysis, sre]",
        "---",
        f"# {repo_id(cfg, name).split('/')[1]}",
        "",
        m.get("description", ""),
        "",
        f"Base: `{m.get('base', 'rule baseline')}`. Method: {m.get('method', '')}.",
        f"Training data: `{dataset_repo(cfg, m.get('dataset', 'incidents'))}`.",
        "",
        "## Results",
        "",
        "| split | metric | base | this model |",
        "| --- | --- | --- | --- |",
    ]
    for split, r in res.get("splits", {}).items():
        b = base.get("splits", {}).get(split, {})
        for k in [key] + [x for x in m.get("report_metrics", []) if x != key]:
            if k in r:
                lines.append(f"| {split} | {k} | {_fmt_ci(b.get(k, 'n/a'), b.get(f'{k}_ci95'))} | {_fmt_ci(r[k], r.get(f'{k}_ci95'))} |")
    lines += ["", "Numbers in brackets are 95% bootstrap intervals. Every number comes from a results file in the eval folder of this repo.", ""]
    if m.get("trade_offs"):
        lines += ["## Trade offs", "", m["trade_offs"], ""]
    if m.get("usage"):
        lines += ["## Usage", "", "```python", m["usage"].replace("{repo}", repo_id(cfg, name)), "```", ""]
    if res.get("meta"):
        mt = res["meta"]
        lines += ["## Reproduce", "", f"commit `{mt.get('commit')}`, config `{mt.get('config')}`, seed {mt.get('seed')}, data hash `{mt.get('data_hash')}`.", ""]
    return "\n".join(x for x in lines if x is not None)
