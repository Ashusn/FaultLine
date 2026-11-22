from .common import dataset_repo, read_json, repo_id, results_path, wpath, write_json


def hf_url(repo, path, dataset=False):
    return f"https://huggingface.co/{'datasets/' if dataset else ''}{repo}/blob/main/{path}"


def proof_url(cfg, anchor=None):
    u = hf_url(dataset_repo(cfg, "bench"), "PROOF.md", dataset=True)
    return f"{u}#{anchor}" if anchor else u


def _pick_split(res, c):
    for sp in [c.get("split")] + list(c.get("fallback_splits", [])):
        if sp and sp in res.get("splits", {}):
            return sp
    return None


def claim_value(cfg, c):
    if c.get("source") == "bench":
        p = wpath(cfg, "results", "bench", "bench_results.json")
        if not p.exists():
            return None
        res = read_json(p)
        h = res.get("headline", {})
        if h.get(c["key"]) is None:
            return None
        sysname = c.get("system") or h.get("system")
        sysres = res.get("systems", {}).get(sysname, {}) if sysname else {}
        ci = sysres.get(f"{c.get('metric', '')}_ci95")
        ci = [round(x * 100, 1) for x in ci] if ci and c.get("kind", "percent") == "percent" else None
        return {"value": h[c["key"]], "ci95": ci, "n": h.get("n"), "split": f"FaultLine Bench {res['version']}", "base": None, "source": hf_url(dataset_repo(cfg, "bench"), f"results/{res['version']}/bench_results.json", dataset=True)}
    if c.get("source") == "external":
        p = wpath(cfg, "results", c["file"])
        if not p.exists():
            return None
        d = read_json(p)
        if c["key"] not in d:
            return None
        return {"value": d[c["key"]], "ci95": d.get(f"{c['key']}_ci95"), "n": d.get("n"), "split": d.get("split", "agent eval"), "source": hf_url(dataset_repo(cfg, "incidents"), f"report/{c['file']}", dataset=True), "base": None}
    full_p, base_p = results_path(cfg, c["model"], "full"), results_path(cfg, c["model"], "base_full")
    if not full_p.exists():
        return None
    res = read_json(full_p)
    sp = _pick_split(res, c)
    if not sp:
        return None
    r = res["splits"][sp]
    v, ci = r.get(c["metric"]), r.get(f"{c['metric']}_ci95")
    if v is None:
        return None
    base = read_json(base_p)["splits"].get(sp, {}).get(c["metric"]) if base_p.exists() else None
    kind = c.get("kind", "percent")
    if kind == "percent":
        value, ci = round(v * 100, 1), [round(x * 100, 1) for x in ci] if ci else None
    elif kind == "one_minus_percent":
        value, ci = round((1 - v) * 100, 1), [round((1 - ci[1]) * 100, 1), round((1 - ci[0]) * 100, 1)] if ci else None
    elif kind == "relative_gain_percent":
        if not base:
            return None
        value, ci = round((v / base - 1) * 100, 1), None
    else:
        value = v
    return {"value": value, "ci95": ci, "n": r.get("n"), "split": sp, "raw": v, "base": base, "source": hf_url(repo_id(cfg, c["model"]), f"eval/{c['model']}_full.json")}


def build_proof(cfg, out_dir):
    p = cfg.get("proof", {})
    claims = []
    for c in p.get("claims", []):
        v = claim_value(cfg, c)
        claims.append({"id": c["id"], "text": c["text"], "model": c.get("model"), "metric": c.get("metric", c.get("key")), "kind": c.get("kind", "percent"), "result": v})
    lines = [
        "# FaultLine: proof for every number",
        "",
        "Each number below is read from a results file that is public on HuggingFace. Click the source to see the file. Run the reproduce command to compute it again yourself.",
        "",
        "```bash",
        "git clone https://github.com/amareshhebbar/FaultLine && cd FaultLine/models && bash setup.sh",
        "FLM_CONFIG=configs/config.yaml FLM_WORK=/tmp/faultline_reproduce .venv/bin/python run.py reproduce",
        "```",
        "",
    ]
    for c in claims:
        r = c["result"]
        lines.append(f'<a id="{c["id"]}"></a>')
        lines.append(f"## {c['id']}")
        lines.append("")
        if not r:
            lines += ["Not measured yet.", ""]
            continue
        lines.append(c["text"].format(value=r["value"], n=r.get("n")))
        lines.append("")
        lines.append("| value | 95% interval | n | split | base | source |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        lines.append(f"| {r['value']} | {r.get('ci95') or 'n/a'} | {r.get('n')} | {r['split']} | {r.get('base') if r.get('base') is not None else 'n/a'} | [file]({r['source']}) |")
        lines.append("")
        if c.get("model"):
            lines.append(f"Reproduce only this one: `CLAIMS={c['id']} .venv/bin/python run.py reproduce`")
            lines.append("")
    vals = {c["id"]: (c["result"]["value"] if c["result"] else None) for c in claims}
    import re

    latex = p.get("latex", "")

    def _optional(m):
        seg = m.group(1)
        ids = re.findall(r"\{([a-z0-9_]+?)(?:_n)?\}", seg)
        return "" if any(i in vals and vals[i] is None for i in ids) else seg

    latex = re.sub(r"\[\[(.*?)\]\]", _optional, latex, flags=re.S)
    for c in claims:
        n = c["result"].get("n") if c["result"] else None
        latex = latex.replace("{" + c["id"] + "_n}", str(n) if n else "X")
    for cid, v in vals.items():
        shown = "X" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))
        latex = latex.replace("{" + cid + "}", "\\href{" + proof_url(cfg, cid) + "}{" + shown + "}")
    latex = latex.replace("{proof}", proof_url(cfg))
    lines += ["## Resume line", "", "```latex", latex.strip(), "```", ""]
    (out_dir / "PROOF.md").write_text("\n".join(lines))
    write_json(out_dir / "proof.json", {"claims": claims, "latex": latex.strip()})
    for c in p.get("claims", []):
        if c.get("source") == "external":
            src = wpath(cfg, "results", c["file"])
            if src.exists():
                (out_dir / c["file"]).write_text(src.read_text())
    return out_dir
