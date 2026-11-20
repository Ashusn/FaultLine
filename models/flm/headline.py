from .common import read_json, repo_id, results_path, wdir, write_json


def change(label, old, new, unit=""):
    if old is None or new is None:
        return f"{label}: {new}{unit}"
    verb = "raises" if new > old else "lowers" if new < old else "keeps"
    if verb == "keeps":
        return f"keeps {label} at {new}{unit}"
    return f"{verb} {label} from {old}{unit} to {new}{unit}"


def _row(cfg, name):
    m = cfg["models"][name]
    key = m["main_metric"]
    full_p, base_p, ver_p = results_path(cfg, name, "full"), results_path(cfg, name, "base_full"), results_path(cfg, name, "verify")
    if not full_p.exists() or not base_p.exists():
        return None
    full, base = read_json(full_p), read_json(base_p)
    ver = read_json(ver_p) if ver_p.exists() else {}
    out = {"model": name, "repo": repo_id(cfg, name), "metric": key, "verify_passed": ver.get("passed"), "splits": {}}
    for sp, r in full["splits"].items():
        b = base["splits"].get(sp, {})
        out["splits"][sp] = {"base": b.get(key), "model": r.get(key), "model_ci95": r.get(f"{key}_ci95"), "base_ci95": b.get(f"{key}_ci95"), "n": r.get("n")}
    return out


def build_headline(cfg):
    rows = [r for r in (_row(cfg, n) for n in cfg["models"]) if r]
    lines = ["# FaultLine models: eval page", "", "Every number below is read from a results file. Base means the starting point for that model: the base model, the rule baseline, or the previous stage.", ""]
    for r in rows:
        lines += [f"## {r['model']}", "", f"Repo: https://huggingface.co/{r['repo']}", f"Verify gate: {'PASSED' if r['verify_passed'] else 'not passed or not run'}", ""]
        lines += ["| split | base | model | model 95% interval | n |", "| --- | --- | --- | --- | --- |"]
        for sp, v in r["splits"].items():
            lines.append(f"| {sp} | {v['base']} | {v['model']} | {v['model_ci95']} | {v['n']} |")
        lines.append("")
        for sp, v in r["splits"].items():
            lines.append(f"* {sp}: {change(r['metric'], v['base'], v['model'])}")
        lines.append("")
    study_p = results_path(cfg, "classifier", "study")
    if study_p.exists():
        lines += ["## Agent signals versus classifier study", "", "```json", __import__("json").dumps(read_json(study_p), indent=2), "```", ""]
    data_p = results_path(cfg, "data", "report")
    if data_p.exists():
        d = read_json(data_p)
        lines += ["## Data", "", f"Split sizes: {d.get('sizes')}", f"Leak check: {'PASSED' if d.get('ok') else 'FAILED'}", ""]
    out = wdir(cfg, "report")
    (out / "HEADLINE.md").write_text("\n".join(lines))
    write_json(out / "headline.json", rows)
    return out
