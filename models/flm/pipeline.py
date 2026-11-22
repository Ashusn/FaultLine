import os
import shutil

from .common import log, meta, read_json, repo_id, results_path, wdir, wpath, write_json
from .gates import relative_gate
from .models import module


def base_ref(cfg, name):
    m = cfg["models"][name]
    if m.get("init_from"):
        from .models.compressor import init_ref

        return init_ref(cfg, name)
    if m["type"] in ("classifier", "verifier"):
        return "rule"
    if m["type"] == "reranker" and m.get("base_is_first_stage"):
        return "first_stage"
    return m["base"]


def quick_n(cfg):
    return int(cfg["eval"]["quick_n"])


def full_splits(cfg):
    return cfg["eval"]["full_splits"]


def _full(cfg, name, ref):
    mod = module(cfg, name)
    out = {"ref": ref, "splits": {}, "meta": meta(cfg)}
    for sp in full_splits(cfg):
        res = mod.evaluate(cfg, name, ref, sp, cfg["eval"].get("full_n"))
        if res:
            out["splits"][sp] = res["summary"]
            label = "trained" if os.path.exists(ref) else ref.replace("/", "_")
            write_json(wpath(cfg, "results", "items", f"{name}_{label}_{sp}.json"), res["items"])
    return out


def step_baseline(cfg, name):
    mod = module(cfg, name)
    ref = base_ref(cfg, name)
    q = mod.evaluate(cfg, name, ref, "val", quick_n(cfg))
    if not q:
        raise SystemExit(f"{name}: no validation data")
    write_json(results_path(cfg, name, "base_quick"), {"ref": ref, "summary": q["summary"], "items": q["items"], "meta": meta(cfg)})
    write_json(results_path(cfg, name, "base_full"), _full(cfg, name, ref))
    extra = list(cfg["models"][name].get("baselines", []))
    if cfg["models"][name]["type"] == "reranker":
        extra = [{"id": "first_stage", "label": "first_stage"}] + extra
    for b in extra:
        if b["id"] == ref:
            continue
        r = _full(cfg, name, b["id"])
        write_json(results_path(cfg, name, f"baseline_{b.get('label', b['id'].split('/')[-1])}_full"), r)
    log(f"{name} baseline", q["summary"])


def step_train(cfg, name):
    out = module(cfg, name).train(cfg, name)
    write_json(results_path(cfg, name, "train"), {**out, "meta": meta(cfg)})
    log(f"{name} trained", out["local"])


def step_push(cfg, name):
    from .hub import push_model

    t = read_json(results_path(cfg, name, "train"))
    extra = {"training": t["training"]} if t.get("training") else {}
    repo = push_model(cfg, name, t["local"], adapter_dir=t.get("adapter"), extra_dirs=extra)
    write_json(results_path(cfg, name, "push"), {"repo": repo, "meta": meta(cfg)})


def step_verify(cfg, name):
    from .hub import download_model

    local = wdir(cfg, "verify", name)
    try:
        download_model(cfg, name, local)
        res = module(cfg, name).evaluate(cfg, name, str(local), "val", quick_n(cfg))
    finally:
        shutil.rmtree(local, ignore_errors=True)
    base = read_json(results_path(cfg, name, "base_quick"))
    gate = relative_gate(cfg, name, res["items"], base["items"])
    report = {"repo": repo_id(cfg, name), "quick": res["summary"], "base_quick": base["summary"], "gate": gate, "passed": gate["passed"], "meta": meta(cfg)}
    write_json(results_path(cfg, name, "verify"), report)
    if gate.get("small_sample_warning"):
        log(f"verify {name}: only {gate['n']} items, below min_quick_items")
    if not gate["passed"]:
        raise SystemExit(f"verify {name} FAILED: delta {gate['delta']} ci {gate['delta_ci95']} allowed drop {gate['allowed_drop']}")
    log(f"verify {name} PASSED: delta {gate['delta']} ci {gate['delta_ci95']}")


def _model_ref(cfg, name):
    t = read_json(results_path(cfg, name, "train"))
    return t["local"] if os.path.exists(t["local"]) else repo_id(cfg, name)


def step_eval(cfg, name):
    from .hub import push_folder

    ref = _model_ref(cfg, name)
    res = _full(cfg, name, ref)
    write_json(results_path(cfg, name, "full"), res)
    if cfg["models"][name]["type"] == "classifier":
        from .models.classifier import study

        res["study"] = study(cfg, name, ref)
        write_json(results_path(cfg, name, "full"), res)
    ev = wdir(cfg, "eval_upload", name)
    shutil.rmtree(ev, ignore_errors=True)
    ev.mkdir(parents=True)
    for p in wdir(cfg, "results").glob(f"{name}_*"):
        if p.is_file():
            shutil.copy(p, ev / p.name)
    items = [p for p in wdir(cfg, "results", "items").glob(f"{name}_*.json")]
    if items:
        (ev / "items").mkdir()
        for p in items:
            shutil.copy(p, ev / "items" / p.name)
    push_folder(repo_id(cfg, name), ev, path_in_repo="eval", message="Upload eval results")
    shutil.rmtree(ev, ignore_errors=True)
    log(f"{name} eval", {k: v.get(cfg["models"][name]["main_metric"]) for k, v in res["splits"].items()})


def step_publish(cfg, name):
    from .card import model_card
    from .hub import add_to_collection, push_text

    repo = repo_id(cfg, name)
    push_text(repo, "README.md", model_card(cfg, name), message="Model card with results")
    logs = [p for p in wdir(cfg, "logs").glob(f"{name}__*.log")]
    if logs:
        from .hub import push_folder

        ld = wdir(cfg, "logs_upload", name)
        for p in logs:
            shutil.copy(p, ld / p.name)
        push_folder(repo, ld, path_in_repo="logs", message="Training and eval logs")
        shutil.rmtree(ld, ignore_errors=True)
    add_to_collection(cfg, [(repo, "model")])
    log(f"{name} published", repo)


def step_report(cfg):
    from .common import dataset_repo
    from .hub import add_to_collection, push_folder
    from .headline import build_headline

    out = build_headline(cfg)
    push_folder(dataset_repo(cfg, "incidents"), out, path_in_repo="report", repo_type="dataset", message="Eval page")
    add_to_collection(cfg, [(dataset_repo(cfg, k), "dataset") for k in cfg["hub"]["datasets"]])
    if os.environ.get("KEEP_WEIGHTS") != "1":
        for d in wdir(cfg, "models").iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
        log("local weights deleted")


MODEL_STEPS = {"baseline": step_baseline, "train": step_train, "push": step_push, "verify": step_verify, "eval": step_eval, "publish": step_publish}
