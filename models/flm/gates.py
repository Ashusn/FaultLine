from .metrics import mean, paired_delta_ci


def gate_settings(cfg, name):
    e = cfg["eval"]
    m = cfg["models"][name]
    drop = e.get("gate_drop_by_model", {}).get(name, e["gate_drop"])
    return {"metric": m["main_metric"], "drop": float(drop), "min_items": int(e.get("min_quick_items", 100))}


def relative_gate(cfg, name, model_items, base_items):
    g = gate_settings(cfg, name)
    key = g["metric"]
    mid = {it["id"]: float(it[key]) for it in model_items}
    bid = {it["id"]: float(it[key]) for it in base_items}
    common = sorted(set(mid) & set(bid))
    if not common:
        return {"passed": False, "reason": "no shared items between model and base", **g}
    a = [mid[i] for i in common]
    b = [bid[i] for i in common]
    delta, ci = paired_delta_ci(a, b, seed=cfg.get("seed", 0))
    passed = ci[0] >= -g["drop"]
    warn = len(common) < g["min_items"]
    return {
        "passed": bool(passed),
        "metric": key,
        "model": round(mean(a), 4),
        "base": round(mean(b), 4),
        "delta": delta,
        "delta_ci95": ci,
        "allowed_drop": g["drop"],
        "n": len(common),
        "small_sample_warning": warn,
    }
