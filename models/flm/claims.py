import random
import re

PERTURB = ["wrong_service", "wrong_number", "wrong_direction", "wrong_citation", "invented_id", "unsupported_cause"]
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def evidence_items(inc):
    items = []
    for a in inc["signals"]["anomalies"]:
        eid = f"E{len(items) + 1}"
        if a["signal"] == "trace_errors" and a.get("span_id"):
            text = f"[{eid}] span {a['span_id']} in trace {a['trace_id']} belongs to {a['service']}; origin error spans for {a['service']} went from {a['before']} to {a['after']}"
        elif a["signal"] == "log_errors":
            text = f"[{eid}] {a['service']} error log lines went from {a['before']} to {a['after']}; example: {a.get('example', '')[:120]}"
        else:
            text = f"[{eid}] metric {a['service']} {a['metric']} ({a['signal']}) mean before={a['before']} after={a['after']}"
        items.append({"id": eid, "service": a["service"], "signal": a["signal"], "before": a["before"], "after": a["after"], "span_id": a.get("span_id", ""), "text": text})
    return items


def _true_claim(it):
    verb = "rose" if float(it["after"]) >= float(it["before"]) else "fell"
    sig = it["signal"].replace("_", " ")
    return f"{it['service']} {sig} {verb} from {it['before']} to {it['after']} [{it['id']}]."


def _fmt_num(x):
    return float(f"{x:.3g}")


def perturb(kind, it, items, services, rng):
    sig = it["signal"].replace("_", " ")
    up = float(it["after"]) >= float(it["before"])
    if kind == "wrong_service":
        others = [s for s in services if s != it["service"]]
        if not others:
            return None
        return f"{rng.choice(others)} {sig} {'rose' if up else 'fell'} from {it['before']} to {it['after']} [{it['id']}]."
    if kind == "wrong_number":
        f = rng.choice([0.2, 0.4, 2.5, 5.0, 10.0])
        return f"{it['service']} {sig} {'rose' if up else 'fell'} from {it['before']} to {_fmt_num(float(it['after']) * f + (1 if float(it['after']) == 0 else 0))} [{it['id']}]."
    if kind == "wrong_direction":
        return f"{it['service']} {sig} {'fell' if up else 'rose'} from {it['before']} to {it['after']} [{it['id']}]."
    if kind == "wrong_citation":
        others = [o for o in items if o["service"] != it["service"]]
        if not others:
            return None
        o = rng.choice(others)
        return f"{it['service']} {sig} {'rose' if up else 'fell'} from {it['before']} to {it['after']} [{o['id']}]."
    if kind == "invented_id":
        return f"{it['service']} {sig} {'rose' if up else 'fell'} from {it['before']} to {it['after']} [E{len(items) + rng.randint(5, 40)}]."
    if kind == "unsupported_cause":
        others = [s for s in services if s != it["service"]]
        if not others:
            return None
        return f"{it['service']} {sig} changed because {rng.choice(others)} crashed [{it['id']}]."
    raise ValueError(kind)


def resolve(claim, items):
    by = {i["id"]: i for i in items}
    ids = re.findall(r"\[(E\d+)\]", claim)
    texts = [by[i]["text"] if i in by else f"[{i}] MISSING" for i in ids]
    return "\n".join(texts) if texts else "NO CITATION"


def build_for_incident(inc, rng, per_item_false=1):
    items = evidence_items(inc)
    rows = []
    for it in items:
        c = _true_claim(it)
        rows.append({"claim": c, "evidence": resolve(c, items), "label": 1, "kind": "true"})
        for k in rng.sample(PERTURB, min(per_item_false, len(PERTURB))):
            fc = perturb(k, it, items, inc["services"], rng)
            if fc:
                rows.append({"claim": fc, "evidence": resolve(fc, items), "label": 0, "kind": k})
    for i, r in enumerate(rows):
        r["id"] = f"c:{inc['incident_id']}:{i}"
        r["incident_id"] = inc["incident_id"]
    return rows


def build(cfg, splits):
    rng = random.Random(cfg.get("seed", 0))
    k = int(cfg["claims"].get("false_per_item", 2))
    return {name: [r for inc in incs for r in build_for_incident(inc, rng, k)] for name, incs in splits.items()}


def rule_check(claim, evidence):
    if "NO CITATION" in evidence or "MISSING" in evidence:
        return 0
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s", claim)
    if m and m.group(1).lower() not in evidence.lower():
        return 0
    body = re.sub(r"\[E\d+\]", "", claim)
    ev_nums = [float(x) for x in NUM.findall(re.sub(r"\[E\d+\]", "", evidence))]
    for x in NUM.findall(body):
        v = float(x)
        if not any(abs(v - e) <= max(1e-9, 0.05 * abs(e)) for e in ev_nums):
            return 0
    return 1
