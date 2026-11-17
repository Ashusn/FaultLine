import json


def _rel(t, t0):
    d = int(t) - int(t0)
    return f"+{d}s" if d >= 0 else f"{d}s"


def _stride(rows, n):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def render_window(inc, max_chars=12000):
    t0 = inc["inject_time"]
    head = [
        f"app: {inc['app']}",
        f"services: {', '.join(inc['services'])}",
        f"window: {_rel(inc['window']['start'], t0)} to {_rel(inc['window']['end'], t0)} around the alert at +0s",
    ]
    mlines = [
        f"{m['service']} {m['name']} before mean={m['before']['mean']} p95={m['before']['p95']} after mean={m['after']['mean']} p95={m['after']['p95']}"
        for m in inc["metrics"]
    ]
    llines = [f"{_rel(r['t'], t0)} {r['service']} {r.get('level', 'info')}: {r['message']}" for r in sorted(inc["logs"], key=lambda r: r["t"])]
    slines = [
        f"{_rel(s['t'], t0)} trace={s['trace_id'][:16]} span={s['span_id'][:16]} parent={(s.get('parent_id') or '-')[:16]} {s['service']} {s.get('operation', '')} {s['ms']}ms {'ERROR' if s['error'] else 'ok'}"
        for s in sorted(inc["spans"], key=lambda s: s["t"])
    ]
    budget = max(1000, max_chars - sum(len(x) + 1 for x in head) - 64)
    parts = {"metrics": (mlines, 0.4), "logs": (llines, 0.35), "spans": (slines, 0.25)}
    out = list(head)
    for name, (lines, share) in parts.items():
        cap = int(budget * share)
        avg = (sum(len(x) + 1 for x in lines) / len(lines)) if lines else 1
        keep = _stride(lines, max(1, int(cap / avg))) if lines else []
        out.append(f"## {name} ({len(keep)} of {len(lines)})")
        out.extend(keep)
    text = "\n".join(out)
    return text[:max_chars]


def render_signals(signals):
    return json.dumps(signals, separators=(",", ":"), sort_keys=True)


def signals_query(signals, k=5):
    parts = []
    for a in signals.get("anomalies", [])[:k]:
        sig = a["signal"].replace("_", " ")
        parts.append(f"{a['service']} {sig} {a['before']} to {a['after']}")
        if a.get("example"):
            parts.append(f"log: {a['example'][:120]}")
    return "; ".join(parts) or "no clear anomalies"
