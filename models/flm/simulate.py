import math
import random
import uuid

APPS = {
    "faultline": {
        "edges": {
            "gateway": ["orders"],
            "orders": ["inventory", "payments", "notifications"],
            "payments": ["ledger"],
            "inventory": [],
            "notifications": [],
            "ledger": [],
        },
        "db": ["orders", "payments", "inventory", "ledger"],
        "entry": "gateway",
    },
    "bookstore": {
        "edges": {
            "frontend": ["catalog", "cart"],
            "catalog": ["search", "pricing"],
            "cart": ["pricing", "checkout"],
            "checkout": ["billing"],
            "search": [],
            "pricing": [],
            "billing": [],
        },
        "db": ["catalog", "cart", "billing"],
        "entry": "frontend",
    },
}

SIM_FAULTS = ["latency", "error", "crash", "db_slow", "cpu_hog", "memory_leak", "packet_loss", "pool_exhaustion"]

INFO = [
    "request completed status=200 path=/{op} duration_ms={ms}",
    "handled {op} for customer cus_{n}",
    "cache hit key=item:{n}",
    "health check ok",
]

FAULT_LOGS = {
    "error": ["internal error while processing {op}: invalid state for item {n}", "request failed status=500 path=/{op}"],
    "crash": ["panic: runtime error: invalid memory address or nil pointer dereference", "process exited with code 2, restarting"],
    "db_slow": ["slow query detected duration_ms={ms} statement=UPDATE {svc}_rows", "db query exceeded threshold duration_ms={ms}"],
    "cpu_hog": ["request queue backlog length={n}", "worker pool saturated, waiting for free worker"],
    "memory_leak": ["gc pause duration_ms={ms} heap_mb={n}", "heap usage high heap_mb={n}"],
    "packet_loss": ["retrying request attempt=2 reason=connection reset by peer", "read tcp: i/o timeout"],
    "pool_exhaustion": ["timeout acquiring connection from pool after 5000ms", "connection pool exhausted active={n} max={n}"],
    "latency": [],
}

CALLER_LOGS = {
    "crash": "call to {dst} failed: dial tcp {dst}:8080: connect: connection refused",
    "error": "upstream {dst} returned status=500",
    "timeout": "call to {dst} failed: context deadline exceeded",
}


def _parents(edges):
    p = {s: [] for s in edges}
    for a, bs in edges.items():
        for b in bs:
            p[b].append(a)
    return p


def _ancestors(edges, svc):
    par = _parents(edges)
    seen, stack = set(), [svc]
    while stack:
        for p in par[stack.pop()]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _series_stats(rng, mean, rel_noise, n=60):
    xs = [max(0.0, rng.gauss(mean, abs(mean) * rel_noise + 1e-9)) for _ in range(n)]
    xs.sort()
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / n)
    return {"mean": round(mu, 4), "p95": round(xs[int(0.95 * (n - 1))], 4), "std": round(sd, 4)}


def _baseline(rng, app, svc):
    depth = 1 + len(_ancestors(APPS[app]["edges"], svc))
    base = {
        "latency_p95_ms": rng.uniform(15, 60) * (1 + 0.3 * (4 - min(depth, 4))),
        "error_rate": rng.uniform(0.0005, 0.006),
        "rps": rng.uniform(40, 120) / depth,
        "cpu": rng.uniform(0.15, 0.4),
        "mem": rng.uniform(0.3, 0.55),
    }
    if svc in APPS[app]["db"]:
        base["db_p95_ms"] = rng.uniform(2, 8)
    return base


def _apply(rng, app, fault, culprit, base):
    edges = APPS[app]["edges"]
    after = {s: dict(v) for s, v in base.items()}
    c = after[culprit]
    upstream_err, upstream_lat, upstream_crash = 0.0, 0.0, False
    if fault == "latency":
        add = c["latency_p95_ms"] * rng.uniform(3, 10)
        c["latency_p95_ms"] += add
        upstream_lat = add
    elif fault == "error":
        e = rng.uniform(0.2, 0.7)
        c["error_rate"] += e
        upstream_err = e * rng.uniform(0.6, 0.9)
    elif fault == "crash":
        c["rps"] *= rng.uniform(0.0, 0.05)
        c["cpu"] *= 0.1
        upstream_err = rng.uniform(0.5, 0.95)
        upstream_crash = True
    elif fault == "db_slow" and "db_p95_ms" in c:
        add = c["db_p95_ms"] * rng.uniform(15, 80)
        c["db_p95_ms"] += add
        c["latency_p95_ms"] += add * 0.9
        upstream_lat = add * 0.9
    elif fault == "db_slow":
        return _apply(rng, app, "latency", culprit, base)
    elif fault == "cpu_hog":
        c["cpu"] = rng.uniform(0.9, 1.0)
        add = c["latency_p95_ms"] * rng.uniform(1, 3)
        c["latency_p95_ms"] += add
        upstream_lat = add
    elif fault == "memory_leak":
        c["mem"] = rng.uniform(0.88, 0.99)
        add = c["latency_p95_ms"] * rng.uniform(0.4, 1.5)
        c["latency_p95_ms"] += add
        upstream_lat = add
    elif fault == "packet_loss":
        add = c["latency_p95_ms"] * rng.uniform(1, 4)
        c["latency_p95_ms"] += add
        c["error_rate"] += rng.uniform(0.02, 0.12)
        upstream_lat = add
        upstream_err = c["error_rate"] * 0.5
    elif fault == "pool_exhaustion" and "db_p95_ms" in c:
        c["db_p95_ms"] += rng.uniform(200, 900)
        c["error_rate"] += rng.uniform(0.1, 0.4)
        add = rng.uniform(300, 1500)
        c["latency_p95_ms"] += add
        upstream_lat, upstream_err = add, c["error_rate"] * 0.7
    elif fault == "pool_exhaustion":
        return _apply(rng, app, "error", culprit, base)
    for a in _ancestors(edges, culprit):
        after[a]["latency_p95_ms"] += upstream_lat * rng.uniform(0.85, 1.05)
        after[a]["error_rate"] += upstream_err * rng.uniform(0.6, 0.95)
        if upstream_crash or upstream_err > 0.3:
            after[a]["rps"] *= rng.uniform(0.85, 1.0)
    return after, upstream_crash


def _herring(rng, app, culprit, after):
    others = [s for s in APPS[app]["edges"] if s != culprit]
    s = rng.choice(others)
    k = rng.choice(["cpu", "mem"])
    after[s][k] = min(0.97, after[s][k] * rng.uniform(1.6, 2.4))
    return s


def _metrics(rng, base, after):
    out = []
    for svc in base:
        for name, v in base[svc].items():
            rel = 0.08 if name != "error_rate" else 0.3
            out.append({
                "service": svc,
                "name": name,
                "signal": name,
                "before": _series_stats(rng, v, rel),
                "after": _series_stats(rng, after[svc][name], rel),
            })
    return out


def _fmt(rng, tpl, svc, dst=""):
    return tpl.format(op=rng.choice(["checkout", "order", "pay", "list", "get"]), ms=rng.randint(40, 4000), n=rng.randint(1, 9999), svc=svc, dst=dst)


def _logs(rng, app, fault, culprit, t0, inject, t1, crashed):
    edges = APPS[app]["edges"]
    par = _parents(edges)
    out = []
    for svc in edges:
        for _ in range(rng.randint(8, 16)):
            t = rng.randint(t0, t1)
            out.append({"t": t, "service": svc, "level": "info", "message": _fmt(rng, rng.choice(INFO), svc)})
    for _ in range(rng.randint(0, 2)):
        svc = rng.choice(list(edges))
        out.append({"t": rng.randint(t0, inject), "service": svc, "level": "warn", "message": _fmt(rng, "retrying request attempt=2 reason=timeout", svc)})
    for _ in range(rng.randint(6, 14)):
        t = rng.randint(inject, t1)
        for tpl in FAULT_LOGS.get(fault, [])[:2]:
            if rng.random() < 0.7:
                out.append({"t": t, "service": culprit, "level": "error", "message": _fmt(rng, tpl, culprit)})
        for p in par[culprit]:
            if crashed:
                key = "crash"
            elif fault in ("error", "pool_exhaustion"):
                key = "error"
            elif fault in ("latency", "db_slow", "cpu_hog", "packet_loss", "memory_leak"):
                key = "timeout" if rng.random() < 0.5 else None
            else:
                key = None
            if key:
                out.append({"t": t, "service": p, "level": "error", "message": _fmt(rng, CALLER_LOGS[key], p, culprit)})
    out.sort(key=lambda x: x["t"])
    return out


def _spans(rng, app, after_metrics, base_metrics, culprit, crashed, t0, inject, t1, n_traces):
    edges = APPS[app]["edges"]
    entry = APPS[app]["entry"]
    spans = []
    for i in range(n_traces):
        is_after = i >= n_traces // 2
        t = rng.randint(inject, t1) if is_after else rng.randint(t0, inject - 1)
        m = after_metrics if is_after else base_metrics
        tid = uuid.UUID(int=rng.getrandbits(128)).hex

        def visit(svc, parent, start):
            sid = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
            own = max(0.5, rng.gauss(m[svc]["latency_p95_ms"] * 0.45, m[svc]["latency_p95_ms"] * 0.08))
            children = [c for c in edges[svc] if rng.random() < 0.8]
            err = rng.random() < m[svc]["error_rate"]
            child_ms = 0.0
            child_err = False
            kids = []
            for c in children:
                if is_after and crashed and c == culprit:
                    csid = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
                    kids.append({"trace_id": tid, "span_id": csid, "parent_id": sid, "service": svc, "operation": f"call {c}", "t": start, "ms": round(rng.uniform(1, 5), 2), "error": True, "kind": "client", "peer": c})
                    child_err = True
                    continue
                sub, sub_ms, sub_err = visit(c, sid, start)
                kids.extend(sub)
                child_ms += sub_ms
                child_err = child_err or sub_err
            if child_err and rng.random() < 0.85:
                err = True
            if is_after and svc == culprit and "db_p95_ms" in m[svc]:
                dsid = uuid.UUID(int=rng.getrandbits(128)).hex[:16]
                dms = max(0.2, rng.gauss(m[svc]["db_p95_ms"] * 0.6, m[svc]["db_p95_ms"] * 0.1))
                kids.append({"trace_id": tid, "span_id": dsid, "parent_id": sid, "service": svc, "operation": "db query", "t": start, "ms": round(dms, 2), "error": False, "kind": "client", "peer": "postgres"})
                child_ms += dms
            total = own + child_ms
            me = {"trace_id": tid, "span_id": sid, "parent_id": parent, "service": svc, "operation": f"handle {svc}", "t": start, "ms": round(total, 2), "error": bool(err), "kind": "server", "peer": ""}
            return [me] + kids, total, err

        if is_after and crashed and culprit == entry:
            continue
        sub, _, _ = visit(entry, "", t)
        spans.extend(sub)
    return spans


def simulate_one(rng, app, fault, culprit, idx, n_traces=24, window_s=600):
    inject = 1_760_000_000 + idx * 3600 + rng.randint(0, 600)
    t0, t1 = inject - window_s // 2, inject + window_s // 2
    base = {s: _baseline(rng, app, s) for s in APPS[app]["edges"]}
    after, crashed = _apply(rng, app, fault, culprit, base)
    if rng.random() < 0.5:
        _herring(rng, app, culprit, after)
    return {
        "incident_id": f"sim-{app}-{culprit}-{fault}-{idx:05d}",
        "source": "sim",
        "app": app,
        "label": {"service": culprit, "fault": fault, "detail": fault},
        "inject_time": inject,
        "window": {"start": t0, "end": t1},
        "services": sorted(APPS[app]["edges"]),
        "metrics": _metrics(rng, base, after),
        "logs": _logs(rng, app, fault, culprit, t0, inject, t1, crashed),
        "spans": _spans(rng, app, after, base, culprit, crashed, t0, inject, t1, n_traces),
    }


def simulate(n_per_combo=3, seed=0, apps=None, faults=None, n_traces=24):
    rng = random.Random(seed)
    out = []
    idx = 0
    for app in apps or list(APPS):
        for svc in sorted(APPS[app]["edges"]):
            for fault in faults or SIM_FAULTS:
                if fault in ("db_slow", "pool_exhaustion") and svc not in APPS[app]["db"]:
                    continue
                for _ in range(n_per_combo):
                    out.append(simulate_one(rng, app, fault, svc, idx, n_traces=n_traces))
                    idx += 1
    return out
