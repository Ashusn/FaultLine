import os
import random
import re
from pathlib import Path

from .common import log, read_jsonl, wdir
from .schema import map_fault, validate
from .simulate import simulate

SIGNAL_RULES = [
    ("db_p95_ms", re.compile(r"(db|sql|query|postgres|mongo|redis)", re.I)),
    ("latency_p95_ms", re.compile(r"(latency|duration|response_time|rt)", re.I)),
    ("error_rate", re.compile(r"(error|fail|5xx)", re.I)),
    ("rps", re.compile(r"(workload|rps|qps|request|throughput)", re.I)),
    ("cpu", re.compile(r"cpu", re.I)),
    ("mem", re.compile(r"(mem|rss|heap)", re.I)),
    ("disk", re.compile(r"(disk|fs_|io)", re.I)),
    ("net", re.compile(r"(net|receive|transmit|socket|tcp)", re.I)),
]


def classify_metric(name):
    for sig, rx in SIGNAL_RULES:
        if rx.search(name):
            return sig
    return "other"


def to_seconds(x):
    x = float(x)
    if x > 1e17:
        return int(x / 1e9)
    if x > 1e14:
        return int(x / 1e6)
    if x > 1e11:
        return int(x / 1e3)
    return int(x)


def _stats(values):
    import numpy as np

    v = np.asarray([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return None
    return {"mean": round(float(v.mean()), 4), "p95": round(float(np.percentile(v, 95)), 4), "std": round(float(v.std()), 4)}


def _match_service(name, services):
    n = str(name).lower()
    best = None
    for s in services:
        if n == s or n.startswith(s) or s in n:
            if best is None or len(s) > len(best):
                best = s
    return best


def _read(case_dir, kind):
    import pandas as pd

    for fname in (f"{kind}.parquet", f"{kind}.json", f"{kind}.csv"):
        p = case_dir / fname
        if p.exists():
            if fname.endswith(".parquet"):
                return pd.read_parquet(p)
            if fname.endswith(".csv"):
                return pd.read_csv(p, low_memory=False)
            import json

            raw = {k: v for k, v in json.loads(p.read_text()).items() if v}
            times = sorted({pt[0] for v in raw.values() for pt in v})
            pos = {t: i for i, t in enumerate(times)}
            cols = {k: [float("nan")] * len(times) for k in raw}
            for k, v in raw.items():
                for t, val in v:
                    if val is not None:
                        cols[k][pos[t]] = val
            df = pd.DataFrame(cols)
            df.insert(0, "time", times)
            return df
    if kind == "metrics" and (case_dir / "data.csv").exists():
        return pd.read_csv(case_dir / "data.csv")
    return None


def load_rcaeval_case(case_dir, label, inject_time, cfg):
    case_dir = Path(case_dir)
    half = int(cfg["data"]["window_s"]) // 2
    t0, t1 = inject_time - half, inject_time + half
    rng = random.Random(f"{case_dir.name}:{cfg.get('seed', 0)}")
    m = _read(case_dir, "metrics")
    if m is None:
        return None
    m["time"] = m["time"].map(to_seconds)
    m = m[(m["time"] >= t0) & (m["time"] <= t1)]
    cols = [c for c in m.columns if c != "time" and "_" in c]
    services = sorted({c.split("_", 1)[0].lower() for c in cols})
    metrics = []
    for c in cols:
        svc, name = c.split("_", 1)
        b = _stats(m.loc[m["time"] < inject_time, c].tolist())
        a = _stats(m.loc[m["time"] >= inject_time, c].tolist())
        if b and a:
            metrics.append({"service": svc.lower(), "name": name, "signal": classify_metric(name), "before": b, "after": a})
    logs = []
    lg = _read(case_dir, "logs")
    if lg is not None and len(lg):
        tcol = next((c for c in ("timestamp", "time") if c in lg.columns), None)
        scol = next((c for c in ("container_name", "service", "container") if c in lg.columns), None)
        mcol = next((c for c in ("message", "log", "msg") if c in lg.columns), None)
        if tcol and mcol:
            lg = lg.assign(tsec=lg[tcol].map(to_seconds))
            lg = lg[(lg["tsec"] >= t0) & (lg["tsec"] <= t1)]
            rows = []
            for t, s, msg in zip(lg["tsec"], lg[scol] if scol else [""] * len(lg), lg[mcol]):
                svc = _match_service(s, services)
                if svc:
                    msg = str(msg)
                    level = "error" if re.search(r"(error|exception|fail|panic|refused|timeout)", msg, re.I) else "info"
                    rows.append({"t": int(t), "service": svc, "level": level, "message": msg[:400]})
            errs = [r for r in rows if r["level"] == "error"]
            rest = [r for r in rows if r["level"] != "error"]
            cap = int(cfg["data"]["max_logs"])
            keep = rng.sample(errs, min(len(errs), cap // 2))
            keep += rng.sample(rest, min(len(rest), cap - len(keep)))
            logs = sorted(keep, key=lambda r: r["t"])
    spans = []
    tr = _read(case_dir, "traces")
    if tr is not None and len(tr):
        tcol = next((c for c in ("time", "startTime", "startTimeMillis", "timestamp") if c in tr.columns), None)
        if tcol and "traceID" in tr.columns:
            tr = tr.assign(tsec=tr[tcol].map(to_seconds))
            tr = tr[(tr["tsec"] >= t0) & (tr["tsec"] <= t1)]
            first = tr.groupby("traceID")["tsec"].min()
            before = sorted(first[first < inject_time].index.tolist())
            after = sorted(first[first >= inject_time].index.tolist())
            n = int(cfg["data"]["max_traces"]) // 2
            pick = set(rng.sample(before, min(n, len(before))) + rng.sample(after, min(n, len(after))))
            for r in tr[tr["traceID"].isin(pick)].itertuples(index=False):
                d = r._asdict()
                svc = _match_service(d.get("serviceName", ""), services)
                if not svc:
                    continue
                status = d.get("statusCode")
                try:
                    status = int(float(status))
                except (TypeError, ValueError):
                    status = None
                err = status is not None and status != 0 and not (200 <= status < 400)
                spans.append({
                    "trace_id": str(d["traceID"]),
                    "span_id": str(d.get("spanID", "")),
                    "parent_id": str(d.get("parentSpanID", "") or ""),
                    "service": svc,
                    "operation": str(d.get("operationName", d.get("methodName", "")) or ""),
                    "t": int(d["tsec"]),
                    "ms": round(float(d.get("duration", 0) or 0) / 1000.0, 3),
                    "error": bool(err),
                    "kind": "",
                    "peer": "",
                })
    svc = label["service"].lower()
    if svc not in services:
        services = sorted(set(services) | {svc})
    kind, detail = map_fault(label["fault"])
    inc = {
        "incident_id": f"rcaeval-{case_dir.name}",
        "source": "rcaeval",
        "app": label["system"],
        "label": {"service": svc, "fault": kind, "detail": detail},
        "inject_time": int(inject_time),
        "window": {"start": t0, "end": t1},
        "services": services,
        "metrics": metrics,
        "logs": logs,
        "spans": spans,
    }
    return validate(inc)


def fetch_rcaeval(cfg):
    from huggingface_hub import snapshot_download

    src = cfg["data"]["rcaeval"]
    local = wdir(cfg, "cache", "rcaeval")
    patterns = src.get("patterns") or (["cases.parquet"] + [f"{s}*" for s in src["suites"]])
    snapshot_download(repo_id=src["repo"], repo_type="dataset", allow_patterns=patterns, local_dir=str(local))
    return local


def load_rcaeval(cfg, root):
    import pandas as pd

    root = Path(root)
    idx_path = root / "cases.parquet"
    index = {}
    if idx_path.exists():
        idx = pd.read_parquet(idx_path)
        for r in idx.to_dict("records"):
            index[str(r["case"])] = r
    out = []
    limit = cfg["data"]["rcaeval"].get("max_cases")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and re.match(r"re\d(ob|ss|tt)_", p.name))
    suites = tuple(s.lower() for s in cfg["data"]["rcaeval"]["suites"])
    dirs = [d for d in dirs if d.name.lower().startswith(suites)]
    if limit:
        dirs = dirs[: int(limit)]
    for d in dirs:
        m = re.match(r"(re\d)(ob|ss|tt)_(.+)_([a-z0-9]+)_(\d+)$", d.name)
        if not m:
            continue
        r = index.get(d.name, {})
        service = str(r.get("root_cause_service") or m.group(3))
        fault = str(r.get("fault") or m.group(4))
        inject = r.get("inject_time")
        if inject is None and (d / "inject_time.txt").exists():
            inject = int((d / "inject_time.txt").read_text().split()[0])
        if inject is None:
            continue
        label = {"service": service, "fault": fault, "system": {"ob": "online_boutique", "ss": "sock_shop", "tt": "train_ticket"}[m.group(2)]}
        try:
            inc = load_rcaeval_case(d, label, to_seconds(inject), cfg)
        except ValueError as e:
            log("skip", d.name, e)
            continue
        if inc:
            out.append(inc)
    return out


def load_jsonl_sources(cfg):
    out = []
    for p in cfg["data"].get("jsonl", []) or []:
        if os.path.exists(p):
            out.extend(validate(r) for r in read_jsonl(p))
    return out


def collect(cfg):
    d = cfg["data"]
    incidents = []
    if d.get("sim", {}).get("enabled", True):
        s = d["sim"]
        incidents += simulate(n_per_combo=int(s["n_per_combo"]), seed=int(cfg.get("seed", 0)), apps=s.get("apps"), faults=s.get("faults"), n_traces=int(s.get("n_traces", 24)))
        log("sim incidents", len(incidents))
    if d.get("rcaeval", {}).get("enabled", False):
        root = d["rcaeval"].get("local_dir") or str(wdir(cfg, "cache", "rcaeval"))
        if not any(Path(root).glob("re*")):
            root = fetch_rcaeval(cfg)
        r = load_rcaeval(cfg, root)
        log("rcaeval incidents", len(r))
        incidents += r
    j = load_jsonl_sources(cfg)
    if j:
        log("jsonl incidents", len(j))
    incidents += j
    return [validate(i) for i in incidents]
