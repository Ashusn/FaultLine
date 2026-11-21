FAULT_KINDS = [
    "latency",
    "error",
    "crash",
    "db_slow",
    "duplicate",
    "poison_message",
    "network_partition",
    "clock_skew",
    "bad_deploy",
    "cache_stampede",
    "pool_exhaustion",
    "cpu_hog",
    "memory_leak",
    "disk_hog",
    "packet_loss",
    "socket_exhaustion",
    "code_bug",
]

RAW_FAULT_MAP = {
    "delay": "latency",
    "loss": "packet_loss",
    "socket": "socket_exhaustion",
    "cpu": "cpu_hog",
    "mem": "memory_leak",
    "disk": "disk_hog",
    "f1": "code_bug",
    "f2": "code_bug",
    "f3": "code_bug",
    "f4": "code_bug",
    "f5": "code_bug",
}

SIGNALS = [
    "latency_p95_ms",
    "error_rate",
    "rps",
    "cpu",
    "mem",
    "disk",
    "net",
    "db_p95_ms",
    "log_errors",
    "trace_errors",
    "trace_self_ms",
]

REQUIRED = ["incident_id", "source", "app", "label", "inject_time", "window", "services", "metrics", "logs", "spans"]


def map_fault(raw):
    r = str(raw).strip().lower()
    if r in FAULT_KINDS:
        return r, r
    if r in RAW_FAULT_MAP:
        return RAW_FAULT_MAP[r], r
    raise ValueError(f"unknown fault kind {raw!r}")


def validate(inc):
    errs = []
    for k in REQUIRED:
        if k not in inc:
            errs.append(f"missing {k}")
    if errs:
        raise ValueError(f"{inc.get('incident_id', '?')}: " + ", ".join(errs))
    lab = inc["label"]
    if lab.get("fault") not in FAULT_KINDS:
        errs.append(f"bad fault {lab.get('fault')!r}")
    if lab.get("service") not in inc["services"]:
        errs.append(f"label service {lab.get('service')!r} not in services")
    w = inc["window"]
    if not (w["start"] <= inc["inject_time"] <= w["end"]):
        errs.append("inject_time outside window")
    for m in inc["metrics"]:
        if not {"service", "name", "before", "after"} <= set(m):
            errs.append("bad metric entry")
            break
    for s in inc["spans"]:
        if not {"trace_id", "span_id", "service", "t", "ms", "error"} <= set(s):
            errs.append("bad span entry")
            break
    for line in inc["logs"]:
        if not {"t", "service", "message"} <= set(line):
            errs.append("bad log entry")
            break
    if errs:
        raise ValueError(f"{inc['incident_id']}: " + ", ".join(errs))
    return inc


def group_key(inc):
    lab = inc["label"]
    return f"{inc['app']}|{lab['service']}|{lab['fault']}"


def public(inc):
    return {k: v for k, v in inc.items() if k != "label"}
