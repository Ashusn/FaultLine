import math
from collections import defaultdict

from .schema import SIGNALS

RATE = {"error_rate"}


def _r(x):
    if x == 0 or x is None or not math.isfinite(x):
        return 0 if x is None or not math.isfinite(x) else x
    return float(f"{x:.3g}")


def _ratio_sev(b, a):
    eps = max(1e-6, 0.01 * abs(b))
    return abs(math.log((a + eps) / (b + eps)))


def metric_anomalies(inc, z_min=3.0, ratio_min=0.405, err_min=0.01):
    out = []
    for m in inc["metrics"]:
        sig = m.get("signal") or m["name"]
        b, a = m["before"], m["after"]
        bm, am = float(b["mean"]), float(a["mean"])
        sd = float(b.get("std", 0) or 0)
        z = abs(am - bm) / (sd + 1e-9 + 0.02 * abs(bm))
        if sig in RATE:
            d = am - bm
            if d < err_min:
                continue
            sev = d / 0.05
        else:
            sev = _ratio_sev(bm, am)
            if sev < ratio_min or z < z_min:
                continue
        out.append({"service": m["service"], "signal": sig if sig in SIGNALS else "other", "metric": m["name"], "before": _r(bm), "after": _r(am), "severity": round(sev, 3)})
    return out


def log_anomalies(inc, min_diff=3):
    t = inc["inject_time"]
    cnt = defaultdict(lambda: [0, 0])
    example = {}
    for line in inc["logs"]:
        if str(line.get("level", "")).lower() not in ("error", "fatal", "critical"):
            continue
        k = line["service"]
        cnt[k][1 if line["t"] >= t else 0] += 1
        if line["t"] >= t and k not in example:
            example[k] = line["message"][:160]
    out = []
    for svc, (b, a) in cnt.items():
        if a - b >= min_diff:
            out.append({"service": svc, "signal": "log_errors", "metric": "error_lines", "before": b, "after": a, "severity": round(math.log((a + 1) / (b + 1)), 3), "example": example.get(svc, "")})
    return out


def _children(spans):
    kids = defaultdict(list)
    for s in spans:
        if s.get("parent_id"):
            kids[s["parent_id"]].append(s)
    return kids


def origin_errors(spans, services):
    kids = _children(spans)
    out = []
    for s in spans:
        if not s["error"]:
            continue
        ch = kids.get(s["span_id"], [])
        if any(c["error"] for c in ch):
            continue
        peer = s.get("peer") or ""
        if s.get("kind") == "client" and peer in services and not ch:
            out.append((peer, s))
        else:
            out.append((s["service"], s))
    return out


def self_times(spans):
    kids = _children(spans)
    out = []
    for s in spans:
        own = s["ms"] - sum(c["ms"] for c in kids.get(s["span_id"], []))
        out.append((s, max(0.0, own)))
    return out


def trace_anomalies(inc, min_err=2, self_ratio=1.5, self_min_ms=5.0):
    spans = inc["spans"]
    if not spans:
        return []
    t = inc["inject_time"]
    services = set(inc["services"])
    errs = defaultdict(lambda: [0, 0])
    ex = {}
    for svc, s in origin_errors(spans, services):
        side = 1 if s["t"] >= t else 0
        errs[svc][side] += 1
        if side and svc not in ex:
            ex[svc] = s
    out = []
    for svc, (b, a) in errs.items():
        if a - b >= min_err:
            s = ex.get(svc)
            out.append({"service": svc, "signal": "trace_errors", "metric": "origin_error_spans", "before": b, "after": a, "severity": round(math.log((a + 1) / (b + 1)), 3), "trace_id": s["trace_id"] if s else "", "span_id": s["span_id"] if s else ""})
    st = defaultdict(lambda: [[], []])
    for s, own in self_times(spans):
        st[s["service"]][1 if s["t"] >= t else 0].append(own)
    for svc, (b, a) in st.items():
        if not b or not a:
            continue
        bm, am = sum(b) / len(b), sum(a) / len(a)
        if am > bm * self_ratio and am - bm > self_min_ms:
            out.append({"service": svc, "signal": "trace_self_ms", "metric": "self_time_ms", "before": _r(bm), "after": _r(am), "severity": round(math.log((am + 1) / (bm + 1)), 3)})
    return out


def extract(inc, top_k=8, cfg=None):
    c = (cfg or {}).get("signals", {})
    anoms = metric_anomalies(inc, c.get("z_min", 3.0), c.get("ratio_min", 0.405), c.get("err_min", 0.01))
    anoms += log_anomalies(inc, c.get("log_min_diff", 3))
    anoms += trace_anomalies(inc, c.get("trace_min_err", 2))
    anoms.sort(key=lambda x: (-x["severity"], x["service"], x["signal"], x["metric"]))
    best = {}
    for a in anoms:
        key = (a["service"], a["signal"])
        if key not in best:
            best[key] = a
    picked = list(best.values())[:top_k]
    return {"anomalies": picked}


def anomaly_key(a):
    return (a.get("service"), a.get("signal"))


def signal_f1(pred, ref):
    p = {anomaly_key(a) for a in (pred or {}).get("anomalies", []) if isinstance(a, dict)}
    r = {anomaly_key(a) for a in (ref or {}).get("anomalies", [])}
    if not p and not r:
        return 1.0
    if not p or not r:
        return 0.0
    tp = len(p & r)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(r)
    return 2 * prec * rec / (prec + rec)


def _signed(a):
    b, x = float(a.get("before", 0) or 0), float(a.get("after", 0) or 0)
    if a.get("signal") in RATE:
        return x - b
    eps = max(1e-6, 0.01 * abs(b))
    return math.log((abs(x) + eps) / (abs(b) + eps))


FEATURES = [f"{s}_chg" for s in SIGNALS] + ["n_anoms", "sum_sev", "max_sev", "rank", "share", "is_top", "has_origin_err", "has_self_time"]


def service_features(signals, services):
    per = {s: defaultdict(float) for s in services}
    for a in signals.get("anomalies", []):
        s = a.get("service")
        if s not in per:
            continue
        sig = a.get("signal", "other")
        k = f"{sig}_chg"
        if k in FEATURES:
            v = _signed(a)
            if abs(v) > abs(per[s][k]):
                per[s][k] = v
        sev = float(a.get("severity", 0) or 0)
        per[s]["n_anoms"] += 1
        per[s]["sum_sev"] += sev
        per[s]["max_sev"] = max(per[s]["max_sev"], sev)
        if sig == "trace_errors":
            per[s]["has_origin_err"] = 1
        if sig == "trace_self_ms":
            per[s]["has_self_time"] = 1
    total = sum(per[s]["sum_sev"] for s in services) or 1.0
    order = sorted(services, key=lambda s: -per[s]["sum_sev"])
    for i, s in enumerate(order):
        per[s]["rank"] = i
        per[s]["share"] = per[s]["sum_sev"] / total
        per[s]["is_top"] = 1 if i == 0 and per[s]["sum_sev"] > 0 else 0
    return {s: [float(per[s][f]) for f in FEATURES] for s in services}


def rule_rank(signals, services):
    feats = service_features(signals, services)
    idx_sum, idx_org = FEATURES.index("sum_sev"), FEATURES.index("has_origin_err")
    return sorted(services, key=lambda s: (-(feats[s][idx_sum] + 2.0 * feats[s][idx_org]), s))
