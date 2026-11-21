import json

import numpy as np
import pandas as pd

from flm.sources import classify_metric, load_rcaeval, to_seconds


def _case(root, name, inject):
    d = root / name
    d.mkdir(parents=True)
    t = np.arange(inject - 600, inject + 600)
    after = t >= inject
    rng = np.random.default_rng(0)
    pd.DataFrame({
        "time": t,
        "cartservice_cpu": np.where(after, 0.95, 0.2) + rng.normal(0, 0.01, len(t)),
        "cartservice_mem": 0.4 + rng.normal(0, 0.01, len(t)),
        "frontend_latency": np.where(after, 300.0, 50.0) + rng.normal(0, 2, len(t)),
        "frontend_cpu": 0.3 + rng.normal(0, 0.01, len(t)),
    }).to_parquet(d / "metrics.parquet")
    lt = np.concatenate([rng.integers(inject - 300, inject, 20), rng.integers(inject, inject + 300, 40)])
    pd.DataFrame({
        "timestamp": lt * 1_000_000_000,
        "container_name": ["cartservice"] * 60,
        "message": ["ok"] * 20 + ["error: deadline exceeded"] * 40,
    }).to_parquet(d / "logs.parquet")
    rows = []
    for i in range(30):
        ts = inject - 200 + i * 15
        tid = f"t{i}"
        err = ts >= inject
        rows.append({"time": ts * 1_000_000, "traceID": tid, "spanID": f"{tid}a", "parentSpanID": "", "serviceName": "frontend", "operationName": "GET /", "duration": 50_000, "statusCode": 500 if err else 200})
        rows.append({"time": ts * 1_000_000, "traceID": tid, "spanID": f"{tid}b", "parentSpanID": f"{tid}a", "serviceName": "cartservice", "operationName": "GetCart", "duration": 40_000, "statusCode": 2 if err else 0})
    pd.DataFrame(rows).to_parquet(d / "traces.parquet")
    (d / "inject_time.txt").write_text(str(inject))
    return d


def test_loader_reads_parquet_case(cfg, tmp_path):
    inject = 1_700_000_000
    _case(tmp_path, "re2ob_cartservice_cpu_1", inject)
    pd.DataFrame([{"case": "re2ob_cartservice_cpu_1", "root_cause_service": "cartservice", "fault": "cpu", "inject_time": inject}]).to_parquet(tmp_path / "cases.parquet")
    c = json.loads(json.dumps(cfg))
    c["data"]["rcaeval"]["max_cases"] = None
    incs = load_rcaeval(c, tmp_path)
    assert len(incs) == 1
    inc = incs[0]
    assert inc["label"] == {"service": "cartservice", "fault": "cpu_hog", "detail": "cpu"}
    assert inc["app"] == "online_boutique"
    assert {"cartservice", "frontend"} <= set(inc["services"])
    assert any(m["signal"] == "cpu" and m["service"] == "cartservice" for m in inc["metrics"])
    assert inc["logs"] and inc["spans"]
    from flm.signals import extract, rule_rank

    sig = extract(inc)
    assert rule_rank(sig, inc["services"])[0] == "cartservice"


def test_helpers():
    assert to_seconds(1_700_000_000_123) == 1_700_000_000
    assert to_seconds(1_700_000_000_000_000) == 1_700_000_000
    assert to_seconds(1_700_000_000_000_000_000) == 1_700_000_000
    assert classify_metric("latency-90") == "latency_p95_ms"
    assert classify_metric("mem") == "mem"
