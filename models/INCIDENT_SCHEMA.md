# Incident format (the contract)

This is the one format the models read. The FaultLine eval harness writes one JSON object per line in this format. The model lab reads the file with no code change.

`flm/schema.py` checks every incident with `validate`. A bad incident stops the data build with a clear message.

## Fields

```json
{
  "incident_id": "faultline-000123",
  "source": "faultline",
  "app": "faultline",
  "label": {"service": "payments", "fault": "db_slow", "detail": "seconds=0.3"},
  "inject_time": 1760000000,
  "window": {"start": 1759999700, "end": 1760000300},
  "services": ["gateway", "inventory", "ledger", "notifications", "orders", "payments"],
  "metrics": [
    {"service": "payments", "name": "db_p95_ms", "signal": "db_p95_ms",
     "before": {"mean": 4.1, "p95": 5.0, "std": 0.4},
     "after": {"mean": 352.0, "p95": 410.0, "std": 30.0}}
  ],
  "logs": [
    {"t": 1760000012, "service": "payments", "level": "error", "message": "slow query detected duration_ms=380"}
  ],
  "spans": [
    {"trace_id": "9f2c...", "span_id": "a1b2...", "parent_id": "", "service": "gateway",
     "operation": "POST /checkout", "t": 1760000015, "ms": 420.5, "error": false,
     "kind": "server", "peer": ""}
  ]
}
```

## Rules

1. `label.fault` must be one of the fault kinds in `flm/schema.py`.
2. `label.service` must be in `services`.
3. `inject_time` must be inside `window`.
4. Times are unix seconds.
5. `metrics` hold the before and after summary of each series. Before means the window part before `inject_time`.
6. `signal` is one of `latency_p95_ms`, `error_rate`, `rps`, `cpu`, `mem`, `disk`, `net`, `db_p95_ms`, or `other`.
7. `logs` and `spans` may be sampled. Keep both sides of `inject_time` in the sample.
8. For client spans that call another service, set `kind` to `client` and `peer` to the called service. This lets the models blame a crashed service that never answered.
9. Nothing in metrics, logs or spans may mention the chaos controller. The label is the only place the answer lives.

## Fault kinds

`latency`, `error`, `crash`, `db_slow`, `duplicate`, `poison_message`, `network_partition`, `clock_skew`, `bad_deploy`, `cache_stampede`, `pool_exhaustion`, `cpu_hog`, `memory_leak`, `disk_hog`, `packet_loss`, `socket_exhaustion`, `code_bug`.
