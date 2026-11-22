# Platform

The shop that gets broken on purpose. Owned by the **platform** side.

## What lives here

One Go module (`go.mod`) for the whole platform.

* `cmd/<service>/`: `main.go` for each binary: gateway, orders, payments, inventory, chaos
* `internal/<service>/`: handlers, service and repository code private to one service
* `pkg/`: code shared by all services
  * `events/`: event structs
  * `outbox/`: transactional outbox table and relay
  * `inbox/`: `processed_events` dedup
  * `idempotency/`: HTTP middleware
  * `telemetry/`: OpenTelemetry, slog and Prometheus setup
* `migrations/<service>/`: SQL migrations, one folder per service database
* `deploy/compose/`: `docker-compose.yml` and Grafana dashboards
* `deploy/k8s/`: Kubernetes manifests or Helm chart
* `loadtest/`: k6 scripts
* `docs/adr/`: one file per architecture decision record
* `Makefile`: `make build`, `make test`, `make up`

## Stack

Fill this in before writing code.

* Language:
* HTTP framework:
* Database: Postgres
* Cache: Redis
* Queue: NATS JetStream or Kafka
* Telemetry: OpenTelemetry SDK, exported over OTLP

## Contract with the AI engine

1. Every service sets `service.name` to its folder name.
2. Every log line carries `trace_id` and `span_id`.
3. Faults look like real failures. No log, metric or span may mention the chaos controller.
4. The chaos API and the `faults` table shape stay stable once Phase 1 ends.

## Done means

One command starts everything, and a single checkout shows up in Grafana as one trace across gateway, orders and payments.
