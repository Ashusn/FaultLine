# 0001: Day one conventions

* Status: accepted
* Date: 2026-11-21

## Context

Five services share one Postgres server and one Kafka cluster. If each service picks its own event shape, topic names and table habits, the outbox relay, the inbox dedup and the AI engine's parsers all need special cases. These rules are fixed before any service code exists.

## Decision

### 1. One Postgres instance, one database per service

| Service   | Database    |
|-----------|-------------|
| orders    | `orders`    |
| payments  | `payments`  |
| inventory | `inventory` |
| chaos     | `chaos`     |

* A service connects only to its own database, with its own Postgres role.
* No service reads or writes another service's tables. Data crosses service lines only through HTTP calls or events.
* Migrations live in `migrations/<service>/`.

### 2. Event envelope

Every event on Kafka is JSON with exactly these top level fields:

```json
{
  "eventId": "7f3c9a2e-5b1d-4e8a-9c6f-2d4b8e1a0f37",
  "eventType": "OrderCreated",
  "orderId": "ord_123",
  "occurredAt": "2026-11-21T10:15:30.123Z",
  "payload": { }
}
```

| Field        | Type            | Rule |
|--------------|-----------------|------|
| `eventId`    | string (UUID)   | Unique per event. The inbox dedups on it. |
| `eventType`  | string          | PascalCase, past tense: `OrderCreated`, `PaymentFailed`. |
| `orderId`    | string          | The order the event belongs to. Also the Kafka message key. |
| `occurredAt` | RFC 3339, UTC   | When the change happened, not when it was published. |
| `payload`    | object          | Event specific body. |

The Go type is `events.Envelope` in `pkg/events`.

### 3. Kafka key is `orderId`

Every event is produced with `orderId` as the message key. Kafka sends all messages with the same key to the same partition, so one order's events are consumed in the order they were written. There is no ordering across different orders, and no service may depend on it.

### 4. Topic names

| Topic              | Dead letter topic      | Producer  |
|--------------------|------------------------|-----------|
| `orders.events`    | `orders.events.dlt`    | orders    |
| `inventory.events` | `inventory.events.dlt` | inventory |
| `payments.events`  | `payments.events.dlt`  | payments  |

* Each topic has 3 partitions.
* A service produces only to its own topic.
* A consumer moves a message to `<topic>.dlt` after its retries run out, or straight away if the message cannot be parsed. It never blocks the partition.
* Names are constants in `pkg/events`. Do not type topic strings by hand.

### 5. Every table has `created_at` and `updated_at`

```sql
created_at timestamptz NOT NULL DEFAULT now(),
updated_at timestamptz NOT NULL DEFAULT now()
```

Every `UPDATE` sets `updated_at = now()`. Times are stored as `timestamptz` and handled in UTC.

## Consequences

* Joins across services are impossible. A service that needs another service's data keeps its own copy, built from events.
* One envelope means one outbox relay, one inbox and one parser in the AI engine.
* Ordering holds only within one order. Handlers must still be idempotent, because Kafka delivers at least once.
* Changing any rule here needs a new ADR that supersedes this one.
