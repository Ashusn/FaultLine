package events

import (
	"encoding/json"
	"time"
)

// Topics. A service produces only to its own topic. See docs/adr/0001-day-one-conventions.md.
const (
	TopicOrders    = "orders.events"
	TopicInventory = "inventory.events"
	TopicPayments  = "payments.events"
)

// Partitions is the partition count for every topic.
const Partitions = 3

// dltSuffix marks a dead letter topic.
const dltSuffix = ".dlt"

// DLT returns the dead letter topic for topic, for example "orders.events.dlt".
func DLT(topic string) string { return topic + dltSuffix }

// Envelope is the JSON shape of every event on Kafka.
// OrderID is also the Kafka message key, so one order's events stay in order.
type Envelope struct {
	EventID    string          `json:"eventId"`
	EventType  string          `json:"eventType"`
	OrderID    string          `json:"orderId"`
	OccurredAt time.Time       `json:"occurredAt"`
	Payload    json.RawMessage `json:"payload"`
}

// Key returns the Kafka message key for the event.
func (e Envelope) Key() []byte { return []byte(e.OrderID) }
