# Service runbooks

One section per service. Each says what the service does, what it depends on and what breaks first.

#### service gateway
Front door of the FaultLine shop. Accepts checkouts and calls orders.
Depends on: orders.
Breaks first: when orders is slow or down, gateway latency and 502 responses rise. Gateway itself rarely holds the root cause.

#### service orders
Creates orders, reserves stock and asks payments to charge. Runs the checkout saga.
Depends on: inventory, payments, notifications, its own Postgres.
Breaks first: slow inventory or payments calls, row lock waits on the orders table, saga compensation errors.

#### service payments
Charges each order exactly once using an idempotency key.
Depends on: ledger, its own Postgres.
Breaks first: unique key conflicts under retries, slow inserts, ledger timeouts.

#### service inventory
Keeps stock counts and reserves items with conditional updates.
Depends on: its own Postgres and the Redis cache.
Breaks first: hot rows on popular items, cache stampede after a flush.

#### service notifications
Sends order emails from queue events.
Depends on: the message queue and an email provider.
Breaks first: poison messages and slow provider calls. Failures here should not block checkout.

#### service ledger
Writes the double entry record of every payment.
Depends on: its own Postgres.
Breaks first: lock contention on account rows and slow fsync on the database disk.

#### service frontend
Web front of the bookstore and of Online Boutique. Renders pages and calls backend services.
Depends on: catalog, cart, or product catalog, currency, cart and checkout in Online Boutique.
Breaks first: shows errors of every backend, so it is usually a symptom, not the cause.

#### service catalog
Bookstore catalog. Lists and searches books.
Depends on: search, pricing, its own database.
Breaks first: slow search queries and database load during sales.

#### service cart
Bookstore cart. Keeps items per customer.
Depends on: pricing, checkout, its own database.
Breaks first: database pool pressure on busy days.

#### service checkout
Places orders in the bookstore and in Online Boutique (checkoutservice).
Depends on: billing or payment, shipping, email.
Breaks first: timeouts to payment and billing.

#### service search
Bookstore search index.
Depends on: none.
Breaks first: CPU during index rebuilds.

#### service pricing
Bookstore price and discount rules.
Depends on: none.
Breaks first: CPU spikes from complex discount rules.

#### service billing
Bookstore billing. Charges cards.
Depends on: its own database.
Breaks first: slow database writes.

#### service cartservice
Online Boutique cart, written in C sharp, backed by Redis.
Depends on: redis.
Breaks first: Redis latency, memory growth, connection errors to Redis.

#### service checkoutservice
Online Boutique checkout. Calls cart, product catalog, currency, shipping, payment and email.
Depends on: many services, so it shows errors of all of them.
Breaks first: timeouts to payment or shipping.

#### service currencyservice
Online Boutique currency conversion, written in Node.
Depends on: none.
Breaks first: CPU, because it is called on almost every page.

#### service productcatalogservice
Online Boutique product list, written in Go.
Depends on: none.
Breaks first: latency when its list grows, CPU on search.

#### service recommendationservice
Online Boutique recommendations, written in Python.
Depends on: productcatalogservice.
Breaks first: memory and CPU under load.

#### service adservice
Online Boutique ads, written in Java.
Depends on: none.
Breaks first: JVM memory and GC pauses.

#### service emailservice
Online Boutique order emails.
Depends on: none.
Breaks first: slow template rendering, rarely on the checkout path for errors.

#### service paymentservice
Online Boutique payments.
Depends on: none.
Breaks first: errors on invalid cards and CPU on bursts.

#### service shippingservice
Online Boutique shipping quotes.
Depends on: none.
Breaks first: latency and errors that break checkout.

#### service carts
Sock Shop cart service, Java, backed by MongoDB.
Depends on: carts database.
Breaks first: JVM memory and slow database calls.

#### service catalogue
Sock Shop product catalogue, Go, backed by MySQL.
Depends on: catalogue database.
Breaks first: slow queries and CPU.

#### service orders_sockshop
Sock Shop orders service. Calls payment, shipping, user and carts.
Depends on: payment, shipping, user, carts.
Breaks first: timeouts to payment and shipping.

#### service payment
Sock Shop payment authorisation, Go.
Depends on: none.
Breaks first: latency and errors that fail orders.

#### service user
Sock Shop user accounts, Go, backed by MongoDB.
Depends on: user database.
Breaks first: slow logins when the database is slow.
