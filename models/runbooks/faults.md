# Fault runbooks

One section per fault kind. Each section says what you see and how to confirm it.

#### fault latency
A service answers slowly while its error rate stays mostly flat.
What you see: p95 latency of the service jumps several times, and every caller above it gets slower by about the same amount.
How to confirm: in a slow trace, the span with the most self time belongs to this service, not to its children. Callers mostly wait on it.
Next step: check recent config or dependency changes, thread pools and GC pauses on that service.

#### fault error
A share of requests fail inside one service.
What you see: error rate rises in the service and in every caller, with 500 responses and upstream errors in caller logs.
How to confirm: the deepest failing span in an error trace belongs to this service. Its own logs show internal errors or invalid state.
Next step: read the first error log lines after the alert and compare with the last deploy.

#### fault crash
The process dies and restarts in a loop.
What you see: request rate of the service drops close to zero, CPU drops, callers log connection refused or dial errors.
How to confirm: client spans in callers fail with no matching server span in the crashed service. Panic or exit lines appear in its logs.
Next step: read the panic stack trace and check the restart count.

#### fault db_slow
Database calls of one service become slow.
What you see: database p95 of the service rises by an order of magnitude, its latency follows, and callers slow down.
How to confirm: inside slow traces the database client span of this service holds most of the time. Slow query logs name the statement.
Next step: check locks, missing indexes and long running transactions on that database.

#### fault duplicate
The same message is delivered many times.
What you see: payment or order requests repeat with the same idempotency key. Charge counts would rise if idempotency is broken.
How to confirm: spans marked as deduplicated, and logs saying a duplicate request was suppressed.
Next step: check consumer acknowledgements and retry settings of the queue.

#### fault poison_message
One message always crashes its consumer.
What you see: consumer restarts, queue depth grows, the dead letter queue fills.
How to confirm: the same message id appears before every consumer failure.
Next step: move the message to the dead letter queue and fix the parser.

#### fault network_partition
Two services cannot reach each other.
What you see: timeouts and open circuit breakers between one pair of services while other paths work.
How to confirm: client spans to one peer time out with no server span, and only that edge is affected.
Next step: check network policies, DNS and the proxy between the two services.

#### fault clock_skew
One service believes it is a different time.
What you see: token or signature expiry errors, odd ordering of events, timestamps from one service far from the others.
How to confirm: log timestamps of one service drift from the rest of the system by the same offset.
Next step: check NTP on the host of that service.

#### fault bad_deploy
A new version with a bug goes live.
What you see: errors or latency start exactly at a deploy time in one service.
How to confirm: the version label on spans or logs changes at the same moment the errors begin.
Next step: roll back and compare the diff.

#### fault cache_stampede
The cache empties and every request hits the database.
What you see: cache miss rate jumps, database load and latency spike right after an expiry or a flush.
How to confirm: many identical database queries in a short burst, all after cache misses.
Next step: add request coalescing or stagger expiry times.

#### fault pool_exhaustion
The database connection pool runs out.
What you see: timeouts while acquiring a connection, errors and latency in one service, database itself not busy.
How to confirm: logs about the pool being exhausted or a timeout acquiring a connection, and active connections at the maximum.
Next step: look for leaked connections or long transactions, then size the pool.

#### fault cpu_hog
One service runs out of CPU.
What you see: CPU near 100 percent on one service, request queues grow, latency rises moderately.
How to confirm: CPU metric of that service is saturated while its neighbours are normal, and self time rises in its spans.
Next step: profile the hot path and check for runaway loops.

#### fault memory_leak
Memory of one service keeps growing.
What you see: memory near the limit, long GC pauses, latency rising, sometimes out of memory restarts.
How to confirm: memory metric climbs steadily in one service and GC pause logs appear.
Next step: take a heap profile and look for unbounded caches.

#### fault disk_hog
Disk of one service is saturated or full.
What you see: disk usage or IO wait high on one service, slow writes, sometimes write errors.
How to confirm: disk metrics of that service spike while others stay flat.
Next step: find the large files or the noisy writer, clean up or move logs.

#### fault packet_loss
Packets to or from one service are lost.
What you see: retries, connection resets and read timeouts, with both latency and a small error rate rising.
How to confirm: network error and retransmit metrics rise on one service, and callers log connection reset by peer.
Next step: check the network interface and the sidecar proxy of that service.

#### fault socket_exhaustion
A service runs out of sockets or file descriptors.
What you see: cannot assign requested address or too many open files errors, new connections fail.
How to confirm: socket or file descriptor counts at the limit on one service.
Next step: enable connection reuse and raise the limit after finding the leak.

#### fault code_bug
A bug in the code returns wrong values or throws.
What you see: errors with stack traces in one service, or wrong responses that make callers fail.
How to confirm: exception stack traces point into one service, and its error responses start the failing traces.
Next step: read the stack trace, find the function and check its inputs and return values.
