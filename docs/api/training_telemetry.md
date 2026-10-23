# Training telemetry and checkpoint recovery

`training::EventOutbox` stages training observations for foundation
`TelemetrySink` destinations. It remains the only training event queue. Training
depends on foundation; composing an experiment store or an exporter is the
application's responsibility.

## Connect a durable sink

Construct a neutral sink, register the exact semantics and context of every
metric key the Trainer can emit, and supply the outbox in `FitRequest`:

```cpp
#include <jyppx/liuding/training/telemetry_adapter.hpp>
#include <jyppx/liuding/training/trainer.hpp>

jyppx::liuding::training::TrainingTelemetryAdapterOptions telemetry;
telemetry.project_id = "project.demo";
telemetry.attempt_id = "attempt.1";
telemetry.metrics["validation.loss"] = {
    .definition = {
        .metric_id = "training.validation.loss",
        .unit = jyppx::liuding::TelemetryMetricUnit::dimensionless,
        .aggregation = jyppx::liuding::TelemetryAggregation::mean,
        .direction = jyppx::liuding::TelemetryMetricDirection::minimize,
        .context_keys = {"split"}},
    .context = {{"split", "validation"}}};

// durable_sink is a caller-created shared_ptr<jyppx::liuding::TelemetrySink>.
request.event_outbox = std::make_shared<jyppx::liuding::training::EventOutbox>(
    std::vector<std::shared_ptr<jyppx::liuding::TelemetrySink>>{durable_sink}, telemetry);
// Set the remaining FitRequest dependencies, then call trainer.prepare(request).
```

For local SQLite persistence, the composition layer first starts a running
`experiment::RunMetadata`, then creates `experiment::SQLiteRunTelemetrySink`
bound to that run. Neither the outbox nor the adapter needs an experiment header
or owns RunMetadata transitions. The composition layer transitions run metadata
after terminal evidence has been delivered durably.

Missing metric bindings and unspecified live units or aggregation are rejected.
Definitions and contexts are copied at construction and included in checkpoint
topology. Changing the caller's original map cannot alter a staged event.

## One callback transaction and one queue

The dispatcher calls callbacks first. Before committing its event cursor, it
asks the outbox to validate the complete callback snapshot, sample wall and
monotonic clocks once, adapt to a neutral envelope, encode its canonical bytes,
and charge the bounded queue. Any failure at this point rolls back stateful
callbacks and leaves both cursors and the queue unchanged.

Each queue node contains the neutral canonical bytes, the necessary immutable
callback audit snapshot, and one next-sink cursor. Both encoded representations
and a 16-byte node charge count against `max_pending_bytes`; each representation
must also fit `max_event_bytes`. This is a serialization budget, not a promise
that C++ object allocations consume exactly that number of bytes.

Sink I/O occurs only during post-commit `drain_one`, `drain`, `flush`, or `close`.
Destinations are sorted by sink ID. A sink exception or invalid/incomplete
receipt retains the same envelope and its first unacknowledged sink. Retries
never call the clock or metric adapter again. Required evidence cannot be
sampled out or acknowledged as unsupported. An optional observer can explicitly
return an unsupported receipt under the foundation contract.

Delivery remains at least once across process failure or restoration of an
older checkpoint. An acknowledgement interrupted after durable commit can cause
the same bytes to be emitted again. The durable sink must compare operation
identity and content when deduplicating. A new producer attempt must not reuse
the identity of an existing stream with different content.

## State schema 2 and strict N-1 loading

The writer publishes participant schema 2 with exactly two required fields:

| Field | Media type | Contents |
| --- | --- | --- |
| `topology` | `application/vnd.jyppx.event-outbox-topology-v2` | Limits, compatibility mode, ordered sink identities and roles, metric definitions and context, correlation configuration |
| `control` | `application/vnd.jyppx.event-outbox-control-v2` | Stable producer stream, run/epoch/step cursors, terminal flag, callback snapshots, canonical neutral bytes, encoding provenance and next-sink cursors |

An empty `TrainingTelemetryAdapterOptions::stream_id` generates a stream once.
Restoring adopts the saved stream, including when no events remain pending. An
explicitly supplied stream must match the checkpoint. A newly constructed
outbox gets a new monotonic clock ID; restored envelopes retain the original
time and clock ID, while subsequent events use the new clock domain. Monotonic
samples from different clock IDs must never be compared.

`load_state` also recognizes real schema-1 topology/control blobs. It validates
the old manifest and bounds, decodes every callback snapshot, verifies the
delivered prefix and gap-free pending suffix, and converts the complete
candidate before atomically installing any live state. Unsupported/future
schemas, malformed bytes, changed destinations, inconsistent counters, and
insufficient new capacity fail without replacing the live queue.

N-1 migration preserves sorted sink IDs, destination IDs, and every next-sink
cursor. An implementation type can change only through an explicit pair in
`EventOutboxOptions::v1_sink_type_migrations`; a mapping never authorizes a new
logical destination or a reset of delivered history. All new decode bounds must
retain or exceed the corresponding old bounds. The neutral representation can
be larger, so sufficient new queue capacity is also required. When the receiving
destination cannot preserve the old producer prefix, migration is not a way to
silently copy only the pending suffix into a new empty durable stream.

Old observations get a deterministic migration stream, missing timestamps,
explicit `migration.time=unavailable` provenance, and unspecified legacy metric
semantics. No load-time timestamp or new binding is substituted into historical
content. After migration, new events continue the saved sequence using the
current frozen live bindings. Saving again writes schema 2.

The detached `migrate_training_outbox_v1` utility is also available for inspecting
validated candidates, but production recovery uses `EventOutbox::load_state`
against the actual stored fields; the utility is not a replacement queue.

## Existing EventSink compatibility and shutdown

The constructor accepting training `EventSink` objects remains available for
existing JSONL/console sinks and applications. It uses the same schema-2 queue
and retains callback snapshots for their old `publish` interface. Live
compatibility observations capture real times but explicitly retain unknown
legacy metric semantics. New neutral destinations should use registered typed
bindings through the neutral constructor.

Cancellation and an optional absolute steady-clock deadline are checked before
each delivery/finalization call and passed to neutral sinks. Legacy sink calls
only support cancellation internally. A stopped or expired operation retains
pending state; it does not invent a successful flush. Callers may retry using an
active token and a new operation deadline. Final terminal evidence needs such
an explicit completion policy after training cancellation.

`close` drains and flushes before closing sinks, and retries skip sinks already
closed successfully. Checkpointing during or after close is rejected. Destruction
does not perform fallible I/O. Failures after callback commit do not roll back
the accepted training snapshot; the caller handles delivery failure at that
lifecycle boundary.
