# API: Local Run Storage

Target: `jyppx::experiment`  
Headers: `jyppx/experiment/artifact_store.hpp`, `jyppx/experiment/run.hpp`,
`jyppx/experiment/run_store.hpp`, `jyppx/experiment/run_events.hpp`,
`jyppx/experiment/run_telemetry_sink.hpp`
Shared artifact identity: `jyppx/foundation/artifact.hpp`

Local storage provides artifact bytes, run metadata, run-to-artifact
relationships, and canonical neutral telemetry. Metadata `RunSink` retains its
narrow lifecycle/attachment role. Independent `RunEventSink` accepts atomic
event batches; `SQLiteRunStore` implements both contracts and exposes bounded
event/metric queries. `SQLiteRunTelemetrySink` adapts the shared foundation
telemetry sink to an existing SQLite run without a training dependency. Source
presence is not validation evidence; chapter gates remain responsible for the
complete acceptance result.

## Store layout and ownership

`SQLiteRunStore` owns one dedicated local directory:

```text
local-runs/
  runs.sqlite3
  runs.sqlite3-wal          # SQLite-managed, when present
  runs.sqlite3-shm          # SQLite-managed, when present
  artifacts/
    objects/aa/<remaining 62 lowercase SHA-256 digits>
    tmp/.upload-<nonce>-<sequence>.part
```

`RunMetadata` carries stable run, project, and experiment IDs, a display name,
status, caller-supplied Unix epoch millisecond timestamps, immutable lineage,
and tags. Project and experiment IDs are metadata fields; there are no
separate Project/Experiment CRUD APIs in this slice. Timestamps and display
names do not replace run identity.

`jyppx::ArtifactDigest` and `jyppx::ArtifactRef` are shared foundation types.
The digest algorithm is currently SHA-256 with exactly 64 lowercase hex
digits. An artifact reference contains the digest, byte count, logical name,
media type, and producer. `RunArtifactReference` adds a reference ID unique
within one run, `artifact` or `checkpoint` kind, semantic role, and attachment
timestamp. A `checkpoint` relationship identifies content; this store neither
serializes checkpoint participants nor performs training restore.

## Creating a run and publishing content

```cpp
#include <jyppx/experiment/run_store.hpp>

#include <filesystem>
#include <span>
#include <string>

int main() {
  using namespace jyppx::experiment;

  SQLiteRunStoreOptions options;
  SQLiteRunStore runs(std::filesystem::path("local-runs"), options);
  RunSink& sink = runs;

  RunMetadata request;
  request.run_id = "run-001";
  request.project_id = "project-main";
  request.experiment_id = "experiment-baseline";
  request.display_name = "Local baseline";
  request.status = RunStatus::running;
  request.created_at_unix_ms = 100;
  request.updated_at_unix_ms = 100;
  request.lineage.source_digest = "source-identity";
  request.lineage.build_fingerprint = "build-identity";
  request.lineage.config_digest = "redacted-config-identity";
  request.lineage.data_manifest_digest = "dataset-manifest-identity";
  request.tags.emplace("recipe", "baseline");
  sink.start(request);

  // The producer owns serialization; these are example content bytes.
  const std::string payload = "serialized-model-content";
  const auto bytes = std::as_bytes(std::span(payload.data(), payload.size()));
  ArtifactDescriptor descriptor;
  descriptor.logical_name = "model.bin";
  descriptor.producer = "example";
  descriptor.expected_size_bytes = bytes.size();
  const auto reference = runs.put_artifact(
      request.run_id, "model-final", RunArtifactKind::artifact, "model",
      bytes, descriptor, 110);

  sink.transition(request.run_id, RunStatus::completed, 120);
  const auto metadata = runs.find(request.run_id);
  const auto relationships = runs.references(request.run_id);
  return metadata && relationships.size() == 1 &&
                 relationships.front().artifact.digest == reference.artifact.digest
             ? 0
             : 1;
}
```

`put_artifact` accepts an in-memory byte span. For streaming input, use
`FileArtifactStore::begin`, repeatedly call `ArtifactUploadSession::write`,
and pass that same session to `commit`. An optional expected digest on
`commit` or `put_artifact` checks the producer's assertion. A session belongs
to its creating store handle, which must outlive it. Commit/abort closes the
session; destroying an uncommitted session attempts to remove its temporary
file. Store handles cannot be copied or moved.

For standalone CAS use, construct `FileArtifactStore(root, limits)` and use
`put` for an in-memory blob, or the streaming protocol above. `open(digest)`
verifies the full blob before returning an owning `ArtifactReader`;
`read(destination)` returns the number of bytes read, and zero for EOF or an
empty destination. `exists(digest)` checks content validity, rather than only
path presence. Malformed digests are errors; missing or corrupt content
returns false, while some managed-directory inspection failures can throw.
`verify()` scans canonical objects and reports corrupt and unexpected regular
files without repairing them.

## Lineage, lifecycle, and replay

Lineage has five optional nonempty text references: `parent_run_id`,
`source_digest`, `build_fingerprint`, `config_digest`, and
`data_manifest_digest`. They are caller-owned provenance identities, not an
automatic environment/Git collector. The store does not resolve these strings
or enforce a parent-run foreign key. In particular, a lineage reference is
not an `ArtifactDigest` and does not acquire SHA-256 validation merely from
its field name. Store only appropriately redacted configuration/provenance.

The lifecycle is:

```text
created -> running -> completed | failed | cancelled | interrupted
```

`start` accepts `created` or `running`. A created run must enter running
before a terminal state. Repeating the current target status is idempotent:
`transition` returns the persisted state and does not replace its activity
timestamp. A new transition requires a timestamp at least as recent as the
last activity. `heartbeat` updates only running runs and also requires a
nondecreasing timestamp. New attachments require a timestamp at or after run
creation and advance activity to the greater of its old and attachment time.

Run ID is the start idempotency identity. An exact replay of the original
start fields returns the current stored run, including subsequent status and
activity changes. Changing the original fields, initial status, initial
timestamp, lineage, or tags under that ID fails. There is no general metadata
update method.

Reference ID is the attachment idempotency identity within a run. Exact
replay compares kind, role, content identity, size, producer metadata, and
attachment time. Reusing the ID with different values fails. One `attach`
batch commits all new relationships or none. Terminal runs accept exact
relationship replay but reject new relationships. Replayed relationships do
not force another content check; use `integrity_check()` when a fresh check
of all referenced bytes is required. Roles such as `last` and `best` are text
metadata, not mutable alias pointers.

`find` returns `std::nullopt` for a missing run. `references` requires an
existing run and returns detached records in reference ID order.

## Durability and schema v2

SQLite uses WAL, `synchronous=FULL`, foreign-key enforcement, a full-mutex
connection, and a per-handle mutex. Different handles/processes writing the
same database coordinate through `BEGIN IMMEDIATE`. The default busy timeout
is five seconds; exhaustion reports `NotReady`, so callers can retry the
same operation identity. The directory must permit creation of SQLite's WAL
and shared-memory files and support WAL; this is a local store, with no
network filesystem or remote-service guarantee.

The current database has `user_version=2` and application ID `0x4a595250`.
A new empty database is initialized transactionally. Foreign application
identifiers, future/unsupported versions, and nonempty unversioned databases
are refused. Opening prepares the supported run/reference queries; schema or
record validation failures are surfaced at open or the relevant operation,
instead of treating existing data as a fresh database. Recognized schema-v1
stores can be upgraded transactionally when `allow_v1_migration` is true (the
default); this adds empty event tables and preserves existing metadata and
references. It does not fabricate past telemetry, times, or metric definitions.
Disabling migration rejects an old schema explicitly.

`backup(destination)` publishes a consistent database/live-CAS backup to a new
directory. `restore_backup(backup_root, new_root)` validates the backup and
publishes a new store; existing roots are never overwritten. The WAL files
belong to SQLite; copying only a live `runs.sqlite3` file is not a supported
backup procedure. Neutral JSONL/TensorBoard/OTel export is an observation
projection, not a complete restorable store bundle.

CAS publication flushes and closes the upload stream, syncs the temporary
file, hashes its complete contents, checks size and any expected digest, and
publishes by a same-filesystem atomic rename. Relevant directory entries are
synced on POSIX; Windows uses file flushing and write-through publication.
An existing object at the same digest must verify before deduplication is
accepted. Hashing uses bounded 1 MiB heap chunks; it does not load a complete
artifact or require a 1 MiB thread stack allocation. API immutability does
not make files OS read-only; later external modification is detected by
verified open or integrity checks.

## Canonical events, metrics, and snapshots

`append(run_id, batch_key, envelopes, stop, deadline)` validates a complete
bounded input before effects and commits its events, metric definitions,
derived indices, stream state, and receipts in one SQLite transaction. The
run must already exist. Stream/event/idempotency identities, original times,
correlation, typed payload, and metric semantics come from the immutable
foundation envelope. Store-assigned `run_sequence` defines commit pagination;
it never replaces producer sequence, metric step, or observation time.

An exact batch retry returns its original receipts. Stable producer keys with
identical canonical content are duplicates; conflicting content or a producer
sequence gap is an error. New events after a run or producer-stream terminal
are refused while exact replay remains possible. Successful SQLite acceptance
is durable evidence. An interrupted acknowledgement can follow a successful
commit, so callers retain and replay the entire unchanged batch. External
observer effects do not acquire an exactly-once guarantee.

`events()` captures `snapshot_through` on its first page and orders by
`run_sequence > after_sequence` through that fixed boundary. `metrics()` orders
by `(run_sequence, point_index)` and filters an explicit semantic metric ID,
optionally definition version and step kind. Both return an exclusive
`next_after` only when more results exist. Reuse the boundary and cursor on
every subsequent page; concurrent appends are outside the original snapshot.
Neither metric step resets nor out-of-order/absent clocks change this ordering.
The history is append-only through the API; direct SQL mutation is unsupported.

Metric queries decode the authoritative source envelopes and check the derived
index. Returned points preserve unit, aggregation, comparison direction,
parameters, exact context-key set and values, definition digest/version,
step domain/value, optional sample count, finite/NaN/positive-infinity/
negative-infinity/missing status and reason. They also retain producer
identity, clocks, correlation, envelope annotations, and resource identity.
There is no implicit reaggregation, unit scaling, missing-value replacement,
or load-time timestamp.

The product CLI exposes `run events`, `run metrics`, and `run export`; see
[run telemetry commands](run_telemetry_cli.md) for complete options, output
schemas, subprocess configuration, deadlines, and omission exit codes. Query
pages contain canonical hex and SHA-256 provenance. Export reads a fixed
SQLite event snapshot into a bounded JSONL observer or the official
TensorBoard/OTel sidecar. Destination paths, executable/argv, environment, and
collector settings are explicit caller configuration and never come from
event attributes or payloads.

## CAS publication and GC coordination

Use `SQLiteRunStore::put_artifact` for coordinated publication plus attachment.
It holds the database writer lock across CAS publication and the metadata
transaction. `attach` holds the same lock while verifying newly referenced
objects and committing the batch. `collect_garbage` obtains that lock before
reading the live digest set and deleting unreferenced objects. It refuses
collection when SQLite reports broken foreign-key relationships.

All GC for a `SQLiteRunStore` root must go through that store. Calling raw
`FileArtifactStore::collect_garbage` with an incomplete live set can delete
referenced content. Standalone `FileArtifactStore` does not provide thread or
process coordination; its owner must serialize writers and retention and
provide the full live set.

A streaming producer may use `FileArtifactStore(runs.artifact_root(), limits)`
to publish before `runs.attach(...)`. This leaves a window in which the blob
is still unreferenced and coordinated GC may remove it. Attachment then
fails without confirming the relationship; retry publication/attachment or
serialize this complete sequence externally with retention. Publishing and
attaching are not one atomic transaction across SQLite and the filesystem.

GC defaults to a 24-hour grace period, measured from object modification time;
referenced objects remain live regardless of age. Zero grace immediately
permits deletion of unreferenced canonical objects. Unexpected regular paths
are retained. GC deletes files; it does not purge run/relationship history.
An I/O failure can follow earlier unreferenced-file removals, and SQLite
rollback cannot restore those removed files.

`integrity_check(true)` checks SQLite, foreign keys, and every distinct live
artifact's digest and recorded size. `integrity_check(false)` checks only the
database, relationships, canonical telemetry, and derived event/metric indices.
Reports contain at most 64 issues and do not
repair data; database/query errors can also throw. `FileArtifactStore::verify`
checks the whole CAS, including objects with no run reference.

## Stale state and interrupted attempts

Opening the database does not change running runs: another process may still
own them. The owner sends `heartbeat(run_id, now_ms)` during active work.
An explicit `recover_interrupted(cutoff_ms, interrupted_at_ms)` changes only
running runs whose last activity is at or before the cutoff. The replacement
timestamp must be at or after the cutoff. Created, active, and terminal runs
are retained, and repeating recovery is harmless. The caller decides when a
run is abandoned; the store has no lease owner or automatic liveness probe.

Interrupted is terminal. Resume creates a new run ID with `parent_run_id`
and an explicit checkpoint relationship; loading/restoring the checkpoint
belongs to the training/checkpoint APIs.

`FileArtifactStore::recover()` is separate: it removes regular
`.upload-*.part` files whose modification age reaches `stale_temp_age`
(24 hours by default). It does not recover run status or collect committed
blobs. Age is the only abandonment criterion, so invoke it only after ensuring
matching stale uploads are no longer active and while holding the root's
external retention coordination. A zero age permits immediate removal.
`SQLiteRunStore` currently exposes no automatic temporary-upload recovery API.

## Resource bounds, permissions, and failures

Selected defaults are explicit configuration, not disk-space reservations:

| Option | Default | Meaning |
|---|---|---|
| `ArtifactStoreLimits::max_artifact_bytes` | 1 TiB | One blob's bytes |
| `max_total_bytes` | 4 TiB | Regular object-file bytes under CAS objects |
| `max_object_files` | 1,000,000 | Regular object files under CAS objects |
| `max_root_path_bytes` | 4,096 | Canonical CAS root text bytes |
| logical name / media type / producer bytes | 512 / 256 / 512 | Descriptor field bounds |
| `SQLiteRunStoreOptions::max_runs` | 1,000,000 | Persisted run identities |
| `max_references_per_run` | 16,384 | Distinct relationships per run/batch |
| `max_run_artifact_bytes` | 4 TiB | Sum over a run's relationships |
| reference ID / role bytes | 256 / 128 | Relationship identity/text bounds |
| run ID / display name bytes | 256 / 512 | `RunMetadataLimits` field bounds |
| tag count / key bytes / value bytes | 128 / 128 / 512 | Immutable tag bounds |
| lineage reference bytes | 1,024 | Each optional lineage text reference |
| `event_limits.max_events_per_run` | 1,000,000 | Unique canonical envelopes |
| `max_metric_points_per_run` | 10,000,000 | Derived metric point rows |
| `max_event_bytes_per_run` | 1 GiB | Canonical event bytes |
| `max_page_events` / `max_page_metric_points` | 1,024 / 4,096 | Per-query count ceilings |
| `max_page_bytes` | 16 MiB | Canonical source charge per page |

Two distinct relationships to the same deduplicated blob each consume the
per-run linked-byte quota. CAS total-byte limits exclude temporary uploads,
SQLite/WAL files, and exported copies. SQLite timestamps and stored sizes
must fit a nonnegative signed 64-bit integer; run/reference count options
also have representability bounds. SQLite metadata must be valid UTF-8 without
ASCII controls. Logical name, media type, and producer are descriptive text;
they are not used as storage paths. Media type is bounded nonempty metadata,
not content sniffing. A CAS total-byte limit must contain the configured
maximum single artifact, and required capacity/text bounds must be positive.

Use a dedicated local root with application-appropriate filesystem permissions.
Managed CAS directories/objects and SQLite database paths are checked against
symbolic-link redirection; scans do not follow links. The root is canonicalized.
The store uses the caller's filesystem access and does not add authentication,
encryption, credential storage, or an ACL management API. SHA-256 establishes
content identity and detects changes; it does not grant access or sign content.

| Failure | Observable behavior |
|---|---|
| Malformed metadata, foreign/closed session, conflicting replay, illegal lifecycle | `InvalidArgument` |
| Exceeded upload, field, count, or byte quota | `ResourceExhausted` |
| Expected digest mismatch or blob hash mismatch | `SerializationError` |
| Filesystem access, read/write/flush failure | `IoError` |
| SQLite busy/locked after configured wait | `NotReady` |
| Unsupported/foreign database identity or version | `CompatibilityError` |
| Other SQLite/backend failure | Structured framework `Error`, including `BackendError` |

Some size-validation sites distinguish malformed requests from exhausted
bounds: writing beyond an upload's declared size reports `ResourceExhausted`,
while committing fewer bytes than the declared exact size reports
`InvalidArgument`. SQL constraint/type failures map to `InvalidArgument`;
SQLite corruption maps to `SerializationError`.

On a failed metadata transaction, no new relationship is acknowledged.
A failure after CAS publication can leave an orphan object for later GC;
confirmed relationships and their content are not deleted as rollback cleanup.
An aborted upload attempts temporary cleanup, which can itself fail under
filesystem errors and leave a recoverable stale file. Retain operation IDs
for retries and preserve damaged data for diagnosis instead of rewriting it
as a fresh store.

## Validation evidence

The chapter gate owns compilation and runtime verification. Source/test
presence is not a passing result. Fill these entries after that gate completes:

| Gate | Status / evidence |
|---|---|
| Windows chapter build and CAS/run-store tests | Pending unified validation |
| Ubuntu + CUDA chapter build and runtime tests | Pending remote unified validation |
| Installed `jyppx::experiment` consumer | Pending unified package validation |

Relevant tests are `tests/unit/experiment/test_artifact_store.cpp` and
`tests/unit/experiment/test_run_store.cpp`. Their cases cover content hashing,
multipart streaming, deduplication, limits, stale files, metadata restart,
idempotency, batch rollback, orphan GC, busy writers, independent handles,
explicit interruption recovery, schema refusal, and damaged database/content.
Event persistence/query tests live in `tests/unit/experiment/test_run_events.cpp`;
`tests/cli/test_run_telemetry.py` exercises product queries, immutable snapshot
export, and bounded subprocess launch/protocol failures. Official SDK/collector
interoperability is a separate mandatory gate under
`tests/integration/telemetry_sidecar`. Capacity/overhead measurements and complete
chapter acceptance still need their recorded evidence.
