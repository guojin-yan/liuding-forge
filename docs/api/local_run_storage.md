# API: Local Run Storage

Target: `jyppx::liuding::experiment` Headers: `jyppx/experiment/artifact_store.hpp`, `jyppx/experiment/run.hpp`,
`jyppx/experiment/run_store.hpp`, `jyppx/experiment/run_events.hpp`,
`jyppx/experiment/run_telemetry_sink.hpp`
Shared artifact identity: `jyppx/foundation/artifact.hpp`

Local storage provides artifact bytes, run metadata, run-to-artifact relationships, and canonical neutral telemetry. Metadata `RunSink` retains its narrow lifecycle/attachment role. Independent `RunEventSink` accepts atomic event batches; `SQLiteRunStore` implements both contracts and exposes bounded event/metric queries. `SQLiteRunTelemetrySink` adapts the shared foundation telemetry sink to an existing SQLite run without a training dependency. Source presence is not validation evidence; chapter gates remain responsible for the complete acceptance result.

## Store layout and ownership

`SQLiteRunStore` owns one dedicated local directory:

```text
local-runs/ runs.sqlite3 runs.sqlite3-wal          # SQLite-managed, when present runs.sqlite3-shm          # SQLite-managed, when present artifacts/
    objects/aa/<remaining 62 lowercase SHA-256 digits>
    tmp/.upload-<nonce>-<sequence>.part
```

`RunMetadata` carries stable run, project, and experiment IDs, a display name,
status, caller-supplied Unix epoch millisecond timestamps, immutable lineage, and tags. Project and experiment IDs are metadata fields; there are no separate Project/Experiment CRUD APIs in this slice. Timestamps and display names do not replace run identity.

`jyppx::liuding::ArtifactDigest` and `jyppx::liuding::ArtifactRef` are shared foundation types.
The digest algorithm is currently SHA-256 with exactly 64 lowercase hex
digits. An artifact reference contains the digest, byte count, logical name,
media type, and producer. `RunArtifactReference` adds a reference ID unique
within one run, `artifact` or `checkpoint` kind, semantic role, and attachment
timestamp. A `checkpoint` relationship identifies content; this store neither
serializes checkpoint participants nor performs training restore.

## Creating a run and publishing content

```cpp
#include <jyppx/liuding/experiment/run_store.hpp>

#include <filesystem> #include <span> #include <string>

int main() { using namespace jyppx::liuding::experiment;

SQLiteRunStoreOptions options; SQLiteRunStore runs(std::filesystem::path("local-runs"), options); RunSink& sink = runs;

RunMetadata request; request.run_id = "run-001"; request.project_id = "project-main"; request.experiment_id = "experiment-baseline"; request.display_name = "Local baseline"; request.status = RunStatus::running; request.created_at_unix_ms = 100; request.updated_at_unix_ms = 100; request.lineage.source_digest = "source-identity"; request.lineage.build_fingerprint = "build-identity"; request.lineage.config_digest = "redacted-config-identity"; request.lineage.data_manifest_digest = "dataset-manifest-identity"; request.tags.emplace("recipe", "baseline"); sink.start(request);

// The producer owns serialization; these are example content bytes. const std::string payload = "serialized-model-content"; const auto bytes = std::as_bytes(std::span(payload.data(), payload.size())); ArtifactDescriptor descriptor; descriptor.logical_name = "model.bin"; descriptor.producer = "example"; descriptor.expected_size_bytes = bytes.size(); const auto reference = runs.put_artifact(
      request.run_id, "model-final", RunArtifactKind::artifact, "model",
      bytes, descriptor, 110);

sink.transition(request.run_id, RunStatus::completed, 120); const auto metadata = runs.find(request.run_id); const auto relationships = runs.references(request.run_id); return metadata && relationships.size() == 1 &&
                 relationships.front().artifact.digest == reference.artifact.digest
             ? 0
             : 1;
}
```

`put_artifact` accepts an in-memory byte span. For streaming input, use
`FileArtifactStore::begin`, repeatedly call `ArtifactUploadSession::write`,
and pass that same session to `commit`. An optional expected digest on
`commit` or `put_artifact` checks the producer's assertion. A session belongs
to its creating store handle, which must outlive it. Commit/abort closes the session; destroying an uncommitted session attempts to remove its temporary file. Store handles cannot be copied or moved.

For standalone CAS use, construct `FileArtifactStore(root, limits)` and use
`put` for an in-memory blob, or the streaming protocol above. `open(digest)`
verifies the full blob before returning an owning `ArtifactReader`;
`read(destination)` returns the number of bytes read, and zero for EOF or an
empty destination. `exists(digest)` checks content validity, rather than only path presence. Malformed digests are errors; missing or corrupt content returns false, while some managed-directory inspection failures can throw.
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

For a successfully published local output file, `record_published_file_evidence` accepts a stable run/reference ID, the file path, exact size and SHA-256, descriptor, attachment time, event batch key, and a telemetry envelope whose artifact payload contains only the role. It streams the file into CAS, attaches the run reference, then appends the completed artifact event. The two commits are separate: if the event fails, retain the same request identity and retry with `path` omitted. `find_reference(run_id, reference_id)` uses the run-local primary key to inspect the durable relationship without loading every reference or requiring the original file. A retry rejects changed digest, size, descriptor, role, kind, or attachment time; event retries still require the original stream, sequence, batch key, and envelope fields. The event creates a second run reference, so allow for both references and their linked bytes in the run quotas. Neither this API nor a successful CAS copy measures end-to-end training speed, RSS, or GPU memory.

需要运行引用与制品事件同时持久化时，使用 `record_published_file_evidence_atomic(request, stop, deadline)`，请求字段与上述接口相同。该方法持有同一个 SQLite 写事务完成流式 CAS 复制、运行引用登记、事件及其 GC 引用追加；事件冲突、配额或取消导致的失败会回滚本次新增的全部数据库关系，已确认的旧引用保留。CAS 中可能留下未引用对象，可由该 store 的 GC 清理；原文件在调用前已经发布，不属于数据库事务。成功后即使原文件移走，也可保留全部请求身份并省略 `path` 重放；联合提交失败且引用未持久化时，重试仍需原文件。每次成功操作占用两个运行引用及相应字节配额。

提交阶段出现 I/O 或同步错误时，调用方可能没有收到成功回执，而完整事务已经写入 WAL；这类错误的提交结果未知，不能据异常认定数据库未写入。停止使用发生错误的句柄，重新打开存储，以原有 run、reference、batch、stream、event、sequence、摘要和描述符重试；已提交的请求会返回原回执，未提交的请求会重新提交，两种情况都不会产生半个联合提交或重复事件。在确认引用前保留原文件及完整请求身份。


## 可恢复的最终文件发布

需要在生产者进程和输入文件丢失后继续发布输出时，使用 `FilePublicationRequest`、`prepare_file_publication` 和 `recover_file_publication`。请求包含本 run 内稳定的 `publication_id`、最终 `destination` 和原 `PublishedFileEvidenceRequest`；首次准备时 `evidence.path` 指向输入文件，描述符包含精确大小和摘要。父目录须已存在、由调用方控制，最终路径位于 run 存储之外；已有不同内容、重定向路径或身份冲突会被拒绝。

`prepare_file_publication(request, stop, deadline)` 在同一 SQLite 写事务里将输入内容和完整意图各登记为一个普通 run 引用。意图保存规范事件、原描述符、时间戳、目标路径及结果身份；成功后可删除输入文件。准备阶段不会发布最终路径或结果事件，失败时新增的两个引用一起回滚；未引用 CAS 字节仍可由协调 GC 清理。意图最多 4 MiB，内容沿用制品字节上限；最终复制按 64 KiB 分块，支持取消和截止时间。重试准备须使用原意图；已成功准备后可省略 `evidence.path`。

编码结果已在内存时，将 `evidence.path` 留空，调用 `prepare_file_publication_bytes(request, encoded.bytes, stop, deadline)`。接口借用缓冲区到返回，按 64 KiB 分块写入 CAS，无需额外的输入文件或完整副本。首次准备须匹配精确大小和 SHA-256；相同已准备意图重放可传空 span，仍核验 CAS。文件与字节入口使用相同的持久格式和恢复接口。

重启后用 `file_publications(run_id, after_id, limit)` 分页发现意图 ID。每页 1–1024 项，按 ID 升序排列，`after_id` 为排他游标；已完成的意图也保留，适合统一幂等重放。恢复入口只需要 run ID 和意图 ID，直接读取原持久化请求，不重新执行生产者：

```cpp
auto ids = runs.file_publications("run-1");
for (const auto& id : ids) {
  auto confirmed = runs.recover_file_publication("run-1", id);
  // confirmed.path and confirmed.evidence describe the complete publication.
}
```

`recover_file_publication` 持有 SQLite 写锁，核验意图和 CAS，先在未提交事务里检查并登记结果引用与事件，再将 CAS 内容写入同目录专用暂存、同步并以不覆盖方式发布最终文件，最后提交 SQLite。现有文件必须完整匹配大小和摘要；匹配重放保留文件身份及修改时间。进程在文件发布后、数据库提交前退出时，意图和内容仍由原引用保护，下一次恢复确认文件并补交数据库。提交已成功但回执丢失时，相同身份返回原事件回执。不同连接按数据库写锁串行，协调 GC 无法进入复制和提交窗口。

准备阶段占两个 run 引用；完整提交后共占四个引用，分别为内容、意图、结果和事件，各自计入链接字节配额。内容即使去重也按每个引用计费，意图和内容保留供后续恢复、GC 和备份。首次完成沿用 run 的可写生命周期，调用方应先完成输出再把 run 标记为终态；终态只允许已登记证据的幂等重放。`backup` 包含这些引用及对应 CAS，`restore_backup` 不自动写入外部文件。意图保留原绝对目标路径，恢复不会重新定位目录；只对受信任的本地存储显式执行恢复。专用 `.liuding-publication-<意图摘要>.stage` 名称保留给该协议，进程中断后的文件仅按原意图恢复，不扫描删除其他暂存。

最终发布要求本地文件系统支持同目录硬链接；Windows 暂存句柄使用关闭时删除标志，成功创建的最终硬链接独立保留。成功回执确认文件与数据库证据均已完成。原始文件可能在 SQLite 提交前被直接观察到，任意文件读取者和数据库之间仍无瞬时共同可见性；数据游标不属于这个事务，进程强杀验证也不等同断电恢复资格。`ImageOutputSink` 回调可以只执行准备，此时回调 future 只确认准备，完整输出须等待恢复回执。可运行示例见[图像文件发布](../../examples/experiment/PUBLISHED_FILE.md)。

## Durability and schema v2

SQLite uses WAL, `synchronous=FULL`, foreign-key enforcement, a full-mutex connection, and a per-handle mutex. Different handles/processes writing the same database coordinate through `BEGIN IMMEDIATE`. The default busy timeout is five seconds; exhaustion reports `NotReady`, so callers can retry the same operation identity. The directory must permit creation of SQLite's WAL and shared-memory files and support WAL; this is a local store, with no network filesystem or remote-service guarantee.

The current database has `user_version=2` and application ID `0x4a595250`. A new empty database is initialized transactionally. Foreign application identifiers, future/unsupported versions, and nonempty unversioned databases are refused. Opening prepares the supported run/reference queries; schema or record validation failures are surfaced at open or the relevant operation, instead of treating existing data as a fresh database. Recognized schema-v1 stores can be upgraded transactionally when `allow_v1_migration` is true (the default); this adds empty event tables and preserves existing metadata and references. It does not fabricate past telemetry, times, or metric definitions. Disabling migration rejects an old schema explicitly.

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
bounded input before effects and commits its events, metric definitions, derived indices, stream state, and receipts in one SQLite transaction. The run must already exist. Stream/event/idempotency identities, original times, correlation, typed payload, and metric semantics come from the immutable foundation envelope. Store-assigned `run_sequence` defines commit pagination; it never replaces producer sequence, metric step, or observation time.

An exact batch retry returns its original receipts. Stable producer keys with identical canonical content are duplicates; conflicting content or a producer sequence gap is an error. New events after a run or producer-stream terminal are refused while exact replay remains possible. Successful SQLite acceptance is durable evidence. An interrupted acknowledgement can follow a successful commit, so callers retain and replay the entire unchanged batch. External observer effects do not acquire an exactly-once guarantee.

`events()` captures `snapshot_through` on its first page and orders by
`run_sequence > after_sequence` through that fixed boundary. `metrics()` orders
by `(run_sequence, point_index)` and filters an explicit semantic metric ID, optionally definition version and step kind. Both return an exclusive
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
referenced content. Standalone `FileArtifactStore` does not provide thread or process coordination; its owner must serialize writers and retention and provide the full live set.

A streaming producer may use `FileArtifactStore(runs.artifact_root(), limits)` to publish before `runs.attach(...)`. This leaves a window in which the blob is still unreferenced and coordinated GC may remove it. Attachment then fails without confirming the relationship; retry publication/attachment or serialize this complete sequence externally with retention. Publishing and attaching are not one atomic transaction across SQLite and the filesystem.

GC defaults to a 24-hour grace period, measured from object modification time; referenced objects remain live regardless of age. Zero grace immediately permits deletion of unreferenced canonical objects. Unexpected regular paths are retained. GC deletes files; it does not purge run/relationship history. An I/O failure can follow earlier unreferenced-file removals, and SQLite rollback cannot restore those removed files.

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

Two distinct relationships to the same deduplicated blob each consume the per-run linked-byte quota. CAS total-byte limits exclude temporary uploads, SQLite/WAL files, and exported copies. SQLite timestamps and stored sizes must fit a nonnegative signed 64-bit integer; run/reference count options also have representability bounds. SQLite metadata must be valid UTF-8 without ASCII controls. Logical name, media type, and producer are descriptive text; they are not used as storage paths. Media type is bounded nonempty metadata, not content sniffing. A CAS total-byte limit must contain the configured maximum single artifact, and required capacity/text bounds must be positive.

Use a dedicated local root with application-appropriate filesystem permissions. Managed CAS directories/objects and SQLite database paths are checked against symbolic-link redirection; scans do not follow links. The root is canonicalized. The store uses the caller's filesystem access and does not add authentication, encryption, credential storage, or an ACL management API. SHA-256 establishes content identity and detects changes; it does not grant access or sign content.

| Failure | Observable behavior |
|---|---|
| Malformed metadata, foreign/closed session, conflicting replay, illegal lifecycle | `InvalidArgument` |
| Exceeded upload, field, count, or byte quota | `ResourceExhausted` |
| Expected digest mismatch or blob hash mismatch | `SerializationError` |
| Filesystem access, read/write/flush failure | `IoError` |
| SQLite busy/locked after configured wait | `NotReady` |
| Unsupported/foreign database identity or version | `CompatibilityError` |
| Other SQLite/backend failure | Structured framework `Error`, including `BackendError` |

Some size-validation sites distinguish malformed requests from exhausted bounds: writing beyond an upload's declared size reports `ResourceExhausted`, while committing fewer bytes than the declared exact size reports
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
| Installed `jyppx::liuding::experiment` consumer | Pending unified package validation |

Relevant tests are `tests/unit/experiment/test_artifact_store.cpp` and
`tests/unit/experiment/test_run_store.cpp`. Their cases cover content hashing,
multipart streaming, deduplication, limits, stale files, metadata restart, idempotency, batch rollback, orphan GC, busy writers, independent handles, explicit interruption recovery, schema refusal, and damaged database/content. Event persistence/query tests live in `tests/unit/experiment/test_run_events.cpp`;
`tests/cli/test_run_telemetry.py` exercises product queries, immutable snapshot
export, and bounded subprocess launch/protocol failures. Official SDK/collector
interoperability is a separate mandatory gate under
`tests/integration/telemetry_sidecar`. Capacity/overhead measurements and complete
chapter acceptance still need their recorded evidence.
