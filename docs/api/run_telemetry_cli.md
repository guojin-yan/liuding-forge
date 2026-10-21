# CLI: run events, metrics, and export

These product commands read existing local `runs.sqlite3` evidence. They do not
create missing storage or fabricate observations. Paths are UTF-8 and converted
with `std::filesystem::u8path`, including Windows Unicode paths. Append events
through `SQLiteRunStore`/`SQLiteRunTelemetrySink` from a producer; this CLI has no
unvalidated JSON-to-database event injection command.

All commands accept `--format json|text` (text renders nested field/value rows), `--root PATH`,
`--id RUN_ID`, `--timeout-ms N` (default 60,000, maximum 600,000), and
`--page-bytes N` (default/maximum 16 MiB, minimum 1 MiB). The timeout is one
absolute steady-clock deadline across open, reads, export, and close. It is not
an observation timestamp. SQLite waits and subprocess exchanges observe this
deadline; operating-system file I/O is checked before/after bounded steps and
cannot promise interruption of an indefinitely stalled filesystem call.

## Fixed snapshot queries

```text
jyppx run events --root local-runs --id run-001 --limit 128 --format json
jyppx run events --root local-runs --id run-001 --after 128 --snapshot 600 --limit 128 --format json

jyppx run metrics --root local-runs --id run-001 --metric validation.accuracy --limit 128 --format json
jyppx run metrics --root local-runs --id run-001 --metric validation.accuracy --after-sequence 80 --after-index 2 --snapshot 600 --limit 128 --format json
```

The first query without a nonzero cursor captures and returns
`data.snapshot_through`. Continue with that exact `--snapshot` and the returned
`data.next_after` until it is null. CLI continuation with a nonzero cursor
requires an explicit snapshot to avoid accidentally including concurrent
appends. Keep the same run and filters throughout the scan. A supplied boundary
must not be in the future, and a cursor must not be beyond it. Snapshot zero is
valid for an empty history. These are logical append-only snapshots, with no
long-running SQLite reader held between invocations; direct database mutation
is unsupported.

| Command | Extra options |
|---|---|
| `events` | `--after N` (exclusive commit sequence, default 0), `--snapshot N`, `--limit N` (default 256, maximum 1,024), `--kind NAME` |
| `metrics` | required `--metric ID`, `--after-sequence N` and `--after-index N` (both default 0), `--snapshot N`, `--limit N` (default 256, maximum 4,096), `--definition-version N`, `--step-kind NAME` |

Event kinds are `lifecycle`, `metric_batch`, `log`, `span`, `resource_sample`,
and `artifact_reference`. Step kinds are `micro_step`, `global_step`, `epoch`,
`sample`, and `elapsed_nanoseconds`. Definition version is a positive uint32.
Cursors are nonnegative signed-64-bit-representable database coordinates;
metric step values retain their full original uint64 range. Count and
canonical-source byte bounds both apply. CLI JSON content is additionally
capped at 64 MiB per response; reduce the page count/byte budget on exhaustion.

Event records in `data.events` contain `schema_version`, `run_sequence`,
`content_digest` (SHA-256), `canonical_hex`, and the shared neutral JSON
`event` projection. The canonical bytes remain the hash/persistence authority;
JSON number formatting is not the canonical codec.

Metric records in `data.points` contain:

- `cursor: {run_sequence, point_index}` and `source_content_digest`.
- Original event schema, identity, reliability, clocks, correlation,
  annotations, and optional `resource_id`.
- `point`: complete immutable definition with digest, units, aggregation,
  direction, parameters, context keys/values, step kind/value, value status,
  numerical value/reason, and optional sample count.
- `status_name`: `finite`, `not_a_number`, `positive_infinity`,
  `negative_infinity`, or `missing`.

`data.source_events` contains each referenced canonical event record once per
page. This preserves event-level provenance without copying a complete event
beside every point. NaN/Inf/missing observations have JSON `value: null` and
retain their distinct status. Missing clocks remain null, negative wall times
remain signed integers, sample counts are not inferred, and no reaggregation,
step sorting, unit conversion, or reset stitching occurs.

Projection enum numbers are the persisted foundation schema: units 1–9
(`dimensionless`, `count`, `seconds`, `bytes`, `ratio`, `percent`,
`count_per_second`, `bytes_per_second`, `unspecified`); aggregations 1–6 (`last`,
`sum`, `mean`, `minimum`, `maximum`, `unspecified`); directions 1–3 (`none`,
`minimize`, `maximize`); statuses 1–5 in the order listed above; step kinds 1–5
in the order listed above. IDs and complete definition digests distinguish
incompatible series.

## Export from authoritative evidence

```text
jyppx run export --root local-runs --id run-001 --backend jsonl --output exported.jsonl --format json
jyppx run export --root local-runs --id run-001 --backend jsonl --output exported.jsonl --resume true --snapshot 600 --format json
jyppx run export --root local-runs --id run-001 --backend tensorboard --sidecar-config tensorboard.json --format json
jyppx run export --root local-runs --id run-001 --backend opentelemetry --sidecar-config otel.json --format json
```

Export captures a boundary once and reads only original canonical SQLite
events through it, using keyset pages. `--snapshot N` selects an explicit
already committed boundary; `--after N` selects an exclusive starting
coordinate and requires the snapshot when nonzero. There are no implicit
event/metric filters or transformations. `--page-size N` defaults to 256 and
is capped at 1,024. `--max-events N` defaults to/cannot exceed 1,000,000; the
selected unfiltered coordinate range is checked before creating output or
launching a process. Protocol batches obey count, canonical-byte and projected
request-byte bounds independently of database page size.

JSONL requires `--output PATH` outside the managed SQLite/CAS root. The caller
owns this path exclusively; concurrent writers or replacement are unsupported.
Existing paths are refused unless `--resume true`. Resume accepts only a
regular file with one filesystem link and at least one complete record (or an
empty file). The observer verifies every complete record's canonical bytes,
digest and projection, then truncates only an incomplete final tail. A nonempty
file with no complete record is retained and rejected for diagnosis. Replaying
matching source events returns duplicates rather than adding lines. A resumed
file may already contain other valid observations; replay does not claim the
whole destination is an exact standalone copy of the selected run snapshot.

JSONL records use the shared schema (`schema_version`, `content_digest`,
`canonical_hex`, `event`). Their default bounds are 1 GiB/file, 1,000,000 records,
and 4 MiB/line. Successful close flushes the stream; filesystem fsync durability
is not claimed. Durable acceptance remains SQLite's responsibility.

Official backends use `--sidecar-config PATH` instead of `--output/--resume`.
The config is a strict JSON object, capped at 256 KiB, schema 1. It selects the
executable and exact argv, including backend destinations; no value is read
from an event or converted into shell syntax. Example TensorBoard config:

```json
{
  "schema_version": 1,
  "backend": "tensorboard",
  "executable": "C:/telemetry-venv/Scripts/python.exe",
  "arguments": [
    "-I", "-u", "C:/jyppx/tools/telemetry_sidecar/jyppx_telemetry_sidecar.py",
    "--backend", "tensorboard",
    "--tensorboard-output", "C:/exports/tensorboard-run-001"
  ],
  "working_directory": "C:/exports/isolated-sidecar-cwd",
  "inherit_environment": false,
  "environment": {},
  "operation_timeout_ms": 10000,
  "startup_timeout_ms": 10000,
  "cleanup_timeout_ms": 2000
}
```

Executable and working directory must be absolute; the working directory must
already exist and be reserved by the caller. Script/output/clock-map arguments
should be explicit absolute paths. A TensorBoard destination must not exist.
The config backend must match `--backend`; the supplied process is trusted to
implement that backend's official protocol. Use the delivered sidecar and its
pinned requirements, not an arbitrary unverified executable.

For OTel, use `backend: "opentelemetry"`, the corresponding fixed `--backend`
argv, and explicitly configured `--otlp-base-url`, `--allowed-host`,
`--resource-instance-id`, optional `--clock-mappings`, and
`--http-timeout-seconds`. For example the destination argv may include
`--otlp-base-url http://127.0.0.1:4318 --allowed-host 127.0.0.1
--resource-instance-id worker-001`, passed as six separate strings. The
official sidecar disables proxies/redirects and permits plain HTTP only for a
literal loopback address. No environment/clock/collector setting is inferred
from stored observations.

| Optional sidecar config field | Default / maximum |
|---|---|
| `environment` | child-only string map, at most 128 entries |
| `inherit_environment` | `false`; `true` explicitly copies parent environment before overrides |
| `operation_timeout_ms`, `startup_timeout_ms` | 10,000 / 600,000 |
| `cleanup_timeout_ms` | 2,000 / 10,000 |
| `max_request_bytes` | 16 MiB / 16 MiB, minimum 1,024 |
| `max_reply_bytes` | 128 KiB / 128 KiB |
| `max_stderr_bytes` | 64 KiB / 64 KiB cumulative |

All numeric config limits are positive integers. Argv is capped at 256 entries,
with a combined native launch bound of 256 KiB from the process transport.
Unknown fields and duplicate keys are errors. The process receives no shell,
PATH lookup, parent cwd/environment mutations, or inherited arbitrary handles.
Its stdout is protocol-only; stderr is discarded under its quota and only the
byte count is reported. The earlier of the absolute command deadline and each
configured process-operation deadline wins. Failure poisons/terminates the
process tree. Cleanup may extend the operation deadline by the bounded cleanup
grace. See [process transport](telemetry_sidecar_process.md) for platform
containment and ownership details.

## Results, omissions, and retries

Successful command execution returns the normal outer JSON envelope with
`status: "ok"`. Inspect `data.scan_completed`, `fully_represented`,
`snapshot_through`, `last_after`, `read_events`, `accepted_events`,
`duplicate_events`, `sampled_out_events`, `unsupported_events`, and cumulative
`destination_counters`. JSONL additionally reports verified opening records
and truncated tail bytes; official exports report bounded transport byte and
request counts. Final counters must agree with receipts from this invocation.

| Exit code | Meaning |
|---|---|
| 0 | Query complete, or export complete with no omissions |
| 1 | Export scanned/closed successfully but some observations were unsupported or sampled out |
| 2 | Invalid option, schema/configuration, corrupt serialization, or compatibility failure |
| 3 | I/O, quota, process/backend, or other runtime failure |
| 124 | Command or sidecar operation deadline reached |
| 130 | Cancellation propagated by the operation layer |

An observer omission does not modify the durable run or its outcome. The
official TensorBoard mapping explicitly rejects payloads/scalars it cannot
represent, including missing wall time or missing numerical values. OTel spans
require explicit clock anchors; unsupported spans are counted. Full backend
mapping semantics are documented with the delivered sidecar.

A failure may occur after a JSONL prefix or external collector/SDK effects were
written. Failure output does not claim a complete scan, rollback, or final
counter proof. Keep SQLite evidence and retry the same boundary and unchanged
envelopes after repairing the destination. JSONL can verify/deduplicate its
complete recovered records. A new official process has no persistent replay
index; collector writes (including OTel metric and companion audit-log writes)
can duplicate across retries. The reported delivery model is at least once,
with explicit omission accounting, never exactly-once external effects.

`tests/cli/test_run_telemetry.py --cli ABSOLUTE_EXECUTABLE` checks actual CLI
queries, canonical provenance, concurrent append boundaries, JSONL replay,
explicit launch configuration and controlled real-process faults. The
controlled peer does not replace official SDK readback or collector tests;
`tests/integration/telemetry_sidecar/run_acceptance.py` is a separate mandatory
chapter gate. These files are implementation/test definitions, not a claim
that either platform has passed.
