OFFICIAL OBSERVER SIDECAR - OPTIONAL RUNTIME
Date: 2026-10-21 Asia/Shanghai
Status: security remediation lock published; Windows upgraded with pip check,
exact-version and event writer/readback/OTLP import probes passed. Ubuntu
upgrade and consolidated official acceptance are pending. Earlier environment reports
remain historical evidence for their exact lock snapshots. Collector archive
hashes and version probes are recorded separately.

Ownership and deployment

The experiment C++ target has no runtime dependency on Python, TensorBoard,
OTel, or training when the optional observer is not selected. The tool runner
owns one child per logical observer destination and serializes all transact
calls. It must use a fixed argv vector, no shell, an isolated directory and
locked Python environment, bounded pipes, and operation deadline/cancellation
termination. stdout is protocol JSON only. stderr diagnostics are separate,
bounded and redacted. No credential/path/collector configuration comes from an
event. The OS subprocess transport lives in tools/cli/sidecar_process.* and is
qualified by the consolidated Windows/Ubuntu chapter build and fault tests.

Qualification environment

CPython 3.11, Windows amd64 and Ubuntu x86_64. requirements.lock pins direct and
transitive runtime packages, with allowed wheel hashes fetched from official
PyPI metadata. dependency-metadata.json records filename/hash/source/requirements.
The pins include TensorBoard 2.20.0, opentelemetry-proto 1.37.0, protobuf 6.33.5,
Werkzeug 3.1.9, Pillow 13.0.0, and setuptools 83.0.0. The four updated packages
replace versions with advisories found in the 2026-10-21 OSV snapshot. Exact
new wheels, hashes and notices were checked before publishing this lock; an
empty OSV response for a new pin is a dated query result only.
scripts/interop/optional-runtime-provenance.json records the exact interpreters,
pip 25.2, actual selected wheel URLs/hashes, input-lock snapshots and pip report
digests. Full reports live under out/deps/optional-runtime-provenance/.

Prepared interpreters, relative to the corresponding repository root:
  Windows: out/envs/telemetry-py311/Scripts/python.exe (CPython 3.11.9, 64-bit)
  Ubuntu:  out/envs/telemetry-py311/bin/python (CPython 3.11.17, 64-bit)

Windows uses the PSF official NuGet python 3.11.9 portable distribution. Ubuntu
uses Astral python-build-standalone release 20261020, not a PSF Linux binary.
Both archives were checked against pinned upstream hashes. No system Python or
global PATH was changed. pip check reported no broken requirements on either
platform. Imports of official TensorBoard/event_accumulator, generated OTLP
messages, grpc, protobuf and NumPy passed. These are environment checks only.

To reproduce dependency installation in a dedicated CPython 3.11 environment:
  <venv-python> -m pip --isolated install --index-url https://pypi.org/simple
    --require-hashes --only-binary=:all: -r tools/telemetry_sidecar/requirements.lock
  <venv-python> -m pip check
Pass the install arguments together on one command line. Reuse the prepared
environment during chapter validation instead of creating one per build.

The previous target locks were installed with hash enforcement. Their reports
do not establish installation or acceptance of the four new pins. Install the
current lock into the same dedicated environments, record fresh pip reports,
run pip check and exact-version probes, then run the mandatory chapter gate
below. The Windows upgrade report and exact input-lock snapshot are recorded in
scripts/interop/optional-runtime-provenance.json under security_remediation.
The Windows offline hash-enforced dry run is retained under
out/deps/dependency-audit-20261021/remediation/pip-dry-run-windows.json.
Complete distribution licensing/SBOM and the OFF/consumer/package gates remain
part of chapter/release qualification.

The sidecar uses TensorBoard event writing and event_accumulator readback; it
does not launch the TensorBoard web UI. TensorBoard 2.20 UI/server modules
import the retired pkg_resources API and are not qualified with setuptools
83.0.0. Deploying that UI requires a separately qualified upstream toolchain;
do not downgrade a security pin to make an unqualified UI path import.
tests/integration/telemetry_sidecar/collector-artifacts.json separately pins
official otelcol-contrib 0.135.0
Windows/Linux archives, official checksum manifests, actual archive sizes,
verified SHA-256 and executable hashes. Both executables returned
"otelcol-contrib version 0.135.0". Signature/cosign verification was not performed.
Set JYPPX_TEST_OTELCOL to the corresponding prepared executable:
  Windows: out/deps/otelcol-contrib-0.135.0/windows/otelcol-contrib.exe
  Ubuntu:  out/deps/otelcol-contrib-0.135.0/linux/otelcol-contrib

Mandatory chapter gate (separate from dependency preparation):
  <venv-python> tests/integration/telemetry_sidecar/run_acceptance.py

Run this command from the repository root. Acceptance runners, tests, collector
fixtures and collector provenance belong in tests/integration/telemetry_sidecar,
not in this product runtime directory. Keep generated environments under out/.

That entry point requires every pinned runtime package and the actual pinned
collector. It rejects skipped tests. It covers protocol preflight/replay/bounds,
official TensorBoard event accumulator readback, official generated OTLP
schemas, HTTP error/partial/redirect behavior, and a real collector started by
argv in a unique fixture directory. The collector's file exporter output is
read back through official generated protobuf consumers for exact binary64,
timestamp, metric metadata, log severity and mapped span values. This gate has
not been executed by this dependency preparation. C++ exporter tests, bounded
runner faults, Windows/Ubuntu
chapter tests, and observed overhead/capacity gates are additional requirements.

Example future sidecar argv (the runner passes each argument separately):
  python -I jyppx_telemetry_sidecar.py --backend tensorboard
    --tensorboard-output <new-exclusive-output-directory>
  python -I jyppx_telemetry_sidecar.py --backend opentelemetry
    --otlp-base-url http://127.0.0.1:4318 --allowed-host 127.0.0.1
    --resource-instance-id <explicit-source-instance> --http-timeout-seconds 5
    [--clock-mappings <isolated-config.json>]

Configuration is fixed when the process starts. TensorBoard's directory must
not already exist. Collector HTTP is permitted only for a literal loopback
address; non-loopback requires HTTPS and an exact configured hostname match.
Environment proxies and redirects are disabled. The injected OS process runner
must enforce the overall absolute deadline in addition to the HTTP socket bound.

Projection semantics and limits

TensorBoard uses official EventFileWriter/generated schemas, the official text
summary API and the scalar plugin with DT_DOUBLE tensors. It never writes
TFRecord/protobuf bytes itself.
Tags include the full definition digest, exact context, step domain and resource
identity as a hashed series identity. Human-readable definition metadata remains
in the summary description. Scalar batches/resources without wall time, with a
missing scalar, or with an out-of-range signed backend step are explicitly
unsupported. Lifecycle payloads use the official text plugin to store the entire
received neutral envelope as one UTF-8 JSON string tensor: identities, clocks,
original integer ns, attributes, terminal state, metrics and callback decisions
are all retained. The text event's step domain is explicitly telemetry.sequence,
mapped from identity.sequence, not inferred as a training/global step. Its tag
groups by stream, project/run/attempt and lifecycle name. Non-missing lifecycle
metrics additionally use the same DT_DOUBLE scalar projection and their own
declared producer steps. A missing lifecycle metric remains explicit in full
text with its status/reason and produces no fabricated zero or NaN scalar; this
declared text representation preserves the complete lifecycle. An empty-metric
lifecycle still produces its complete text. Missing wall time or an out-of-range
text/metric backend step rejects the whole lifecycle before any write. Other
payloads, including spans, remain explicitly unsupported and counted. No partial
projection is silently represented as complete. TensorBoard wall-time uses its
required floating seconds; canonical SQLite/JSONL remain the durable evidence
source of truth.

OTLP uses official generated protobuf requests for metrics/logs/traces. Scalars
are Gauges with the complete definition metadata; no cumulative sum temporality,
mean statistics, histogram or unit scaling is invented. Missing observations use
the official no-recorded-value flag. The metric series dimensions include only
the declared definition/context/step domain/resource identity. The runner must
explicitly configure service.instance.id to separate source instances. Event,
trace, step value, sample count and timestamp are retained in a companion audit
log, rather than automatically becoming one metric series per observation.
Lifecycle metrics use the same Gauge path, including the official missing-value
flag, followed by the complete lifecycle audit log. Its body is the lifecycle
payload; jyppx.envelope retains the complete received neutral envelope JSON,
including monotonic clock provenance and decisions. Any lifecycle with metrics
requires a real nonnegative wall timestamp; without it the whole event is
unsupported and neither signal is written. An empty-metric lifecycle needs only
the log path and can explicitly represent missing time. Neither path invents a
current time or an implicit metric step.
Gauge and audit-log writes are separate external effects; partial success can
cause duplicate collector points on retry. A failure in either signal does not
acknowledge or increment completed-event counters for the envelope. This remains
at-least-once delivery.

Lifecycle, log and artifact-reference projections use OTLP logs. Missing wall
time is explicitly annotated; no current timestamp is invented. Spans require
an explicit per-clock affine anchor in --clock-mappings:
  {"clock.test":{"monotonic_ns":100,"unix_ns":1700000000000000000}}
Mappings are configuration evidence, never inferred from unrelated clocks.
Without a mapping a span is unsupported. Opaque native IDs map by a fixed
domain-separated SHA-256 truncation to official widths, with native provenance
retained as attributes. Unsupported schemas, invalid bounded data and backend
errors are distinct from sampling. No sampling policy is implicit.

The process bounds line bytes (16 MiB), response bytes (128 KiB), nesting (32),
JSON nodes (200000), batch events (1024), metric points (4096/envelope), and its
immutable replay index (100000 entries/32 MiB charged key state). Capacity is
explicit and errors never echo source/credentials. External official SDK memory,
series cardinality, queue saturation and measured overhead still need chapter
benchmarks. The canonical content digest is computed/verified by the C++ neutral
contract; the sidecar is a trusted projection consumer, not a second canonical
codec or a replacement durable evidence store.
