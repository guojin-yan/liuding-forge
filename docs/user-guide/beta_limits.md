# SDK Beta scope and migration limits

This document describes the current integration candidate. A source file or an
enabled CMake option does not establish a qualified release. Exact platform,
dependency and test evidence belongs to `compatibility.json` and the release
evidence index. The candidate is not a signed v1.0 GA distribution.

## Supported contracts

The SDK uses separate foundation, core, interop, data, nn, optim, training and
experiment targets. Product tools are the `liuding-forge` CLI (with the installed
`jyppx` compatibility command) and the optional official
TensorBoard/OpenTelemetry Python sidecar. Repository development commands live
under `scripts/`; their tests and backend acceptance live under `tests/`.

Local experiment storage uses SQLite WAL/FULL transactions and a content-addressed
artifact store. Its process-crash tests exercise committed and uncommitted
boundaries. They do not simulate a physical power failure, faulty storage firmware
or a network filesystem. A backup is a consistent explicit snapshot; copying an
open SQLite database alone is not the supported backup procedure.

Telemetry distinguishes durable required evidence from sampled observer output.
Required training evidence cannot be downgraded by sampling. A sidecar failure
retains producer/outbox state for explicit replay; delivery to an external backend
does not imply that backend provides exactly-once writes.

Evaluation reports bind their versioned suite, inputs and lineage. A passing
local report does not grant model promotion or deployment authorization.

## Interoperability and trust

Safetensors and model bundles are bounded data formats. Hash verification proves
integrity against the supplied digest. Producer authentication requires an
externally trusted signature verifier. Bundle verification alone does not execute
a model or prove runtime, task-quality or device compatibility.

TorchScript is an existing-artifact compatibility path using the real LibTorch
JIT loader and forward implementation. It requires explicit source trust and a
trusted complete-artifact digest; loading potentially executable model code is
not a sandbox. It does not provide a general tracing, scripting or export API.
Cancellation and deadlines are checked at documented boundaries and cannot
preempt a running CPU operator or CUDA kernel. CUDA operations synchronize before
successful return. See [runtime details](../api/interop_backends.md).

The ONNX/backend protocol supports explicit capability and preflight evidence.
The core SDK does not implement an ONNX protobuf writer or silently claim an
official exporter/ONNX Runtime run. Actual exporter/runtime integration is deferred
to v1.1+ and remains unavailable without independent execution evidence.

## Breaking changes and upgrade procedure

Retired package aliases, global logger, legacy DataLoader/ThreadSafeQueue,
single-optimizer Trainer and distributed placeholders are removed. The current
checkpoint participant, explicit data epoch/cursor, model invocation and optimizer
topology contracts require application review. No forwarding header or blanket
namespace rewrite makes these changes behaviorally equivalent.

Use `liuding-forge api scan --root SOURCE_DIRECTORY --format json` for bounded lexical
diagnostics. It includes comments and inactive code and is not an AST migration
or proof of API/ABI compatibility. Review every suggested change.

Configuration migration uses an application-owned schema identifier, explicit
version chain and typed schema for every step. `config migrate` previews by
default. Apply writes a new directory containing the original bytes, migrated
document and digest-bound receipt; rollback verifies the receipt and writes a new
restored file. It never overwrites the original configuration. Reverse upgrades
require an explicit reverse chain. See [configuration migration](../api/config_migration.md).

Checkpoint readers accept only the explicit current/N-1 schemas documented by
each participant. Unrecognized history and incompatible topology fail before
mutation. Keep the original application, dependency lock, data and checkpoint for
a rollback; do not assume arbitrary older code can read newly written state.

## Deferred and optional functionality

Multi-process distributed training, HPO/recipe orchestration, remote runners,
platform registration/promotion, HTTP/TLS hosting and deployment control are
outside this candidate. macOS qualification is deferred and is not a supported
v1.0 combination. Windows and Ubuntu/CUDA qualification is reported separately.

Models, quantization and serving are independent optional Preview components.
They remain off by default and their individual manifests describe actual
capabilities. Storage quantization does not establish quantized computation or
QAT. A missing official weight/license/hash, runtime or profiler result remains
unavailable rather than being inferred from a descriptor.

The current dependency/license/CVE audit, complete current coverage and
performance review, Beta/RC rollback drill and externally trusted release signing
remain GA gates. Consult the candidate's machine-readable blockers before release.
