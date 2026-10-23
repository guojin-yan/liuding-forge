# Capability Manifest

Header: `jyppx/foundation/capability.hpp`<br>
Target: `jyppx::liuding::foundation`

`CapabilityManifest` is the machine-readable schema-v1 contract for reporting
what an exact JYPPX build and configured runtime object can do. It complements
`BuildFingerprint`: the fingerprint identifies the binary and backend build,
while capability entries describe verified feature scope without inferring
support from a library name or successful compilation.

Each `CapabilityDescriptor` has a unique feature ID and positive feature
version. It separately records evidence status (`available`, `experimental`,
or `unsupported`), implementation kind (`native`, `delegated`, `adapted`, or
`unsupported`), observation source (`build`, `runtime`, or
`configured-instance`), backend, devices, dtypes, schemas, executable
operations, numeric limits, and human-readable limitations.

Use `make_capability_manifest()` with an explicit build fingerprint when
reading stored evidence. Use `make_current_capability_manifest()` for live
diagnostics. Both validate and canonicalize all entries. Duplicate feature
IDs, collection values, and limit names are rejected, as are unsupported
features that claim executable operations. Collections and strings have fixed
resource bounds.

`capability_manifest_json()` validates a copy and emits deterministic compact
JSON. Feature entries, token collections, numeric limits, and limitations are
sorted without modifying the caller's object. Limits remain JSON numbers.

`current_capability_manifest()` queries the exact package build without asking
upper-layer targets to register into foundation. It reports one build-source
entry for each of the twelve entity components. Components disabled at CMake
configure time remain visible as `unsupported`, while enabled components are
`available` except for the experimental data_image, models, quantization and
serving profiles. `current_capability_manifest_json()` exposes the same inventory as
deterministic JSON. Aggregate targets are package conveniences and are not
reported as entity capabilities.

Domain targets do not define competing manifest types. They convert their own
validated descriptors into foundation entries. The first integration is
`describe_data_capability()`, which maps an existing configured
`DataCapability`; future optimizer, scheduler, metric, strategy, runtime, and
tool integrations use the same foundation schema.
