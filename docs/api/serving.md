# Serving Preview

`jyppx::liuding::serving` is an opt-in Preview component. It depends on foundation,
core, interop and nn. It has no dependency on data, training or the models
catalog. The component is absent from the default build and default umbrella.
Its versioned capability manifest explicitly excludes HTTP/gRPC listeners, TLS,
ONNX Runtime, streaming, reload, deployment control planes and GA qualification.

The native path uses actual safetensors state and an explicit C++ model factory.
The optional compatibility path really loads and invokes an explicitly trusted
TorchScript archive through the existing interop runtime. A digest establishes
integrity; authentication requires the configured bundle signature policy or the
separate explicit JIT source-trust decision.

## Loading and immutable ownership

`NativeStateBackend::load` verifies the complete model bundle inventory, paths,
sizes, hashes, safetensors mappings and configured signature policy. It compares the exact backend ID/version, environment-lock digest, dependency list, CPU/CUDA family, dtypes, dynamic-shape requirement and current platform token. The LibTorch dependency must match the linked build version.
`serving_platform_id()` supplies the lowercased configure-time OS/architecture
token expected in the bundle matrix. The caller attests the environment-lock
identity; this code does not run a package manager or reconstruct an environment.

The backend snapshots every verified data file and rechecks its size and digest.
The directory must remain immutable and under trusted filesystem ownership for
the entire load: reopening these paths is not a hostile concurrent filesystem
snapshot guarantee. Inventory snapshots plus the merged encoded state share one
logical snapshot-byte budget. The merge uses the official safetensors codec with
the remaining output size checked before allocation. Strict interop import
requires complete matching state keys, shapes and dtypes; partial or renamed
state is rejected. The factory must create a fresh module, supply the exact
selected device, exclude external state readers/writers and decode the actual
owner descriptors. No class, operator plugin or executable is discovered from
bundle data. Factory callbacks must own/copy data retained after the factory
returns, including data taken from the borrowed VerifiedBundle argument.

The owner `Pipeline` binds input, output, preprocessing and postprocessing to
the exact verified descriptor schema ID, version and digest. Its callbacks are
trusted synchronous application code. Empty callbacks mean an explicitly bound
identity stage. The owner is responsible for interpreting descriptor semantics;
hash equality alone cannot implement an arbitrary schema or preprocessing
language. Callbacks must be bounded, serialize with the instance, avoid retaining
mutable request references and never reenter the same instance.

At least one caller-supplied known vector is mandatory. Actual pre/invoke/post
output must pass the output schema, resource bounds and recursive numerical
comparison before a pointer is returned as ready. Load/warmup cancellation and
deadlines are cooperative around stages. Failed loading never publishes a ready
instance. Native state is switched to eval mode and forward runs in inference
mode. Ambient autocast on the chosen runtime device is rejected.

`load_torchscript_predictor` accepts archive bytes, their independently trusted
digest, exact input/output schemas, explicit trust options and a separate pinned
`RuntimeDescriptor`. It inherits the real compatibility runtime's supported
boundary, explicit device, defaults, serialization and post-forward failure
behavior. It does not put executable JIT archives inside data-only bundles.
Its environment lock is separate from the archive hash. Batching and OOM retry
are refused for this compatibility adapter.

## Requests, budgets and results

`PredictRequest` contains the existing `nn::Invocation`, a model-version-bound
request context and an absolute steady-clock deadline. It introduces no second tensor container. Supported values are dense CPU/CUDA tensors, exact scalars, None, unnamed tuples, homogeneous lists and string-keyed dictionaries. The request/output budgets independently bound logical tensor bytes, tensor count, rank, dimension, shape product, traversal nodes/depth and total strings. Finite and optional numerical range checks occur before tensor cloning. Input schema, dtype and device checks are exact; no automatic cast, transfer or fallback occurs.

Admission has a hard `max_active_requests` limit including serialization waiters. Inputs must remain quiescent during snapshotting. Outputs and queued inputs own independent contiguous tensor snapshots; storage aliases between values are not preserved. A result can outlive close. CUDA success includes device synchronization, which can also wait for unrelated work on the device.

These bounds constrain declared boundary/state/staging bytes, not total process RSS, backend workspaces, native callback allocations or a kernel's duration. Codec expansion is owned by the trusted codec. Hard CPU/GPU memory or execution time limits require an external host/process supervisor. No such isolation is implied by this in-process Preview.

## Dynamic batching

`DynamicBatcher` has one worker and deliberately supports one positional tensor
with leading dimension one, zero kwargs and one tensor result with a matching
leading batch dimension. Input and output schemas must declare fixed ranks and
shapes with a dynamic leading dimension. The backend must explicitly assert
independent row semantics and batch-preserving preprocessing/postprocessing.
Other structured boundaries remain available through direct `Predictor` calls.

A queue key contains the fixed predictor/model/pipeline/schema identity,
dtype, exact device, rank/trailing dimensions, priority class, deadline class,
project, principal and deployment. Only a compatible queue prefix is coalesced,
preserving FIFO scheduling between incompatible groups. There is no result cache.
Capacity is reserved before snapshotting. Queue count includes submitting
reservations; the tensor-byte budget includes queued and active work until
completion. Merged inputs also respect the predictor input-byte budget.

`max_batch_size` and coalescing `max_wait` are bounded. The wait limit applies
after the oldest compatible request is queued; it cannot bound time behind an active compute call. Expired/cancelled members are removed before a new batch. A merged compute does not inherit one member's stop token or deadline. Each member is checked again before independent result delivery, so cancelling one member never aborts peers. Kernels remain cooperative and cannot be preempted.

Only a confirmed allocation failure from an explicitly retry-safe native forward raises `RetryableOutOfMemory`. Ordinary resource-budget violations, pre/post stage errors, output errors and non-allocation backend failures do not trigger retries. Splitting is iterative with one finite retry budget for the whole original batch and at most eight retries. A singleton is never retried. JIT models refuse this policy because trusted code can mutate private state.

## Leases, close and health

A predictor serializes execution and retains a shared active lease through input ownership, execution, synchronization and result snapshotting.
`close` first refuses new work, then waits for active leases before unloading.
A timed/cancelled close returns false and leaves ownership retained in draining
state; it never unloads a running session. Destruction waits safely, and can block
indefinitely on stuck trusted native code. Exclude concurrent destruction.

A batcher closes admission, resolves queued requests as cancelled and waits for
active compute and snapshot reservations. The caller then closes the predictor.
The service adapter performs this order. The scheduler never detaches its worker.
A post-forward failure degrades the instance conservatively, except confirmed
OOM in a declared retry-safe native backend. Subsequent admission fails.
Pre-forward schema/resource/cancel/deadline rejections keep the model usable.
Readiness is false while draining, degraded or closed; liveness means the local
adapter process is responsive. Streaming is explicitly false.

## Transport and observation

`InferenceServiceAdapter` accepts a versioned trusted protocol decoder/encoder
and authorizer. A mature host owns TLS/authentication and derives the principal, project, model version, deployment, request/trace IDs and deadline. Missing authentication or missing principal/project is rejected; authorization runs before decoding. Request bytes are bounded before decode and response bytes after encode. Protocol callbacks must enforce allocation/expansion bounds. A supplied batcher must own the same predictor object.

The adapter maps typed framework failures to stable public statuses: invalid_argument, unauthenticated, permission_denied, resource_exhausted, cancelled, deadline_exceeded, unavailable and internal. Public replies contain fixed status text, never private backend exceptions, input values or bearer tokens. Direct C++ exceptions can retain nested backend causes for authorized local diagnostics; do not dump them into public responses or production logs.

`ObservationHook` receives redacted foundation log/correlation/attribute records
with request, trace, project, deployment, model, backend and device identities.
The application can publish these using its foundation telemetry envelope owner.
The callback must be bounded/nonblocking and nonreentrant. Exceptions increment
observation failures without replacing inference results. Per-request IDs belong
in logs/traces rather than unbounded metric labels. Responses provide queue,
preprocess, compute, postprocess and total timings; service replies add actual
decode/encode measurements and service total time. Scheduler counters provide
accepted/completed/failed, residency, active depth and OOM retry counts. This
Preview does not claim unmeasured p99, RSS or hardware throughput baselines.

The local JSONL example uses the official nlohmann JSON parser and a fixed
generated two-feature arithmetic model. It provides bounded stdin/stdout unary
requests, health and manifest modes, explicit CUDA selection and no network
listener. See [example usage](../../examples/serving/README.md). It is a local
application protocol, not an HTTP/OpenAPI/Protobuf replacement.

## Validation and limits

The chapter supplies tests for real native bundle load/import/warmup/forward,
hash/trust/runtime/schema/resource failures, numerical output, independent
snapshots, multiple inputs/tuple/dict, active leases, timed drain, queue limits,
deadlines, cancellation isolation, finite OOM splitting, sanitized service status,
authorization and response limits. CPU and available CUDA exercise actual
native and TorchScript numerical paths. The installed consumer exercises the
exported target with real bundle import, forward, batch and close.

These are supplied tests, not execution evidence. Before chapter acceptance,
run them in the unified Windows CPU and remote Ubuntu/CUDA build directories,
exercise the JSONL host and installed consumer, and record actual platform
results. Hardware latency/throughput/memory/soak baselines and external host
security are not implied by passing unit tests. ONNX Runtime, streaming, reload,
instance pools, networking, online rollbacks, deployment policy and multi-tenant
control planes remain outside this Preview capability manifest.
