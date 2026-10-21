# State formats and explicit backend boundaries

Target: `jyppx::interop`. The foundation artifact types identify exact bytes;
format adapters define their own bounded versioned contracts.

| Surface | Current operation |
| --- | --- |
| `safetensors.hpp` | Strict bounded raw tensor reader/writer for 13 byte-addressable types, scalar/empty tensors and string metadata. |
| `weight_import_report.hpp` | Pure metadata strict/explicit-partial mapping plan and complete versioned diagnostics. |
| `model_weight_import.hpp` | Optional nn adapter verifying and applying staged model weights on CPU/CUDA. |
| `torchscript_compatibility.hpp` | Optional nn adapter loading explicitly trusted existing JIT bytes and executing a validated inference boundary. |
| `model_bundle.hpp` | Immutable, bounded model bundle publication and complete local inventory verification. |
| `checkpoint_manifest.hpp`, `file_state_archive.hpp` | Versioned participant manifest and integrity-checked atomic file storage. |
| `export_backend.hpp` | Explicit registry, capability descriptors, real-probe preflight and strict sidecar metadata codecs. |

The safetensors format contains no checksum or device identity. Use an
independently trusted ArtifactRef/digest to check the complete file. The codec
preserves opaque little-endian bits; it does not construct a Module, infer a
forward graph, interpret labels or execute code. Unsupported FP8, sub-byte,
complex and future dtype tokens fail explicitly. It rejects duplicate fields,
unknown tensor keys, invalid UTF-8, overflowing shapes, sparse/overlapping
offsets, gaps, trailing bytes and configured resource bounds.

`BackendRegistry` registration is explicit. A descriptor is declared capability,
not evidence that a package or executable exists. A missing real environment
probe remains `conditional`; a request outside its source/format/operation,
dtype/device/options matrix is `unsupported`. Request-bound model preflight,
verified evidence bytes and process/environment binding are runner duties.
Deserializing a probe cannot make an untrusted statement into execution evidence.

`TorchScriptCompatibilityRuntime` is a real compatibility runtime using official
`torch::jit::load` and `forward`. It snapshots and hashes the exact archive bytes
before loading and requires `trust_source=true`; a digest establishes byte
identity and does not authenticate an executable producer. Inspection also
requires that trusted load. It accepts a fixed positional prefix and exact JIT
keyword names; omitted arguments must have actual validated defaults in the
artifact. The optional `interop -> nn` adapter reuses `nn::Invocation` and
`ValueSchema`, and is absent from an nn-disabled install.

Its boundary is restricted to dense supported-dtype CPU/explicit `cuda:N`
tensors, exact scalars, fixed unnamed tuples, typed homogeneous lists, closed
string-keyed dictionaries and Optional/None. It rejects Any/classes, extra
keywords, unsupported devices/dtypes, active ambient autocast and implicit
fallback. Inputs/defaults/outputs are independent snapshots and do not preserve
cross-field storage aliasing. Calls serialize, and any failure after entering
forward closes the instance without claiming state rollback. Byte/traversal
limits and stop/deadline checks are cooperative boundaries; they cannot bound
host/operator allocations or preempt JIT/CUDA kernels. See
[the runtime contract](torchscript_compatibility.md) for the exact matrix and
failure/lifecycle/resource semantics.

TorchScript remains deprecated compatibility for existing artifacts. The runtime
does not trace or export models, change the data-only bundle format, imply a
generic C++ graph exporter, or make a descriptor/probe into execution evidence.
The separate metadata descriptor advertises declared operations only; the linked
LibTorch lane, actual loaded artifact, cases, device and dtype still determine
runtime results.

ONNX export from a Python factory or ExportedProgram belongs to an optional,
separately locked official `torch.export`/ONNX/onnxscript/ORT environment. Native
C++ state has no common forward graph and cannot be exported by inference from
a Module type or filename. A model-specific graph adapter is required. The
current C++ code checks requests/probes/candidate metadata; it is not an ONNX
exporter or independent ONNX numerical verifier. ONNX remains a v1.1+ backend
under the development plan. The environment probe in `scripts/interop/` also
remains outside the installed product runtime and does not qualify a model.

The candidate metadata verifier requires exact request/artifact/environment
binding, official schema/checker/shape inference and real runtime/numerical
case evidence. Those booleans still require independent verification of their
referenced evidence before publication; JSON success alone never proves export.

Official safetensors interchange and the new C++ chapter are awaiting the
consolidated Windows and Ubuntu/CUDA qualification. Optional Python test locks
live in `scripts/interop/`; they do not add a core runtime dependency.

The install-only `tests/install/interop_consumer` uses installed public headers
and `jyppx::interop` to read/write safetensors, plan strict mappings, shard whole
tensors, serialize every required/optional descriptor association, publish a
complete data-only bundle and reverify its exhaustive file/tensor inventory.
It checks domain-separated signing bytes and rejects an unsigned bundle when
trusted signatures are required; it does not invent a cryptographic algorithm
or treat fixture SBOM/evaluation data as validated release evidence. It also
round-trips request/probe/candidate metadata and proves unexecuted ONNX metadata
cannot pass the candidate gate. No Python/exporter/ORT is run in that path.

When the installed build enables nn, the same consumer really applies verified
safetensors to module state, saves an official C++ JIT fixture, checks default
trust rejection and runs trusted CPU forward. With nn disabled its preprocessor
branch excludes both optional headers and fails compilation if either leaked
into the package. These consumers must run against each qualified install;
their source is not a substitute for a recorded successful run. The independent
optional Python JIT fixture workflow requires its own locked PyTorch environment
and records success only after actual Python/C++ execution, with no dependency
installation into a telemetry environment.
