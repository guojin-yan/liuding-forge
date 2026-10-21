# State formats and explicit backend boundaries

Target: `jyppx::interop`. The foundation artifact types identify exact bytes;
format adapters define their own bounded versioned contracts.

| Surface | Current operation |
| --- | --- |
| `safetensors.hpp` | Strict bounded raw tensor reader/writer for 13 byte-addressable types, scalar/empty tensors and string metadata. |
| `weight_import_report.hpp` | Pure metadata strict/explicit-partial mapping plan and complete versioned diagnostics. |
| `model_weight_import.hpp` | Optional nn adapter verifying and applying staged model weights on CPU/CUDA. |
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

TorchScript is a compatibility and deprecated backend for existing artifacts;
an exporter does not become available just because LibTorch is linked. Tracing
requires model-specific support and explicit consent. The current backend
contract describes preflight and does not implement a generic TorchScript
loader/runner/tracer.

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
