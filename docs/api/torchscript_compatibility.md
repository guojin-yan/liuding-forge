# TorchScript compatibility runtime

`jyppx/interop/torchscript_compatibility.hpp` provides real official
`torch::jit::load` and `forward` execution for **explicitly trusted existing
TorchScript archives**. It is deprecated compatibility support and requires the
optional `interop -> nn` closure. It does not implement trace/export, discover
plugins, resolve a model factory, or add executable formats to the data-only
model bundle.

This chapter's source and tests require integration and unified platform
validation before any supported release claim. Presence of a backend descriptor
or this source file is not execution evidence.

```cpp
using namespace jyppx;
using namespace jyppx::interop;

nn::InvocationSchema input{
    .positional = {ValueSchema::tensor(), ValueSchema::tensor()},
    .keyword = {{"scale", ValueSchema{.kind = ValueKind::floating,
                                     .required = false}}}};
TorchScriptLoadOptions options;
options.trust_source = true;  // Explicit executable producer/source approval.
options.device = Device(c10::kCPU);
auto runtime = TorchScriptCompatibilityRuntime::load(
    archive_bytes, trusted_sha256, input,
    ValueSchema::tuple({ValueSchema::tensor(), ValueSchema::tensor()}), options);
auto result = runtime.invoke({.positional = {x, y}, .keyword = {}});
runtime.close();
```

The complete bytes are privately snapshotted and the **same snapshot** is hashed
and passed to LibTorch's stream loader. A matching digest establishes byte
identity, not trust in the producer. `trust_source` defaults to false and is
required even for inspection, because loading an executable archive is not a
data-only operation. Callers must exclude mutation while the source snapshot is
made. No path is reopened after verification. Host-registered operators are
resolved by official LibTorch; this adapter never loads a plugin from a bundle.

| Boundary | Exact supported contract |
|---|---|
| Device | `cpu` with no index or an actually available explicit `cuda:N`; no implicit CUDA device, MPS, meta or fallback |
| Tensor inputs | Defined contiguous dense strided tensors on that exact device; bool, uint8, int8, int16, int32, int64, float16, bfloat16, float32, float64; no quantized/sparse/complex/conj/negative-view tensors |
| Tensor outputs/state | Same dtype/device/layout families; outputs may have dense strided views and are copied to contiguous independent storage |
| Scalars | Exact JIT `int`, `float`, `bool`, `str`, `None`; int-to-float conversion is not inferred |
| Tuple | Unnamed fixed tuple with an exact schema for each element |
| List | Official typed homogeneous `List[T]` with one matching element schema; empty lists must still have the correct type |
| Dictionary | Official typed `Dict[str,T]`; every permitted key is in a closed schema and every field matches homogeneous `T`; no undeclared keys |
| Optional | `Optional[T]` uses `nullable=true` and the matching non-null category; optionality is not inferred from a missing argument |
| Excluded | `Any`, object/custom classes, named tuples, device IValues, futures, variadic methods, keyword-only JIT arguments and open/extra-key schemas |

These are boundary encodings. They do not claim every graph/operator supports
every dtype/device. Unsupported kernels raise a backend error, with no implicit
cast, transfer, backend switch, precision change or floating fallback. Ambient
autocast must be disabled on the runtime device; an active context is rejected
without changing caller state. The
official JIT artifact itself must be compatible with the linked LibTorch lane;
the adapter does not assert cross-version/ABI compatibility without fixture
evidence.

`InvocationSchema::positional` is an exact required prefix of the actual forward
arguments. Every remaining argument must appear in `keyword` under its **real
JIT name**; extra/missing schema entries fail load. An argument may be omitted
only when its keyword schema has `required=false` and the artifact has a real
default. That default is validated and snapshotted during load, then copied for
each invocation. A defaulted argument can instead be declared required. A
positional default is still explicitly supplied because the prefix is fixed.
All runtime argument names, container types, scalar kinds, tensor constraints
and results are checked. Default values are not copied into inspect metadata.

Each input tensor and mutable container is copied before execution. Shared
storage/identity relationships between fields are intentionally not preserved;
graphs depending on caller-visible aliasing are outside this value contract.
Inputs must remain quiescent during copying. Outputs are independent inference
snapshots, without autograd, and can outlive close. In-place output mutation must
follow LibTorch inference-tensor rules. Scripted code can mutate its private
model state, but returned storage cannot expose that private state.

Calls on one runtime serialize. Close waits for the active call and releases the
module/defaults; repeated close is valid. Move and destruction require exclusive
ownership. Calls rejected before forward leave the runtime ready. Any exception,
result mismatch, resource rejection, cancellation or deadline after entering
forward closes it; partial scripted private-state changes are not rolled back.
Calls after close fail. Backend diagnostics are retained as nested exceptions
and are not echoed into top-level errors. The producer must ensure that graphs
and host operators do not start untracked background mutations or access shared
external mutable state; call serialization is not a sandbox for such code.

Stop tokens and absolute steady-clock deadlines are cooperative checks while
waiting for the runtime and before/after backend work. They cannot preempt
archive parsing, forward, copies, synchronization, kernels or a stuck operator.
CUDA success includes device synchronization, which may also wait for unrelated
device work. On an execution error draining is best effort; a failed CUDA context
requires recovery by the process owner. A hard deadline or hostile model needs
an externally isolated runner.

Limits bound compressed archive bytes, known parameter/buffer logical bytes,
input/output snapshot logical tensor bytes, rank, arguments, strings and
traversal/type metadata. Shared storage is conservatively charged for every
occurrence. Known state is counted **after loading**, and output is bounded
**after forward**, before snapshot publication. Embedded constants, temporary
workspaces, archive decompression and arbitrary operator allocations are not
prebounded process memory. Unknown dynamic attributes are not counted as named
parameters/buffers. Inspect reports that narrow accounting explicitly.

`tests/unit/interop/test_torchscript_compatibility.cpp` uses official C++ JIT
definition/save/load fixtures, without Python, to exercise numerical execution,
default/keyword binding, all ten dtype identity paths, typed containers,
Optional/None, corruption/trust, state accounting, resources, failure closure,
storage ownership, cancellation and concurrent private state. Define
`JYPPX_TEST_TORCHSCRIPT_CUDA=1` only for a real CUDA lane; those tests require
availability and never silently skip.

The optional `scripts/interop/verify_torchscript_compatibility.py` requires an
independently locked Python/PyTorch environment and a separately compiled test
target with `JYPPX_TORCHSCRIPT_PYTHON_INTEROP=1`. It installs nothing, generates
an official scripted artifact, verifies Python eager/scripted results, runs the
real C++ fixture test and records artifact/executable digests and exact versions.
Pass exact `--expected-python-version`, `--expected-torch-version`, the actual
`--jyppx-commit`, `--cpp-test-executable`, and `--work-dir`; `--cuda` additionally
requires both real CUDA and `JYPPX_TEST_TORCHSCRIPT_CUDA=1`. Missing or skipped
tests fail. No Python interoperability evidence is claimed until this opt-in
workflow actually passes. Do not install Torch into the telemetry sidecar env.
