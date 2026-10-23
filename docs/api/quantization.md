# Quantization: storage-only int8 reference path

The `jyppx::liuding::quantization` component exposes the versioned
`jyppx.qint8.symmetric.per_tensor/v1` format. It stores one Linear weight matrix
as symmetric per-tensor int8 values, preserves finite FP32 bias, and records
calibration lineage. The runtime dequantizes the matrix once, then computes
ordinary FP32 Linear output. Reports therefore set `compute_quantized=false`
and `qat=false`, with `execution_claim="storage-quantized"`.

The format changes persistent representation. It does not provide a quantized
compute kernel, backend packing, fake-quant gradients, QAT, hardware acceleration,
or a performance improvement guarantee. `fallback_modules=1` identifies the
Linear executed through FP32 fallback; unsupported-module count is zero because
this API accepts exactly one Linear and does not traverse an arbitrary graph.

## Reference formula

For finite values, `scale=max(abs(x))/127`; an all-zero finite range uses
`scale=1`. Zero point is always zero and the canonical range is `[-127,127]`.
Quantization rounds `x/scale` to nearest with ties to even and clamps to the
canonical range. Dequantization computes `q*scale` and returns FP32 values.
The scalar rounding implementation does not consult the floating-point rounding
mode. Empty and all-nonfinite inputs cannot derive qparams and raise `NotReady`.
The quantizer rejects nonfinite values with `NumericalError`.

`MinMaxObserver` counts each observe call and every input value while skipping
NaN and infinity when updating extrema. Empty/all-nonfinite observations retain
an empty finite range. `merge` combines counters and extrema without changing
sample order. The observer is an exact `CheckpointParticipant` with a bounded,
versioned 40-byte state blob; restore validates the manifest, policy, field
lengths, booleans, counters and extrema before committing state.

## Calibration and conversion

```cpp
#include <jyppx/liuding/quantization/quantization.hpp>
#include <array>

namespace q = jyppx::liuding::quantization;
q::CalibrationSession calibration({
    .dataset_version = "dataset/v1",
    .dataset_digest = {.hex = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    .preprocessing_fingerprint = "normalize/v1",
    .seed = 731U});
const std::array<float, 2> sample{1.0F, -1.0F};
calibration.observe("sample-0", sample);
const std::array<float, 2> weights{1.0F, -1.0F};
const std::array<float, 1> bias{0.25F};
const auto artifact = q::quantize_linear(
    {.in_features = 2U, .out_features = 1U,
     .weights = weights, .bias = bias}, calibration.finalize());
const auto bytes = q::serialize_linear_artifact(artifact);
q::StorageQuantizedLinearRuntime runtime(q::deserialize_linear_artifact(bytes));
const auto output = runtime.forward(sample);
const auto report = q::quantization_report_json(runtime.report(weights));
```

Calibration retains the dataset version, canonical SHA-256 digest,
preprocessing fingerprint, seed and ordered unique sample IDs. IDs and metadata
must be valid UTF-8 and satisfy explicit resource limits. `finalize` requires at
least one sample and a finite observer range. Calibration metadata is stored
with each artifact; changing the dataset, preprocessing or seed changes the
serialized evidence. The observer participant saves numeric state only; callers
resuming a whole calibration session must preserve the matching sample IDs and
lineage as part of their recipe state.

## Artifact and execution contracts

The binary v1 artifact begins with `JYPPXQ1\0` and uses little-endian integer
and IEEE-754 scalar encoding. It includes the artifact version, immutable format
descriptor, dimensions, weight qparams, row-major int8 weights, optional FP32
bias, calibration lineage, sample IDs, observer state and activation qparams.
Reserved fields are zero and signed floating-point zero is canonicalized to
positive zero. Decoding validates resource budgets before allocation and
re-encodes the result to reject noncanonical or trailing bytes. Unknown artifact
versions raise `CompatibilityError`; truncated or structurally malformed data
raises `SerializationError`. Invalid semantic values raise the corresponding
JYPPX argument or numerical error.

`QuantizationLimits` defaults to 16 MiB per artifact, 65,536 features per
dimension, 1,000,000 calibration sample IDs, 256 bytes per ID and 1,024 bytes per
lineage text field. These are independent ceilings: a large shape still must fit
the exact artifact-byte budget. Serialization enforces the complete byte count,
including metadata. Raising limits is a caller decision and never changes the
capability claim.

`StorageQuantizedLinearRuntime::forward` accepts contiguous FP32 rows with a
declared batch size and rejects shape mismatches, nonfinite input or numerical
overflow. The runtime returns row-major `[batch,out_features]` FP32 output.
Calibration qparams provide evidence only; the runtime does not quantize
activations.

## Quality report and acceptance evidence

`compare_reference` reports RMSE, MAE, maximum absolute/relative error, SQNR and
cosine similarity over finite pairs, and counts excluded nonfinite pairs.
Relative error divides by `max(abs(reference),1e-12)`. An exact nonzero match has
infinite SQNR, represented by JSON `null`; an exact zero match has SQNR zero.
The report preserves the execution claims, artifact byte count, fallback count,
unsupported count and calibration lineage. It makes no task-quality claim;
applications must select acceptable error/accuracy thresholds.

The chapter's unit suite covers qparam and quant/dequant golden vectors,
round-to-nearest-even, nonfinite/zero handling, error metrics, observer merge and
atomic restore, calibration lineage and duplicate/budget rejection, canonical
artifact roundtrip, exact and one-byte-short budgets, truncated prefixes,
malformed headers, explicit capability claims and FP32 Linear reference output.
The install consumer links only the exported `jyppx::liuding::quantization` target and
exercises conversion, serialization, inference and machine-readable reporting.
Both suites must pass in the chapter's Windows CPU and Ubuntu+CUDA release
validation before this component can be considered delivered. This document
describes the storage path and does not imply successful validation by itself.
