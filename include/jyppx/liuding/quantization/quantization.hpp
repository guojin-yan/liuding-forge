// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_QUANTIZATION_QUANTIZATION_HPP
#define JYPPX_LIUDING_QUANTIZATION_QUANTIZATION_HPP
/**
 * @file quantization.hpp
 * @brief Versioned storage quantization, calibration, and FP32 Linear runtime.
 *
 * This preview component owns a small, auditable per-tensor symmetric int8
 * format. Its artifact is storage-quantized: the runtime dequantizes weights
 * and executes an ordinary FP32 matrix multiply. No compute-quantized kernel,
 * QAT, fake-quant gradient, or hardware speedup is claimed.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace jyppx::liuding::quantization {

/** @brief Quantization scheme serialized by a format descriptor. */
enum class QuantizationQScheme : std::uint8_t {
  symmetric = 1,  ///< One scale, zero point fixed at zero.
};

/** @brief Quantization granularity serialized by a format descriptor. */
enum class QuantizationGranularity : std::uint8_t {
  per_tensor = 1,  ///< One qparam pair for the complete tensor.
};

/** @brief Reports the only implementation plane supported by this preview. */
enum class QuantizationExecutionPlane : std::uint8_t {
  storage_only =
      1,  ///< Persistent representation changes; execution remains FP32.
};

/** @brief Stable format identity and explicit capability claims. */
struct QuantizationFormatDescriptor {
  std::string format_id =
      "jyppx.qint8.symmetric.per_tensor/v1";  ///< Stable portable format
                                              ///< identity.
  std::uint32_t format_version = 1U;  ///< Supported descriptor schema version.
  std::string dtype = "int8";         ///< Signed eight-bit persistent elements.
  QuantizationQScheme qscheme =
      QuantizationQScheme::symmetric;  ///< Fixed zero-centered scheme.
  QuantizationGranularity granularity =
      QuantizationGranularity::per_tensor;  ///< One scale per tensor.
  std::uint32_t block_size = 0U;       ///< Zero means the tensor is one block.
  std::uint32_t alignment_bytes = 1U;  ///< Required storage alignment in bytes.
  bool little_endian = true;  ///< Fixed byte order for multibyte scalar fields.
  QuantizationExecutionPlane execution_plane =
      QuantizationExecutionPlane::storage_only;  ///< Persistent storage
                                                 ///< implementation plane.
  bool compute_quantized =
      false;  ///< Always false for this format implementation.
  bool qat_supported =
      false;  ///< Always false; training-aware conversion is unavailable.
};

/**
 * @brief Validates the immutable v1 descriptor and its conservative claims.
 * @param descriptor Complete descriptor to validate without modification.
 * @throws InvalidArgument If any fixed v1 field or capability claim differs.
 */
void validate_format_descriptor(const QuantizationFormatDescriptor& descriptor);

/** @brief Bounded parser/encoder limits for quantization metadata and
 * artifacts. */
struct QuantizationLimits {
  std::size_t max_artifact_bytes =
      16U * 1024U * 1024U;  ///< Complete encoded artifact ceiling.
  std::size_t max_features =
      65536U;  ///< Maximum extent of either Linear dimension.
  std::size_t max_samples =
      1000000U;  ///< Maximum recorded calibration sample IDs.
  std::size_t max_sample_id_bytes =
      256U;  ///< UTF-8 byte ceiling for each sample identity.
  std::size_t max_lineage_text_bytes =
      1024U;  ///< UTF-8 byte ceiling for each lineage field.
};

/**
 * @brief Validates positive limits before any allocation or decoding.
 * @param limits Independent resource ceilings to validate.
 * @throws InvalidArgument If any ceiling is zero.
 */
void validate_limits(const QuantizationLimits& limits);

/** @brief Scalar qparams for the symmetric int8 reference formula. */
struct QuantizationParams {
  double scale = 1.0;  ///< Finite positive multiplier during reconstruction.
  std::int32_t zero_point = 0;       ///< Fixed zero for the symmetric format.
  std::int8_t quantized_min = -127;  ///< Inclusive persistent lower bound.
  std::int8_t quantized_max = 127;   ///< Inclusive persistent upper bound.
};

/**
 * @brief Computes symmetric qparams over finite values.
 * @param values Borrowed FP32 values, required to remain valid for the call.
 * @return Zero-centered parameters; an all-zero range uses unit scale.
 * @throws NumericalError If a nonfinite value occurs.
 */
QuantizationParams derive_symmetric_params(std::span<const float> values);

/**
 * @brief Quantizes finite FP32 values with round-to-nearest-even semantics.
 * @param values Borrowed source values, not retained after return.
 * @param params Valid symmetric v1 reconstruction parameters.
 * @return Independently owned int8 values clamped to the declared range.
 */
std::vector<std::int8_t> quantize_symmetric(std::span<const float> values,
                                            QuantizationParams params);

/**
 * @brief Dequantizes one int8 range to FP32 using the reference formula.
 * @param values Borrowed persistent values, not retained after return.
 * @param params Valid symmetric v1 reconstruction parameters.
 * @return Independently owned FP32 reconstructed values.
 */
std::vector<float> dequantize_symmetric(std::span<const std::int8_t> values,
                                        QuantizationParams params);

/** @brief Standard error evidence for one float/reference pair. */
struct QuantizationErrorReport {
  std::uint64_t count =
      0U;  ///< Finite pairs included in the error calculation.
  std::uint64_t nonfinite_inputs =
      0U;             ///< Pairs excluded because either input is nonfinite.
  double rmse = 0.0;  ///< Root mean square reconstruction error.
  double mae = 0.0;   ///< Mean absolute reconstruction error.
  double max_absolute_error =
      0.0;  ///< Largest absolute difference over finite pairs.
  double max_relative_error =
      0.0;  ///< Largest error divided by max(abs(reference), 1e-12).
  double sqnr_db = 0.0;  ///< Signal-to-quantization-noise ratio in decibels.
  double cosine_similarity =
      1.0;  ///< Normalized vector similarity with explicit zero handling.
};

/**
 * @brief Computes RMSE, MAE, maxima, SQNR and cosine similarity.
 * @param original Borrowed reference values.
 * @param reconstructed Borrowed candidate values of the same length.
 * @return Error evidence over finite pairs, including excluded-pair counts.
 */
QuantizationErrorReport compare_reference(std::span<const float> original,
                                          std::span<const float> reconstructed);

/** @brief Bounded observer state suitable for checkpointing and merging. */
struct MinMaxObserverState {
  std::uint64_t sample_count =
      0U;  ///< Number of observed samples, including empty samples.
  std::uint64_t value_count =
      0U;  ///< Number of observed scalar values, including nonfinite values.
  bool has_finite_value =
      false;  ///< Whether extrema represent an observed finite value.
  float minimum =
      0.0F;  ///< Finite minimum, or zero while no finite value exists.
  float maximum =
      0.0F;  ///< Finite maximum, or zero while no finite value exists.
};

/**
 * @brief Deterministic finite min/max observer with explicit empty policy.
 * Mutations and checkpoint operations require caller exclusion. State owns no
 * tensor storage; invalid restore input leaves the previous numeric state
 * intact.
 */
class MinMaxObserver final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates an empty observer with a stable participant identity.
   * @param participant Stable checkpoint identity validated on construction.
   */
  explicit MinMaxObserver(
      std::string participant = "quantization.observer.minmax");
  /** @brief Releases the owned participant identity and numeric state. */
  ~MinMaxObserver() override = default;
  /**
   * @brief Records one calibration sample; NaN/Inf are counted and skipped.
   * @param values Borrowed sample values, not retained after return.
   */
  void observe(std::span<const float> values);
  /**
   * @brief Merges another observer using min/max and exact counters.
   * @param other Source accumulator, which remains unchanged.
   */
  void merge(const MinMaxObserver& other);
  /** @brief Returns accumulated state. @return Independent numeric snapshot. */
  MinMaxObserverState state() const noexcept;
  /**
   * @brief Returns qparams, rejecting an empty/all-nonfinite observer.
   * @return Symmetric parameters derived from the finite extrema.
   */
  QuantizationParams finalize() const;
  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;
  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;
  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;
  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  std::string participant_;    ///< Stable checkpoint-domain identity.
  MinMaxObserverState state_;  ///< Owned exact numeric accumulator state.
};

/** @brief Dataset lineage bound to one calibration run. */
struct CalibrationLineage {
  std::string dataset_version;  ///< Application-owned dataset version identity.
  ArtifactDigest
      dataset_digest;  ///< SHA-256 binding of the calibration data manifest.
  std::string preprocessing_fingerprint;  ///< Stable preprocessing identity
                                          ///< supplied by the recipe.
  std::uint64_t seed = 0U;  ///< Recorded deterministic sample-selection seed.
};

/** @brief Final calibration evidence and weight qparams for one recipe. */
struct CalibrationResult {
  CalibrationLineage
      lineage;  ///< Source and preprocessing identity for this run.
  std::vector<std::string>
      sample_ids;  ///< Ordered unique observed sample identities.
  MinMaxObserverState
      activation_state;  ///< Exact numeric observation evidence.
  QuantizationParams activation_params;  ///< Validated activation range;
                                         ///< execution remains FP32.
};

/**
 * @brief Records bounded samples and preserves dataset lineage.
 * The session owns metadata and numeric state and requires caller exclusion.
 * Its observer checkpoint contains numeric state only; recipes must separately
 * preserve matching sample identities and lineage when resuming a session.
 */
class CalibrationSession final {
 public:
  /**
   * @brief Starts a deterministic calibration session.
   * @param lineage Owned source and preprocessing identity.
   * @param limits Frozen independent metadata and artifact ceilings.
   */
  CalibrationSession(CalibrationLineage lineage,
                     QuantizationLimits limits = {});
  /**
   * @brief Adds one unique sample ID and observes finite values.
   * @param sample_id Unique bounded UTF-8 sample identity, copied on success.
   * @param values Borrowed sample values, not retained after return.
   */
  void observe(std::string_view sample_id, std::span<const float> values);
  /**
   * @brief Returns complete immutable evidence; rejects an empty session.
   * @return Independent complete calibration evidence.
   */
  CalibrationResult finalize() const;
  /**
   * @brief Returns the observer participant for coordinated checkpoints.
   * @return Borrowed participant, valid until session destruction.
   */
  MinMaxObserver& observer() noexcept;
  /**
   * @brief Returns the immutable calibration lineage.
   * @return Borrowed source metadata, valid until session destruction.
   */
  const CalibrationLineage& lineage() const noexcept;

 private:
  QuantizationLimits limits_;  ///< Frozen independent input resource ceilings.
  CalibrationLineage
      lineage_;              ///< Owned immutable calibration source identity.
  MinMaxObserver observer_;  ///< Owned finite-value numeric accumulator.
  std::vector<std::string> sample_ids_;  ///< Owned observation order.
  std::unordered_set<std::string>
      sample_id_index_;  ///< Duplicate-identity lookup index.
};

/** @brief Float Linear source owned by the caller until conversion returns. */
struct FloatLinearModel {
  std::size_t in_features = 0U;    ///< Positive width of each input row.
  std::size_t out_features = 0U;   ///< Positive width of each output row.
  std::span<const float> weights;  ///< Row-major [out_features, in_features].
  std::span<const float> bias;     ///< Optional [out_features].
};

/** @brief Portable int8 storage artifact for one Linear weight matrix. */
struct QuantizedLinearArtifact {
  std::uint32_t artifact_version =
      1U;  ///< Supported complete artifact schema version.
  QuantizationFormatDescriptor
      format;  ///< Portable storage descriptor and conservative claims.
  std::size_t in_features = 0U;   ///< Positive width of each input row.
  std::size_t out_features = 0U;  ///< Positive width of each output row.
  QuantizationParams
      weight_params;  ///< Weight reconstruction scale and fixed integer bounds.
  std::vector<std::int8_t>
      quantized_weights;    ///< Owned row-major persistent weight elements.
  std::vector<float> bias;  ///< Optional owned finite FP32 output bias.
  CalibrationResult
      calibration;  ///< Owned source lineage and activation evidence.
};

/**
 * @brief Validates dimensions, qparams, lineage and all bounded vectors.
 * @param artifact Complete artifact to validate without modification.
 * @param limits Independent resource ceilings.
 */
void validate_linear_artifact(const QuantizedLinearArtifact& artifact,
                              QuantizationLimits limits = {});

/**
 * @brief Converts one Linear source after a completed calibration session.
 * @param model Borrowed dimensions, FP32 weights and optional bias.
 * @param calibration Complete matching calibration evidence.
 * @param limits Independent resource ceilings.
 * @return Independently owned portable storage-only artifact.
 */
QuantizedLinearArtifact quantize_linear(const FloatLinearModel& model,
                                        const CalibrationResult& calibration,
                                        QuantizationLimits limits = {});

/**
 * @brief Encodes one canonical versioned Linear artifact.
 * @param artifact Complete validated source artifact.
 * @param limits Independent input and complete output ceilings.
 * @return Independently owned canonical little-endian byte sequence.
 */
std::vector<std::byte> serialize_linear_artifact(
    const QuantizedLinearArtifact& artifact, QuantizationLimits limits = {});

/**
 * @brief Decodes and canonicality-checks one complete Linear artifact.
 * @param bytes Borrowed complete encoded artifact.
 * @param limits Independent byte, dimension and metadata ceilings.
 * @return Independently owned validated artifact; input storage is not
 * retained.
 */
QuantizedLinearArtifact deserialize_linear_artifact(
    std::span<const std::byte> bytes, QuantizationLimits limits = {});

/** @brief Ordered per-output-channel Linear weights composed of canonical Q0.
 *
 * Channel i is output row i and has exactly one output and an optional scalar
 * bias. Every channel has identical calibration bytes and input width. The
 * aggregate codec binds dimensions, channel indices, weights, biases and
 * calibration to one canonical byte sequence; its SHA-256 is the complete
 * model identity. Q0 v1 and its storage-only capability claims stay unchanged.
 */
struct PerOutputChannelLinearArtifact {
  std::uint32_t artifact_version = 1U;  ///< Aggregate schema version.
  std::size_t in_features = 0U;         ///< Shared positive input width.
  std::size_t out_features = 0U;        ///< Exact ordered channel count.
  std::vector<QuantizedLinearArtifact> channels;  ///< Output rows in order.
};

/** @brief Quantizes each output row with its own symmetric weight scale.
 * @param model Borrowed row-major [out,in] FP32 weights and optional bias.
 * @param calibration Complete common activation calibration evidence.
 * @param limits Feature/channel, metadata and complete aggregate byte bounds.
 * @return Owned ordered Q0 rows; no inference, fallback or kernel claim.
 * @throws Error For invalid, nonfinite or out-of-budget source data.
 */
PerOutputChannelLinearArtifact quantize_linear_per_output_channel(
    const FloatLinearModel& model, const CalibrationResult& calibration,
    QuantizationLimits limits = {});

/** @brief Validates exact dimensions, channel count and shared calibration.
 * @param artifact Owned aggregate to inspect without mutation.
 * @param limits Channel/feature and complete canonical byte ceilings.
 * @throws Error For inconsistent channels, bias presence, lineage or budgets.
 */
void validate_per_output_channel_linear_artifact(
    const PerOutputChannelLinearArtifact& artifact,
    QuantizationLimits limits = {});

/** @brief Encodes canonical little-endian ordered per-output-channel bytes.
 * @param artifact Complete aggregate; channel i is output i.
 * @param limits Complete aggregate byte, feature and calibration ceilings.
 * @return Owned bytes binding all channel indices, Q0 payloads and biases.
 * @throws Error For invalid fields or an exceeded complete byte ceiling.
 */
std::vector<std::byte> serialize_per_output_channel_linear_artifact(
    const PerOutputChannelLinearArtifact& artifact,
    QuantizationLimits limits = {});

/** @brief Decodes one complete canonical per-output-channel aggregate.
 * @param bytes Immutable source bytes; no caller storage is retained.
 * @param limits Aggregate byte, channel, feature and calibration ceilings.
 * @return Owned aggregate with identical calibration on every channel.
 * @throws Error For malformed, reordered, truncated or out-of-budget bytes.
 * @note Verify the complete aggregate SHA-256 against an independent identity
 * to detect a valid model whose rows or biases have intentionally changed.
 */
PerOutputChannelLinearArtifact deserialize_per_output_channel_linear_artifact(
    std::span<const std::byte> bytes, QuantizationLimits limits = {});

/**
 * @brief Explicit static PTQ preparation for the separate ORT CPU adapter.
 *
 * Portable Q0 storage stays symmetric int8 and makes no kernel claim. This
 * derived representation transposes [out,in] to [in,out] and adds 128 to each
 * weight without changing its integer value relative to zero point 128. ORT
 * owns machine-specific packing. Preparation alone does not prove execution,
 * quality, speedup, ISA availability, or backend installation. No ORT types or
 * runtime dependency enter the foundation-only quantization component.
 */
struct StaticInt8LinearPreparation {
  std::string profile_id =
      "jyppx.linear.ptq.ort.u8u8/v1";      ///< Exact derived recipe identity.
  std::string runtime_version = "1.23.2";  ///< Required ORT CPU version.
  std::uint32_t onnx_opset = 17U;  ///< Official ONNX operator-set version.
  std::size_t in_features = 0U;    ///< Positive input width, at most 65536.
  std::size_t out_features = 0U;   ///< Positive output width, at most 65536.
  float activation_scale = 1.0F;   ///< Positive normal FP32 static input scale.
  float weight_scale = 1.0F;   ///< Positive normal FP32 reconstruction scale.
  float product_scale = 1.0F;  ///< Nearest-even FP32 product of both scales.
  std::uint8_t activation_zero_point = 128U;  ///< Unsigned input zero point.
  std::uint8_t weight_zero_point = 128U;  ///< Unsigned repacked weight zero.
  std::vector<std::uint8_t> weights;      ///< Owned row-major [in,out] bytes.
  std::vector<float> bias;  ///< Optional owned finite FP32 output bias.
  ArtifactDigest source_artifact_digest;  ///< Complete canonical Q0 identity.
};

/**
 * @brief Prepares one calibrated Linear for unsigned INT8/int32 computation.
 *
 * Inputs are not retained or changed. The returned representation owns its
 * storage and can be read concurrently. Static activation QuantizeLinear uses
 * nearest-even rounding and saturation to [0,255]; integer dot products
 * accumulate into int32, followed by FP32 rescaling and optional FP32 bias.
 * This function performs no model inference and never sets compute-quantized.
 *
 * @param artifact Validated portable source with complete calibration lineage.
 * @param limits Independent source, feature and preparation payload ceilings.
 * @return Owned versioned recipe, canonical source SHA-256 and repacked data.
 * @throws InvalidArgument If dimensions or source metadata are unsupported.
 * @throws NumericalError If FP32 scales, their product, or a conservative
 * saturated-input output bound are not representable as finite normal scales
 * and finite FP32 output. Scale conversion ignores ambient rounding mode.
 * @throws ResourceExhausted If a declared source or payload budget is exceeded.
 */
StaticInt8LinearPreparation prepare_static_int8_linear(
    const QuantizedLinearArtifact& artifact, QuantizationLimits limits = {});

/**
 * @brief Encodes a bounded diagnostic view of static INT8 preparation.
 *
 * The JSON binds repacked bytes to the canonical Q0 source and its complete
 * calibration identity. Scales are encoded as exact FP32 bit patterns and
 * payloads as canonical base64. This describes derived conversion state, not a
 * new portable weight format or runtime execution report. The separate C++
 * native exporter consumes the original Q0 artifact directly; this optional
 * JSON is a diagnostic/interchange view and is not its source of authority.
 *
 * @param artifact Complete immutable canonical Q0 source.
 * @param limits Source and complete UTF-8 JSON byte ceilings.
 * @return Owned diagnostic JSON describing exactly one validated preparation.
 * @throws ResourceExhausted If the complete JSON exceeds max_artifact_bytes.
 */
std::string static_int8_linear_export_json(
    const QuantizedLinearArtifact& artifact, QuantizationLimits limits = {});

/** @brief Machine-readable execution and quality evidence for one artifact. */
struct QuantizationReport {
  std::uint32_t schema_version =
      1U;                 ///< Supported machine-readable report schema.
  std::string format_id;  ///< Exact portable storage format identity.
  std::string execution_claim =
      "storage-quantized";          ///< Stable implementation-plane claim.
  bool compute_quantized = false;   ///< Always false for the preview runtime.
  bool qat = false;                 ///< Always false; QAT is deferred.
  std::size_t artifact_bytes = 0U;  ///< Complete canonical artifact byte count.
  std::size_t fallback_modules =
      0U;  ///< Modules executing ordinary FP32 operations.
  std::size_t unsupported_modules =
      0U;  ///< Modules rejected as unavailable by the recipe.
  QuantizationErrorReport
      weight_error;  ///< Measured persistent weight reconstruction error.
  CalibrationLineage calibration;  ///< Exact source and preprocessing evidence.
};

/**
 * @brief Runtime that dequantizes weights and computes FP32 output.
 * Construction owns a validated artifact and a dequantized weight copy. Const
 * calls share immutable state and return independent output storage; caller
 * spans must remain valid throughout each call. No quantized kernel is used.
 */
class StorageQuantizedLinearRuntime final {
 public:
  /**
   * @brief Loads a validated storage-only artifact without backend packing.
   * @param artifact Owned portable artifact to validate and reconstruct.
   * @param limits Independent construction resource ceilings.
   */
  explicit StorageQuantizedLinearRuntime(QuantizedLinearArtifact artifact,
                                         QuantizationLimits limits = {});
  /**
   * @brief Computes one or more FP32 rows using dequantized weights.
   * @param input Borrowed contiguous row-major finite FP32 input values.
   * @param batch Positive number of complete input rows.
   * @return Owned row-major FP32 output with batch times out_features elements.
   */
  std::vector<float> forward(std::span<const float> input,
                             std::size_t batch = 1U) const;
  /** @brief Returns the execution claim. @return Process-lifetime path token.
   */
  std::string_view execution_path() const noexcept;
  /** @brief Reports quantized execution. @return Always false for this preview.
   */
  bool compute_quantized() const noexcept;
  /** @brief Reports QAT availability. @return Always false for this preview. */
  bool qat_supported() const noexcept;
  /**
   * @brief Produces complete report evidence for a source comparison.
   * @param original_weights Borrowed FP32 source weights of matching extent.
   * @param limits Independent artifact and report resource ceilings.
   * @return Owned execution claims, error measurements and lineage.
   */
  QuantizationReport report(std::span<const float> original_weights,
                            QuantizationLimits limits = {}) const;

 private:
  QuantizedLinearArtifact
      artifact_;  ///< Owned validated canonical source evidence.
  std::vector<float>
      dequantized_weights_;  ///< Owned row-major FP32 execution weights.
};

/**
 * @brief Produces deterministic compact JSON report bytes for evidence.
 * @param report Complete validated execution and quality evidence.
 * @param limits Independent output and metadata ceilings.
 * @return Owned UTF-8 JSON with explicit storage-only execution claims.
 */
std::string quantization_report_json(const QuantizationReport& report,
                                     QuantizationLimits limits = {});

}  // namespace jyppx::liuding::quantization

#endif  // JYPPX_LIUDING_QUANTIZATION_QUANTIZATION_HPP
