// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_QUANTIZATION_TORCH_STATIC_LINEAR_HPP
#define JYPPX_LIUDING_QUANTIZATION_TORCH_STATIC_LINEAR_HPP
/** @file static_linear.hpp
 * @brief Official LibTorch CPU static INT8 Linear from canonical Q0/QC1.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::quantization_torch {

/** @brief SDK observations, not a platform or model qualification. */
struct QuantizedCpuCapabilities {
  std::string libtorch_version;  ///< Compiled LibTorch version.
  std::string current_engine;    ///< Process-wide engine observed at query.
  std::vector<std::string> supported_engines;  ///< Unmodified SDK engine list.
  bool linear_prepack_registered = false;      ///< QuantizedCPU prepack kernel.
  bool linear_registered = false;              ///< QuantizedCPU Linear kernel.
};

/** @brief Queries official registrations without changing global state.
 * @return Owned SDK observations; a listed engine is not a measured product.
 */
QuantizedCpuCapabilities quantized_cpu_capabilities();

/** @brief Logical ceilings, not a native allocator or RSS hard limit. */
struct StaticLinearOptions {
  std::optional<std::string>
      expected_engine;  ///< Optional ONEDNN/X86 equality constraint.
  quantization::QuantizationLimits artifact_limits;  ///< Q0/QC1 owner limits.
  std::size_t max_batch = 4096U;  ///< Positive maximum two-dimensional batch.
  std::size_t max_io_bytes = 64U * 1024U * 1024U;  ///< Logical I/O workspace.
};

/** @brief Immutable conversion identity; no speed or task-quality claim. */
struct StaticLinearInfo {
  std::string profile_id;        ///< Per-tensor or per-output-channel recipe.
  std::string engine;            ///< Engine captured during official packing.
  std::string libtorch_version;  ///< Exact compiled API version.
  ArtifactDigest source_artifact_digest;     ///< Complete original Q0/QC1 SHA.
  ArtifactDigest output_calibration_digest;  ///< Complete canonical CA1 SHA.
  std::size_t in_features = 0U;              ///< Positive input width.
  std::size_t out_features = 0U;             ///< Positive output width.
  bool per_output_channel = false;           ///< Official weight axis is zero.
  float input_scale = 1.0F;              ///< Frozen normal FP32 input scale.
  float output_scale = 1.0F;             ///< Frozen normal FP32 output scale.
  std::int64_t input_zero_point = 128;   ///< Static QUInt8 input zero.
  std::int64_t output_zero_point = 128;  ///< Static QUInt8 output zero.
  std::vector<float> weight_scales;      ///< One value or one per output row.
  quantization::CalibrationResult output_calibration;  ///< Owned CA1 evidence.
};

/** @brief Process-local official packed Linear, with no floating fallback.
 *
 * Methods on one live object serialize calls to the packed backend. Returned
 * tensors own their storage; borrowed inputs must remain immutable during a
 * call. Move/destruction require caller exclusion. No method changes qEngine;
 * the application must freeze global engine configuration before concurrent
 * work. External concurrent setQEngine is outside this local mutex contract.
 * Packed objects are not portable artifacts and are rebuilt at load.
 */
class StaticLinear final {
 public:
  /** @brief Loads one canonical per-tensor Q0 and packs official QInt8 weight.
   * @param bytes Complete borrowed Q0 bytes, snapshotted before parsing.
   * @param expected_digest Independently approved complete-byte SHA-256.
   * @param output_calibration_bytes Complete canonical calibration result
   * from FP32 training/calibration outputs, snapshotted before parsing.
   * @param expected_output_digest Independent complete-byte calibration SHA.
   * @param options Logical ceilings and optional expected-engine constraint.
   * @return Owned process-local runtime; no caller bytes are retained.
   * @throws Error For identity, format, numeric, budget or backend failure.
   */
  static StaticLinear load_per_tensor(
      std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
      std::span<const std::byte> output_calibration_bytes,
      const ArtifactDigest& expected_output_digest,
      StaticLinearOptions options = {});

  /** @brief Loads canonical QC1 rows as official per-channel QInt8 axis zero.
   * @param bytes Complete borrowed QC1 bytes, snapshotted before parsing.
   * @param expected_digest Independent SHA-256 of the complete aggregate.
   * @param output_calibration_bytes Frozen canonical FP32 output evidence.
   * @param expected_output_digest Independent SHA of the complete evidence.
   * @param options Logical ceilings and optional expected-engine constraint.
   * @return Owned runtime with one scale/zero pair per output weight row.
   * @throws Error For invalid aggregate, identity, calibration or backend.
   */
  static StaticLinear load_per_output_channel(
      std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
      std::span<const std::byte> output_calibration_bytes,
      const ArtifactDigest& expected_output_digest,
      StaticLinearOptions options = {});

  /** @brief Releases official packed state without serializing it. */
  ~StaticLinear();
  /** @brief Transfers runtime ownership; source methods then throw NotReady.
   * @param other Live or already moved-from runtime, caller excluded.
   */
  StaticLinear(StaticLinear&& other) noexcept;
  /** @brief Replaces state by moving; both objects require caller exclusion.
   * @param other Source runtime.
   * @return This runtime after transfer.
   */
  StaticLinear& operator=(StaticLinear&& other) noexcept;
  StaticLinear(const StaticLinear&) = delete;
  StaticLinear& operator=(const StaticLinear&) = delete;

  /** @brief Quantizes finite contiguous CPU FP32 [batch,in] with fixed qparams.
   * @param input Borrowed immutable values; no implicit device/dtype transfer.
   * @return Owned CPU QUInt8 tensor; nearest-even mode is required.
   * @throws Error For numeric, budget, dtype, shape, engine or backend failure.
   */
  Tensor quantize_input(const Tensor& input) const;

  /** @brief Runs quantized::linear and returns actual static QUInt8 output.
   * @param input Contiguous CPU QUInt8 [batch,in] with exact frozen qparams.
   * @return Owned [batch,out] quantized tensor with frozen output qparams.
   * @throws Error For incompatible input/engine; never falls back to FP32.
   */
  Tensor forward_quantized(const Tensor& input) const;

  /** @brief Official input quantization, integer Linear, final dequantization.
   * @param input Borrowed finite contiguous CPU FP32 [batch,in].
   * @return Owned FP32 [batch,out], including output requantization error.
   * @throws Error For the same invalid conditions as the explicit path.
   */
  Tensor forward(const Tensor& input) const;

  /** @brief Borrows immutable metadata until move/destruction.
   * @return Conversion/packing identity; not a timing or task-quality report.
   * Metadata records the packing engine; this does not query live availability.
   * @throws NotReady For a moved-from runtime.
   */
  const StaticLinearInfo& info() const;

 private:
  struct Impl;
  explicit StaticLinear(std::unique_ptr<Impl> impl);
  Impl& require_impl() const;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jyppx::liuding::quantization_torch

#endif  // JYPPX_LIUDING_QUANTIZATION_TORCH_STATIC_LINEAR_HPP
