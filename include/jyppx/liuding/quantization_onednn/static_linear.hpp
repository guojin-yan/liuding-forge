// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_QUANTIZATION_ONEDNN_STATIC_LINEAR_HPP
#define JYPPX_LIUDING_QUANTIZATION_ONEDNN_STATIC_LINEAR_HPP
/** @file static_linear.hpp
 * @brief Official oneDNN/ATen AVX2 static INT8 Linear from canonical Q0/QC1.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/quantization/quantization.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::quantization_onednn {

/** @brief Logical workspace ceilings; allocator/RSS limits remain external. */
struct StaticLinearOptions {
  quantization::QuantizationLimits
      artifact_limits;          ///< Bounded Q0/QC1/CA1 parsing.
  std::size_t max_batch = 64U;  ///< Positive maximum CPU [batch,in] row count.
  std::size_t max_io_bytes =
      1U << 20U;  ///< Owned arrays, client I/O and packing reserve.
};

/** @brief Immutable conversion metadata, independent of global qEngine. */
struct StaticLinearInfo {
  std::string profile_id;  ///< Versioned oneDNN/ATen AVX2 7+1 recipe.
  std::string backend;  ///< Official oneDNN matmul plus ATen INT64 composition.
  std::string
      implementation;  ///< Both component calls explicitly use gemm:jit.
  std::string onednn_version;    ///< Exact compiled oneDNN version.
  std::string libtorch_version;  ///< Exact compiled ATen/LibTorch API version.
  ArtifactDigest source_artifact_digest;  ///< Complete original Q0/QC1 SHA-256.
  ArtifactDigest
      output_calibration_digest;    ///< Complete canonical CA1 SHA-256.
  std::size_t in_features = 0U;     ///< Positive input width, at most 65536.
  std::size_t out_features = 0U;    ///< Positive output width, at most 65536.
  bool per_output_channel = false;  ///< One scale per output weight row.
  float input_scale = 1.0F;         ///< Frozen normal FP32 input scale.
  float output_scale = 1.0F;        ///< Frozen normal FP32 output scale.
  std::int64_t input_zero_point = 128;   ///< Original QUInt8 input zero.
  std::int64_t output_zero_point = 128;  ///< Static QUInt8 output zero.
  std::vector<float> weight_scales;      ///< One scale or one per output row.
  quantization::CalibrationResult output_calibration;  ///< Owned CA1 evidence.
};

/** @brief CPU static INT8 Linear using official oneDNN and ATen operations.
 *
 * Uses two explicit U8/S8-to-S32 gemm:jit calls and ATen INT64 reconstruction,
 * followed by ordered FP32 requantization. Original input bytes, S8 weights,
 * scales and calibration are preserved. No floating GEMM, qEngine or fallback
 * is used. Requires the pinned oneDNN 3.12.0 SEQ/NONE SDK, effective AVX2 and
 * nearest-even rounding. Each weight row must satisfy
 * 127*sum(abs(int64(weight))) <= 2^24, including potential oneDNN FP32 PP.
 *
 * Calls on one object serialize workspace reuse. Returned tensors own their
 * storage; borrowed inputs must remain immutable during a call. Move and
 * destruction require caller exclusion. Invalid input preflight is reusable;
 * failed workspace preparation or computation poisons state until reload.
 * Logical workspace includes a conservative source-derived packing estimate,
 * not a native allocator/RSS hard limit. Artifact metadata has separate limits.
 */
class StaticLinear final {
 public:
  /** @brief Loads canonical per-tensor Q0 and frozen FP32 output calibration.
   * @param bytes Complete borrowed Q0 bytes, snapshotted before parsing.
   * @param expected_digest Independently approved complete-byte source SHA.
   * @param output_calibration_bytes Complete canonical CA1 bytes.
   * @param expected_output_digest Independent complete-byte CA1 SHA.
   * @param options Bounded parsing, batch and logical workspace limits.
   * @return Owned process-local runtime; no caller bytes are retained.
   * @throws Error For identity, calibration, range, budget or backend failure.
   */
  static StaticLinear load_per_tensor(
      std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
      std::span<const std::byte> output_calibration_bytes,
      const ArtifactDigest& expected_output_digest,
      StaticLinearOptions options = {});

  /** @brief Loads canonical QC1 with independent output-row weight scales.
   * @param bytes Complete borrowed QC1 bytes, snapshotted before parsing.
   * @param expected_digest Independently approved aggregate SHA-256.
   * @param output_calibration_bytes Complete frozen canonical CA1 bytes.
   * @param expected_output_digest Independent complete-byte CA1 SHA.
   * @param options Bounded parsing, batch and logical workspace limits.
   * @return Owned runtime with unchanged original S8 weight rows.
   * @throws Error For identity, calibration, range, budget or backend failure.
   */
  static StaticLinear load_per_output_channel(
      std::span<const std::byte> bytes, const ArtifactDigest& expected_digest,
      std::span<const std::byte> output_calibration_bytes,
      const ArtifactDigest& expected_output_digest,
      StaticLinearOptions options = {});

  /** @brief Returns compiled SDK/archive/configuration identity after checks.
   * @return Owned identity strings; installation needs no original build files.
   * This records compiled provenance, not an executed-PC or speed certificate.
   * @throws CompatibilityError For unsupported SDK, ISA or rounding mode.
   */
  static std::map<std::string, std::string> dependency_identity();

  /** @brief Releases process-local oneDNN primitives and owned workspace. */
  ~StaticLinear();
  /** @brief Moves runtime ownership; source methods subsequently throw
   * NotReady.
   * @param other Source object, excluded from concurrent access.
   */
  StaticLinear(StaticLinear&& other) noexcept;
  /** @brief Replaces runtime ownership; both objects require caller exclusion.
   * @param other Source object.
   * @return This runtime after transfer.
   */
  StaticLinear& operator=(StaticLinear&& other) noexcept;
  StaticLinear(const StaticLinear&) = delete;
  StaticLinear& operator=(const StaticLinear&) = delete;

  /** @brief Quantizes finite contiguous CPU FP32 [batch,in] with frozen
   * qparams.
   * @param input Borrowed immutable FP32 values; no implicit dtype/device copy.
   * @return Owned QUInt8 tensor with zero point 128.
   * @throws Error For numeric, shape, budget, SDK or backend failure.
   */
  Tensor quantize_input(const Tensor& input) const;

  /** @brief Runs the integer composition and returns static QUInt8 output.
   * @param input CPU contiguous QUInt8 [batch,in] with exact frozen qparams.
   * @return Owned quantized [batch,out] tensor; never falls back to FP32 GEMM.
   * @throws Error For incompatible input, range, budget or backend failure.
   */
  Tensor forward_quantized(const Tensor& input) const;

  /** @brief Quantizes input, runs integer Linear and dequantizes final output.
   * @param input Borrowed finite contiguous CPU FP32 [batch,in].
   * @return Owned FP32 [batch,out], including static requantization error.
   * @throws Error For the same invalid conditions as the explicit path.
   */
  Tensor forward(const Tensor& input) const;

  /** @brief Borrows immutable conversion metadata until move/destruction.
   * @return Frozen metadata; no runtime qualification or performance claim.
   * @throws NotReady For a moved-from runtime.
   */
  const StaticLinearInfo& info() const;

 private:
  /** @brief Owns backend state and reusable workspace behind the public API. */
  struct Impl;
  /** @brief Adopts a fully prepared runtime implementation.
   * @param impl Exclusive ownership of the validated backend state.
   */
  explicit StaticLinear(std::unique_ptr<Impl> impl);
  /** @brief Borrows the owned implementation after checking moved-from state.
   * @return Implementation owned by this runtime until move or destruction.
   * @throws NotReady If this runtime has been moved from.
   */
  Impl& require_impl() const;
  std::unique_ptr<Impl> impl_;  ///< Exclusive backend state and workspace.
};

}  // namespace jyppx::liuding::quantization_onednn
#endif  // JYPPX_LIUDING_QUANTIZATION_ONEDNN_STATIC_LINEAR_HPP
