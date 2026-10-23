// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_INTEROP_MODEL_WEIGHT_IMPORT_HPP
#define JYPPX_LIUDING_INTEROP_MODEL_WEIGHT_IMPORT_HPP
/** @file model_weight_import.hpp
 * @brief Optional nn adapter applying verified safetensors through staged
 * state.
 */

#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/interop/weight_import_report.hpp>

#include <cstdint>
#include <span>
#include <stop_token>

namespace jyppx::liuding::nn {
class ModelStateView;
}

namespace jyppx::liuding::interop {

/** @brief Explicit mapping, decoding and staging bounds for model import. */
struct ModelWeightImportOptions {
  WeightImportOptions mapping;  ///< Strict by default; partial requires opt-in.
  WeightImportLimits mapping_limits;  ///< Complete source/target/report limits.
  SafeTensorsLimits tensor_limits;  ///< Complete artifact and metadata bounds.
  std::uint64_t max_staging_bytes =
      512U * 1024U * 1024U;  ///< CPU/device stage charge.
};

/** @brief Separates mapping diagnostics from actual applied model state. */
struct ModelWeightImportResult {
  WeightImportReport
      plan;              ///< Complete immutable mapping and retained entries.
  bool applied = false;  ///< True only after the complete staged state commit.
  std::uint64_t applied_tensors =
      0U;  ///< Assigned named parameters and buffers.
  std::uint64_t charged_staging_bytes =
      0U;  ///< Conservative temporary byte charge.
};

/** @brief Verifies, stages and applies complete safetensors state without
 * pickle.
 * @param model Shared model state; caller provides exclusive quiescent access.
 * @param artifact Complete stable safetensors bytes, retained unchanged for
 * call.
 * @param expected_digest Independently trusted SHA-256 of complete artifact.
 * @param options Explicit strict/partial mapping and resource bounds.
 * @param stop Cancellation honored before mutation, never mid-commit.
 * @param deadline Absolute operation deadline honored before mutation.
 * @return Mapping diagnostics with applied=false for a rejected mapping, or a
 * complete applied state. Unsupported conversions fail before any mutation.
 * @throws Error For invalid input, digest mismatch, unsupported target
 * topology, staging/device failure, cancellation/deadline, or commit failure.
 * @note Requires the optional interop->nn adapter. Supports independent dense
 * contiguous leaf CPU/CUDA parameters/buffers. Views, autograd history,
 * inference/lazy neg/conj tensors, aliased state and existing gradients are
 * rejected. Execute outside inference mode. The caller must finish all prior
 * device work and exclude state readers, writers, forward/backward execution,
 * optimizer steps and live autograd graphs for the entire call. Rechecking the
 * topology is diagnostic, not synchronization. Data handles are replaced after
 * complete staging; callers must discard external storage views and rebuild
 * optimizer state before training.
 * Strict import preserves dtype and bits. Partial float conversions require the
 * per-key rule `jyppx.float-cast.v1` and use LibTorch conversion semantics.
 * No code, adapter or executable is loaded from the artifact. Source digest is
 * integrity evidence, not producer authentication. Invalid mappings never
 * mutate the model. A commit failure attempts to restore every attempted data
 * handle, including the failing assignment. If restoration is incomplete,
 * InternalError reports the failure count; discard and reconstruct the model
 * before use. Backend failures retain their original nested cause without
 * copying backend diagnostics into the top-level error. Cancellation/deadline
 * never interrupt commit or rollback.
 */
ModelWeightImportResult import_model_weights(
    nn::ModelStateView& model, std::span<const std::byte> artifact,
    const ArtifactDigest& expected_digest,
    const ModelWeightImportOptions& options = {}, std::stop_token stop = {},
    TelemetryDeadline deadline = std::nullopt);

}  // namespace jyppx::liuding::interop
#endif  // JYPPX_LIUDING_INTEROP_MODEL_WEIGHT_IMPORT_HPP