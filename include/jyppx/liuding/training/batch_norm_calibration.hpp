// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TRAINING_BATCH_NORM_CALIBRATION_HPP
#define JYPPX_LIUDING_TRAINING_BATCH_NORM_CALIBRATION_HPP
/** @file batch_norm_calibration.hpp
 * @brief Transactional, training-input-only CPU BatchNorm2d calibration.
 */

#include <jyppx/liuding/foundation/artifact.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stop_token>

namespace jyppx::liuding::training {

/** @brief One CPU FP32 NCHW training batch, with no task target field. */
struct BatchNormCalibrationBatch {
  Tensor images;  ///< Already-preprocessed images on the model's device.
  ArtifactDigest training_source;  ///< Frozen training/preprocessing identity.
  std::shared_ptr<const void>
      owner;  ///< Retains an optional batch resource lease.
};

/** @brief Supplies one bounded batch; nullopt means exact stream exhaustion. */
using BatchNormCalibrationSource =
    std::function<std::optional<BatchNormCalibrationBatch>(std::stop_token)>;

/** @brief Freezes a finite pass and explicit per-call resource bounds. */
struct BatchNormCalibrationOptions {
  ArtifactDigest training_source;  ///< Must match every source batch.
  std::uint64_t expected_batches =
      0;  ///< Positive exact batch count, not a cap.
  std::uint64_t expected_samples = 0;  ///< Positive exact sample count.
  std::uint64_t max_input_bytes = 0;  ///< Positive bytes in one accepted input.
  std::uint64_t max_model_snapshot_bytes =
      0;  ///< Positive encoded model bound.
};

/** @brief Describes a complete calibration, with no optimizer update. */
struct BatchNormCalibrationResult {
  std::uint64_t batches = 0;
  std::uint64_t samples = 0;
  std::uint64_t batch_norm_modules = 0;
  std::uint64_t model_snapshot_bytes = 0;
};

/** @brief Optionally atomically publishes a coordinated COMPLETED checkpoint.
 *
 * Called only after model checks and explicit mode/momentum/RNG restoration.
 * It must not mutate model tensors or RNG. Its final throwing operation must
 * be the atomic checkpoint commit; successful return is the commit decision.
 * A caller-owned stage participant must revert to READY if publication throws.
 * A store can publish a complete checkpoint and then report a directory-sync
 * failure. In that ambiguous case the caller must abort and reopen the durable
 * checkpoint; only a fully validated READY or COMPLETED state is authoritative.
 */
using BatchNormCalibrationCommit =
    std::function<void(const BatchNormCalibrationResult&)>;

/** @brief Replaces tracked BN2d statistics using one finite training pass.
 *
 * The module is eval except tracked BN2d children, which reset their running
 * statistics and use cumulative averaging. No gradients or optimizer APIs are
 * used. Cumulative averaging weights each batch equally; unequal tail batches
 * do not yield exact sample-weighted whole-dataset mean/variance. Only
 * discovered BN running_mean/running_var/num_batches_tracked may
 * change. CPU FP32 model parameters and CPU FP32/int64 buffers are supported;
 * other normalization variants, devices and custom mutable topology are
 * rejected or outside this first implementation's contract. The image recipe
 * initially exposes this only for its existing CPU ResNet18 architecture.
 *
 * A bounded full ModelCheckpointParticipant snapshot supplies rollback for
 * source, forward, cancellation, validation or publication failure. Original
 * per-module modes, BN momentum and CPU process RNG are explicitly restored
 * on both paths; rollback failure raises InternalError. Source callbacks must
 * be deterministic and must not modify model topology, gradients, optimizer
 * state, global runtime settings or external state. Exclusive ownership of
 * model/default RNG is required. A source's claimed identity is verified as an
 * identity contract, not as a sandbox against dishonest caller code.
 *
 * The accepted input and encoded snapshot have hard bounds. Input production,
 * LibTorch activations and backend workspace require separate caller budgets.
 * Existing model restore may transiently allocate two further model copies.
 * No partially calibrated state is a valid checkpoint boundary.
 *
 * @throws InvalidArgument, ResourceExhausted For incompatible inputs/bounds.
 * @throws Cancelled After restoring the entire pre-call model/RNG state.
 * @throws InternalError If restoration itself cannot finish.
 */
BatchNormCalibrationResult calibrate_batch_norm(
    nn::ModelAdapter& model, const BatchNormCalibrationSource& source,
    const BatchNormCalibrationOptions& options, std::stop_token stop = {},
    const BatchNormCalibrationCommit& commit = {});

}  // namespace jyppx::liuding::training
#endif
