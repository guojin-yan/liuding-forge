// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_DIAGNOSTICS_HPP
#define JYPPX_LIUDING_TRAINING_DIAGNOSTICS_HPP
/**
 * @file diagnostics.hpp
 * @brief Defines opt-in, bounded and checkpointed training diagnostics.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/training/telemetry_adapter.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>

namespace jyppx::liuding::training {

class Trainer;

/** @brief Bounds one read-only gradient measurement before tensor allocation.
 */
struct GradientNormLimits {
  std::size_t max_tensors = 16384;  ///< Maximum supplied gradient handles.
  std::uint64_t max_elements = 1000000000;  ///< Maximum total scan elements.
  std::uint64_t tile_elements = 65536;  ///< Maximum FP64 working tile elements.
  std::uint64_t max_scratch_bytes =
      524304;  ///< Known tensor scratch: one FP64 tile and two scalars.
};

/** @brief Reports one stable L2 measurement of all supplied gradient values. */
struct GradientNormMeasurement {
  double l2 = 0.0;  ///< Global L2 norm, before clipping or optimizer updates.
  double seconds = 0.0;  ///< Host scan time, including scalar device transfers.
  std::uint64_t elements = 0;  ///< Actual scanned values, including zeros.
  std::uint64_t known_scratch_bytes = 0;  ///< Peak known tensor payload bound.
};

/**
 * @brief Measures a global L2 norm without flattening or mutating gradients.
 * @param gradients Defined real floating strided CPU/CUDA tensors, any strides.
 * @param limits Bounds checked for the complete input before tensor allocation.
 * @return A scaled FP64 reduction; an empty input has zero norm.
 * @throws InvalidArgument If options, dtype, layout, or device are invalid.
 * @throws ResourceExhausted If any configured scan or scratch bound is
 * exceeded.
 * @throws NumericalError If a value or the resulting norm is not finite.
 * @note Narrowed views restrict each copy to a tile. Backend-private allocator
 * and reduction workspace are outside the known tensor payload bound.
 */
GradientNormMeasurement measure_gradient_l2(
    std::span<const Tensor> gradients, const GradientNormLimits& limits = {});

/** @brief Selects optional extra work in one diagnostic participant. */
struct TrainingDiagnosticsOptions {
  std::string participant_id =
      "training.diagnostics";  ///< Stable checkpoint identity.
  std::uint64_t gradient_norm_every =
      0;  ///< Sample attempts N, 2N, ...; zero disables norm scans and clocks.
  bool measure_data_wait = false;  ///< Time successful nonempty pipeline reads.
  GradientNormLimits gradient_norm;  ///< Bounds for selected norm scans.
};

/** @brief Retains exact scalar-loss observation counts. */
struct DiagnosticLossCounts {
  std::uint64_t observations = 0;  ///< Defined scalar losses observed.
  std::uint64_t nan =
      0;  ///< Observed NaN losses; normal finite checks still fail.
  std::uint64_t infinity =
      0;  ///< Observed positive or negative infinity losses.
};

/** @brief Aggregates successful nonempty next() host-wall durations. */
struct DiagnosticDataWait {
  std::uint64_t batches = 0;  ///< Successful nonempty next() calls only.
  double seconds = 0.0;       ///< Sum of observed durations in SI seconds.
  double max_seconds = 0.0;   ///< Largest observed duration.
};

/** @brief Fixed-size typed run statistics, independent of callback scalars. */
struct TrainingDiagnosticsSnapshot {
  DiagnosticLossCounts training_loss;    ///< Training loss observations.
  DiagnosticLossCounts validation_loss;  ///< Validation loss observations.
  DiagnosticDataWait training_wait;  ///< Optional training next() durations.
  DiagnosticDataWait
      validation_wait;  ///< Optional validation next() durations.
  std::uint64_t update_attempts =
      0;  ///< Includes nonfinite and AMP-skipped cycles.
  std::uint64_t nonfinite_gradient_attempts =
      0;                                   ///< Failed finite checks/cycles.
  std::uint64_t amp_skipped_attempts = 0;  ///< Scaler-overflow skipped updates.
  std::uint64_t norm_samples = 0;   ///< Completed finite norm samples only.
  std::uint64_t norm_elements = 0;  ///< Total elements in completed samples.
  double last_gradient_l2 = 0.0;    ///< Last completed pre-clipping sample.
  double mean_gradient_l2 = 0.0;  ///< Running arithmetic mean of sampled norms.
  double max_gradient_l2 = 0.0;   ///< Largest completed norm sample.
  double norm_seconds = 0.0;      ///< Sum of completed scan host durations.
  std::uint64_t known_peak_scratch_bytes = 0;  ///< Peak known scan payload.
};

/**
 * @brief Returns typed semantics for every Trainer-owned diagnostic event key.
 * @return Bindings to merge into TrainingTelemetryAdapterOptions::metrics.
 * @note Terminal issue metrics are run-to-date boolean flags, not counters.
 * Exact uint64 counts remain in the participant snapshot and checkpoint.
 */
std::map<std::string, TrainingTelemetryMetricBinding>
training_diagnostic_metric_bindings();

/**
 * @brief Owns optional Trainer diagnostics as a current-only exact participant.
 *
 * Bind through FitRequest::diagnostics. Existing finite-loss/gradient safety
 * checks remain active without this object. Default options add no norm scan
 * or clock read. Sampling uses update attempts, including AMP overflow skips;
 * valid samples follow unscale/tail correction and precede norm clipping.
 * next() timing includes consumer transforms/transfers, excludes EOF/failures,
 * and is neither pure I/O time nor synchronized GPU execution time.
 *
 * Instances have one serialized owner. Checkpoints preserve captured counters,
 * durations and sampling position; future wall-clock durations are not exact.
 */
class TrainingDiagnostics final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates empty statistics with immutable enablement and scan bounds.
   * @param options Identity, sampling, timing, and resource policy.
   * @throws InvalidArgument If identity or limits are invalid.
   */
  explicit TrainingDiagnostics(TrainingDiagnosticsOptions options = {});

  /** @brief Prevents copying mutable participant state. */
  TrainingDiagnostics(const TrainingDiagnostics&) = delete;
  /** @brief Prevents assigning mutable participant state. */
  TrainingDiagnostics& operator=(const TrainingDiagnostics&) = delete;

  /**
   * @brief Returns immutable enablement, identity, and resource policy.
   * @return Options valid for this object's lifetime.
   */
  const TrainingDiagnosticsOptions& options() const noexcept;
  /**
   * @brief Returns a detached fixed-size copy of the current run statistics.
   * @return Exact integer counts and captured floating observations.
   */
  TrainingDiagnosticsSnapshot snapshot() const noexcept;

  /** @copydoc CheckpointParticipant::participant_id */
  std::string participant_id() const override;
  /** @copydoc CheckpointParticipant::describe_state */
  StateManifest describe_state() const override;
  /** @copydoc CheckpointParticipant::save_state */
  void save_state(StateWriter& writer) const override;
  /** @copydoc CheckpointParticipant::load_state */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

 private:
  friend class Trainer;
  /** @brief Observes a scalar already read by the normal loss safety check.
   * @param loss Scalar value, including NaN or infinity.
   * @param validation Whether this belongs to validation.
   */
  void observe_loss(double loss, bool validation);
  /** @brief Adds one successful pipeline duration atomically.
   * @param seconds Successful nonempty next() duration.
   * @param validation Whether this belongs to validation.
   */
  void observe_wait(double seconds, bool validation);
  /** @brief Advances the persistent attempt cursor and returns sample choice.
   * @return Whether this attempt is selected for norm measurement.
   */
  bool begin_gradient_attempt();
  /** @brief Records one nonfinite cycle, optionally skipped by AMP.
   * @param amp_skipped Whether the scaler skipped this cycle.
   */
  void observe_nonfinite_gradient(bool amp_skipped);
  /** @brief Commits one completed finite norm measurement atomically.
   * @param measurement Validated result of a selected gradient scan.
   */
  void observe_norm(const GradientNormMeasurement& measurement);
  /** @brief Returns finite issue flags for an existing terminal event.
   * @return Run-to-date finite boolean issue metrics.
   */
  std::map<std::string, double> terminal_metrics() const;

  TrainingDiagnosticsOptions options_;  ///< Immutable participant topology.
  TrainingDiagnosticsSnapshot state_;   ///< Bounded typed mutable observations.
};

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_DIAGNOSTICS_HPP
