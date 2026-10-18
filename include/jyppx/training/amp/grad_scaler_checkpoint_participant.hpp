// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TRAINING_AMP_GRAD_SCALER_CHECKPOINT_PARTICIPANT_HPP
#define JYPPX_TRAINING_AMP_GRAD_SCALER_CHECKPOINT_PARTICIPANT_HPP

/**
 * @file grad_scaler_checkpoint_participant.hpp
 * @brief Defines storage-neutral exact checkpointing for GradScaler state.
 */

#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/training/amp/grad_scaler.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace jyppx::training::amp {

/** @brief Configures GradScaler checkpoint decode and allocation limits. */
struct GradScalerCheckpointOptions {
  std::size_t max_identifier_bytes =
      4096;  ///< Maximum bytes in the stable scaler identity.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL;  ///< Maximum immutable descriptor bytes.
  std::uint64_t max_control_bytes =
      1024ULL;  ///< Maximum mutable scaler-state bytes.
  std::uint64_t max_total_bytes =
      65ULL * 1024ULL;  ///< Maximum topology plus control bytes.
};

/**
 * @brief Publishes one GradScaler as an exact checkpoint participant.
 *
 * Schema version 1 stores identity and immutable scaling options in
 * `topology`, and current scale plus the finite-cycle counter in `control`.
 * Saving and loading require a completed multi-optimizer `step/update` cycle.
 * Restore fully validates policy, manifest, topology, limits, and mutable
 * values before committing the non-throwing scalar assignment.
 */
class GradScalerCheckpointParticipant final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a participant retaining a shared scaler.
   * @param scaler Non-null scaler whose stable ID owns the participant slot.
   * @param options Positive resource bounds applied before decode.
   * @throws InvalidArgument If `scaler` or a resource limit is invalid.
   * @throws NotImplemented On a non-little-endian runtime.
   */
  explicit GradScalerCheckpointParticipant(
      std::shared_ptr<GradScaler> scaler,
      GradScalerCheckpointOptions options = {});

  /**
   * @brief Returns the scaler's stable registry identity.
   * @return An owned identity matching the published manifest.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes exact topology and control fields.
   * @return A validated schema-version-1 manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes immutable topology and typed boundary state.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws NotReady If an optimizer cycle is in progress.
   * @throws ResourceExhausted If state exceeds configured limits.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Validates and atomically restores exact scaler state.
   * @param reader Source containing a schema-version-1 scaler snapshot.
   * @param policy Restore policy that must accept exact state.
   * @throws NotReady If an optimizer cycle is in progress.
   * @throws CompatibilityError If manifest or topology is incompatible.
   * @throws SerializationError If a required field is malformed.
   * @throws ResourceExhausted If state exceeds configured limits.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns mutable access to the retained scaler.
   * @return A reference valid for the participant lifetime.
   */
  GradScaler& scaler() noexcept;

  /**
   * @brief Returns read-only access to the retained scaler.
   * @return A reference valid for the participant lifetime.
   */
  const GradScaler& scaler() const noexcept;

  /**
   * @brief Returns immutable checkpoint resource limits.
   * @return A reference valid for the participant lifetime.
   */
  const GradScalerCheckpointOptions& options() const noexcept;

 private:
  std::shared_ptr<GradScaler>
      scaler_;  ///< Shared scaler whose exact state is persisted.
  GradScalerCheckpointOptions
      options_;  ///< Immutable decode and manifest resource bounds.
};

}  // namespace jyppx::training::amp

#endif  // JYPPX_TRAINING_AMP_GRAD_SCALER_CHECKPOINT_PARTICIPANT_HPP
