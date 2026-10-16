// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_FOUNDATION_RANDOM_CHECKPOINT_PARTICIPANT_HPP
#define JYPPX_FOUNDATION_RANDOM_CHECKPOINT_PARTICIPANT_HPP

/**
 * @file random_checkpoint_participant.hpp
 * @brief Defines exact storage-neutral checkpointing for process RNG state.
 */

#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/foundation/random.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace jyppx::core {

/** @brief Configures process RNG checkpoint identity, scope, and limits. */
struct RngCheckpointParticipantOptions {
  std::string participant_id =
      "rng.global";  ///< Stable nonempty coordinator registry identity.
  bool include_cuda =
      true;  ///< Whether every visible CUDA generator is part of exact state.
  std::size_t max_identifier_bytes =
      4096;  ///< Maximum bytes in `participant_id`.
  std::size_t max_cuda_devices =
      256;  ///< Maximum visible CUDA generator count accepted.
  std::uint64_t max_generator_state_bytes =
      16ULL * 1024ULL * 1024ULL;  ///< Maximum bytes for one generator state.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL;  ///< Maximum generator descriptor bytes.
  std::uint64_t max_total_bytes =
      512ULL * 1024ULL * 1024ULL;  ///< Maximum topology and all state bytes.
};

/**
 * @brief Publishes process-default CPU and CUDA generators as exact state.
 *
 * Schema version 1 stores generator count and state lengths in `topology`, a
 * required CPU byte field, and one required field per visible CUDA device.
 * Exact restore requires the same selected scope, device ordering, generator
 * state representation, and external synchronization with random consumers.
 */
class RngCheckpointParticipant final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates and validates a process RNG participant.
   * @param options Stable identity, selected generator scope, and limits.
   * @throws InvalidArgument If identity or a resource limit is invalid.
   * @throws ResourceExhausted If current generator state exceeds a limit.
   * @throws DeviceError If selected generators cannot be inspected.
   * @throws NotImplemented On a non-little-endian runtime.
   */
  explicit RngCheckpointParticipant(
      RngCheckpointParticipantOptions options = {});

  /**
   * @brief Returns the stable process RNG registry identity.
   * @return An owned identity matching the published manifest.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes the current exact generator topology and raw fields.
   * @return A validated schema-version-1 manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Snapshots and writes selected process generator state.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws ResourceExhausted If state exceeds configured limits.
   * @throws DeviceError If a selected generator cannot be read.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Validates and atomically restores selected process generators.
   * @param reader Source containing schema-version-1 process RNG state.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If policy, manifest, or topology differs.
   * @throws SerializationError If a required field is malformed.
   * @throws ResourceExhausted If decoded state exceeds configured limits.
   * @throws DeviceError If a backend generator operation fails.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns immutable checkpoint selection and resource limits.
   * @return A reference valid for the participant lifetime.
   */
  const RngCheckpointParticipantOptions& options() const noexcept;

 private:
  RngCheckpointParticipantOptions
      options_;  ///< Immutable identity, scope, and decode resource bounds.
};

}  // namespace jyppx::core

#endif  // JYPPX_FOUNDATION_RANDOM_CHECKPOINT_PARTICIPANT_HPP
