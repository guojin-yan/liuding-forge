// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_NN_MODEL_CHECKPOINT_PARTICIPANT_HPP
#define JYPPX_LIUDING_NN_MODEL_CHECKPOINT_PARTICIPANT_HPP
/**
 * @file model_checkpoint_participant.hpp
 * @brief Defines bounded, storage-neutral checkpoint state for LibTorch models.
 */

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/nn/model_adapter.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace jyppx::liuding::nn {

/**
 * @brief Configures model checkpoint identity and resource limits.
 *
 * Limits are enforced while describing, saving, and loading state. The model
 * participant writes a versioned topology blob plus one raw tensor blob per
 * parameter or buffer; it does not use an executable pickle archive.
 */
struct ModelCheckpointParticipantOptions {
  std::string participant_id =
      "model.main";  ///< Stable registry ID for this model instance.
  std::size_t max_tensors =
      100000;  ///< Positive maximum parameter and buffer count.
  std::size_t max_state_key_bytes =
      4096;  ///< Positive maximum encoded bytes in one state key.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL * 1024ULL;  ///< Maximum topology metadata size.
  std::uint64_t max_tensor_bytes =
      16ULL * 1024ULL * 1024ULL *
      1024ULL;  ///< Maximum uncompressed bytes in one tensor.
  std::uint64_t max_total_bytes =
      64ULL * 1024ULL * 1024ULL *
      1024ULL;  ///< Maximum topology plus tensor bytes.
};

/**
 * @brief Exposes one model's parameters and buffers as an exact participant.
 *
 * State keys, parameter/buffer roles, scalar types, shapes, strides, storage
 * aliases, storage offsets, and `requires_grad` flags form the versioned
 * topology contract. Tensor values are serialized as little-endian raw bytes
 * in logical contiguous order and restored onto each current tensor's device.
 *
 * Loading performs manifest, topology, field-size, and backend allocation
 * validation before modifying the model. Current tensors are snapshotted
 * before commit; a failed copy restores every attempted tensor in reverse
 * order. The participant and retained model are not internally synchronized.
 * Callers must quiesce model mutation during describe, save, and load.
 */
class ModelCheckpointParticipant final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a participant over a shared model state view.
   * @param state Non-null state view retained for the participant lifetime.
   * @param options Stable identity and positive resource limits.
   * @throws InvalidArgument If state, options, topology, or platform byte order
   * is unsupported.
   */
  explicit ModelCheckpointParticipant(
      std::shared_ptr<ModelStateView> state,
      ModelCheckpointParticipantOptions options = {});

  /**
   * @brief Creates a participant retaining an adapter's underlying module.
   * @param adapter Adapter whose shared module state is retained independently.
   * @param options Stable identity and positive resource limits.
   * @throws InvalidArgument If options, topology, or platform byte order is
   * unsupported.
   */
  explicit ModelCheckpointParticipant(
      const ModelAdapter& adapter,
      ModelCheckpointParticipantOptions options = {});

  /**
   * @brief Returns the stable registry identifier.
   * @return The configured participant ID.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes the current model topology and bounded tensor fields.
   * @return An exact schema-version-1 model state manifest.
   * @throws InvalidArgument If current model state is empty or unsupported.
   * @throws ResourceExhausted If current state exceeds a configured limit.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes topology metadata and every model tensor value.
   * @param writer Storage-neutral destination provided by the coordinator.
   * @throws SerializationError If a current tensor cannot be encoded.
   * @throws ResourceExhausted If state exceeds a configured limit.
   * @throws BackendError If LibTorch cannot materialize canonical CPU bytes.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Restores model tensors after complete validation and allocation.
   * @param reader Source containing a schema-version-1 model snapshot.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest, policy, or topology differs.
   * @throws SerializationError If metadata or tensor bytes are malformed.
   * @throws ResourceExhausted If decoded state exceeds a configured limit.
   * @throws BackendError If LibTorch cannot decode or commit tensor values.
   * @throws InternalError If commit fails and model rollback is incomplete.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns mutable access to the retained model state view.
   * @return A reference valid for the participant lifetime.
   */
  ModelStateView& state() noexcept;

  /**
   * @brief Returns read-only access to the retained model state view.
   * @return A reference valid for the participant lifetime.
   */
  const ModelStateView& state() const noexcept;

  /**
   * @brief Returns the immutable participant configuration.
   * @return A reference valid for the participant lifetime.
   */
  const ModelCheckpointParticipantOptions& options() const noexcept;

 private:
  std::shared_ptr<ModelStateView>
      state_;  ///< Shared owner of the model state being checkpointed.
  ModelCheckpointParticipantOptions
      options_;  ///< Immutable identity and resource limits.
};

}  // namespace jyppx::liuding::nn

#endif  // JYPPX_LIUDING_NN_MODEL_CHECKPOINT_PARTICIPANT_HPP