// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_OPTIM_SGD_ADAPTER_HPP
#define JYPPX_OPTIM_SGD_ADAPTER_HPP

/**
 * @file sgd_adapter.hpp
 * @brief Defines the delegated SGD adapter and its checkpoint participant.
 */

#include <jyppx/core/types.hpp>
#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/optim/optimizer_adapter.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace jyppx::optim {

/**
 * @brief Configures one delegated SGD parameter group.
 *
 * Values follow `torch::optim::SGDOptions`. All numeric values must be finite;
 * learning rate, momentum, dampening, and weight decay are non-negative.
 * Nesterov momentum additionally requires positive momentum and zero
 * dampening.
 */
struct SgdAdapterOptions {
  double learning_rate = 0.01;  ///< Non-negative step size for this group.
  double momentum = 0.0;        ///< Non-negative momentum factor.
  double dampening = 0.0;       ///< Non-negative momentum dampening factor.
  double weight_decay = 0.0;    ///< Non-negative L2 penalty coefficient.
  bool nesterov = false;        ///< Whether to use Nesterov momentum.
};

/**
 * @brief Defines a stable parameter group delegated to LibTorch SGD.
 */
struct SgdParameterGroup {
  std::string group_id;  ///< Stable nonempty identity within an optimizer.
  std::vector<OptimizerParameterBinding>
      parameters;  ///< Nonempty, uniquely identified model parameters.
  SgdAdapterOptions options;  ///< Validated algorithm options for this group.
};

/**
 * @brief Runs SGD through the supported LibTorch implementation.
 *
 * The adapter owns native optimizer state while parameter tensors remain model
 * owned. It deliberately does not expose the native optimizer, mutable groups,
 * or pointer-keyed state. Stable identity and checkpoint state are provided by
 * the JYPPX descriptor types and `SgdOptimizerCheckpointParticipant`.
 *
 * Instances are not internally synchronized. Callers must serialize gradient,
 * step, learning-rate, and checkpoint operations. Backend failure during
 * `step()` may leave native optimizer state modified; callers must stop or
 * recover from a checkpoint before continuing.
 */
class SgdOptimizerAdapter final : public OptimizerAdapter {
 public:
  /** @brief Preserves the common closure type on the concrete adapter API. */
  using LossClosure = OptimizerAdapter::LossClosure;

  /**
   * @brief Creates an SGD optimizer from stable parameter groups.
   * @param optimizer_id Stable nonempty identity for scheduler/checkpoint
   * links.
   * @param groups Nonempty groups with unique IDs and parameter identities.
   * @throws InvalidArgument If topology, tensors, or options are invalid.
   * @throws BackendError If LibTorch rejects otherwise validated construction.
   */
  explicit SgdOptimizerAdapter(std::string optimizer_id,
                               std::vector<SgdParameterGroup> groups);

  /** @brief Releases the opaque LibTorch optimizer implementation. */
  ~SgdOptimizerAdapter() override;

  /** @brief Prevents duplicating pointer-bound native optimizer state. */
  SgdOptimizerAdapter(const SgdOptimizerAdapter&) = delete;

  /** @brief Prevents assigning pointer-bound native optimizer state. */
  SgdOptimizerAdapter& operator=(const SgdOptimizerAdapter&) = delete;

  /** @brief Prevents moving an object retained by checkpoint participants. */
  SgdOptimizerAdapter(SgdOptimizerAdapter&&) = delete;

  /** @brief Prevents move assignment of participant-retained state. */
  SgdOptimizerAdapter& operator=(SgdOptimizerAdapter&&) = delete;

  /**
   * @brief Applies one LibTorch SGD update and records successful completion.
   * @param closure Optional closure evaluated by LibTorch before the update.
   * @return The closure result, or an undefined tensor when no closure is used.
   * @throws InvalidArgument If a gradient does not match its parameter.
   * @throws NotImplemented If any bound parameter has a sparse gradient.
   * @throws BackendError If LibTorch cannot complete the step.
   * @throws ResourceExhausted If bookkeeping allocation fails.
   * @throws Any exception raised by `closure` is propagated unchanged.
   *
   * Gradient validation occurs after an optional closure and before LibTorch
   * mutates any parameter. Undefined gradients are skipped. Dense gradients
   * must match the parameter dtype, device, and shape. Sparse gradients are an
   * explicit unsupported capability in schema version 1.
   */
  Tensor step(LossClosure closure = {}) override;

  /**
   * @brief Clears gradients through the delegated optimizer.
   * @param set_to_none Whether to release gradient tensors instead of zeroing.
   * @throws BackendError If LibTorch cannot clear a gradient.
   */
  void zero_grad(bool set_to_none = true) override;

  /** @copydoc OptimizerAdapter::synchronize_parameter_placement */
  void synchronize_parameter_placement() override;

  /**
   * @brief Returns one current learning rate per stable parameter group.
   * @return Rates in the same order as `groups()`.
   */
  std::vector<double> learning_rates() const override;

  /**
   * @brief Atomically validates and sets every group learning rate.
   * @param learning_rates Finite non-negative values matching group count.
   * @throws InvalidArgument If a count or value is invalid.
   */
  void set_learning_rates(const std::vector<double>& learning_rates) override;

  /**
   * @brief Returns the immutable stable optimizer identity.
   * @return Identity used by the optimizer and associated schedulers.
   */
  const std::string& optimizer_id() const noexcept override;

  /**
   * @brief Returns immutable stable parameter-group identities.
   * @return IDs in native optimizer and learning-rate order.
   */
  const std::vector<std::string>& parameter_group_ids() const noexcept override;

  /**
   * @brief Returns the number of successful delegated steps.
   * @return Monotonic count restored by the checkpoint participant.
   */
  std::uint64_t completed_steps() const noexcept override;

  /**
   * @brief Returns the immutable stable optimizer descriptor.
   * @return Groups and bindings in native optimizer order.
   */
  const std::vector<SgdParameterGroup>& groups() const noexcept;

  /**
   * @brief Returns stable flattened parameter bindings.
   * @return Copies of lightweight tensor handles in checkpoint order.
   * @throws InvalidArgument If current topology violates its descriptor.
   */
  std::vector<OptimizerParameterBinding> parameter_bindings() const override;

  /**
   * @brief Returns the total number of bound parameters.
   * @return Sum of all stable group parameter counts.
   */
  std::size_t parameter_count() const noexcept override;

 private:
  /** @brief Opaque owner of LibTorch-specific optimizer state. */
  struct Impl;

  /** @brief Allows the participant to encode and atomically replace state. */
  friend class SgdOptimizerCheckpointParticipant;

  std::unique_ptr<Impl> impl_;  ///< Non-null native implementation owner.
};

/**
 * @brief Configures SGD checkpoint resource limits.
 */
struct SgdOptimizerCheckpointOptions {
  std::size_t max_groups = 4096;  ///< Positive maximum parameter-group count.
  std::size_t max_parameters =
      100000;  ///< Positive maximum total bound parameter count.
  std::size_t max_identifier_bytes =
      4096;  ///< Positive maximum bytes in one stable ID.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL * 1024ULL;  ///< Maximum descriptor metadata size.
  std::uint64_t max_producer_bytes =
      4096;  ///< Maximum producer/runtime compatibility metadata size.
  std::uint64_t max_control_bytes =
      16ULL * 1024ULL * 1024ULL;  ///< Maximum mutable control metadata size.
  std::uint64_t max_tensor_bytes =
      16ULL * 1024ULL * 1024ULL *
      1024ULL;  ///< Maximum bytes in one momentum tensor.
  std::uint64_t max_total_bytes =
      64ULL * 1024ULL * 1024ULL *
      1024ULL;  ///< Maximum metadata plus bounded tensor fields.
};

/**
 * @brief Exposes delegated SGD state through the checkpoint contract.
 *
 * Schema version 2 stores immutable stable topology, explicit producer/runtime
 * metadata, completed step count, current learning rates, and optional raw
 * momentum buffers. The topology includes stable group and parameter IDs plus
 * dtype, device, shape, stride, immutable SGD options, and the compile-time
 * LibTorch version. Native pointer keys and LibTorch archive formats are never
 * persisted.
 *
 * Loading accepts current schema version 2 and the immediately previous schema
 * version 1. Version 1 is migrated in memory by validating its frozen topology,
 * control, and raw momentum fields before constructing current candidate state;
 * no version-1 writer or legacy LibTorch archive reader is retained. Loading
 * validates every field and allocates all candidate tensors before swapping
 * native state. Any exception before commit leaves the optimizer unchanged.
 * The participant shares ownership of its adapter and is not internally
 * synchronized.
 */
class SgdOptimizerCheckpointParticipant final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a checkpoint participant retaining an SGD adapter.
   * @param optimizer Non-null delegated optimizer shared with the trainer.
   * @param options Positive resource limits for state preflight and decoding.
   * @throws InvalidArgument If the optimizer, options, or topology is invalid.
   */
  explicit SgdOptimizerCheckpointParticipant(
      std::shared_ptr<SgdOptimizerAdapter> optimizer,
      SgdOptimizerCheckpointOptions options = {});

  /**
   * @brief Returns the optimizer's stable registry identifier.
   * @return The immutable optimizer ID owned by the retained adapter.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes topology, producer, control, and fixed momentum fields.
   * @return An exact schema-version-2 optimizer manifest.
   * @throws InvalidArgument If current native state violates the descriptor.
   * @throws ResourceExhausted If state exceeds a configured limit.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes stable topology and current delegated SGD state.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws SerializationError If native state cannot be represented.
   * @throws ResourceExhausted If state exceeds a configured limit.
   * @throws BackendError If LibTorch cannot materialize canonical bytes.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Validates and atomically restores delegated SGD state.
   * @param reader Source containing a schema-version-2 or N-1 SGD snapshot.
   * @param policy Restore policy that must accept exact state.
   * @throws InvalidArgument If the stored manifest violates base invariants.
   * @throws CompatibilityError If manifest, policy, or topology differs.
   * @throws SerializationError If a required field is malformed.
   * @throws ResourceExhausted If decoded state exceeds a configured limit.
   * @throws BackendError If LibTorch cannot allocate restored state.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns mutable access to the retained optimizer adapter.
   * @return A reference valid for the participant lifetime.
   */
  SgdOptimizerAdapter& optimizer() noexcept;

  /**
   * @brief Returns read-only access to the retained optimizer adapter.
   * @return A reference valid for the participant lifetime.
   */
  const SgdOptimizerAdapter& optimizer() const noexcept;

  /**
   * @brief Returns the immutable checkpoint configuration.
   * @return A reference valid for the participant lifetime.
   */
  const SgdOptimizerCheckpointOptions& options() const noexcept;

 private:
  std::shared_ptr<SgdOptimizerAdapter>
      optimizer_;  ///< Shared delegated optimizer whose state is persisted.
  SgdOptimizerCheckpointOptions
      options_;  ///< Immutable checkpoint identity and resource limits.
};

}  // namespace jyppx::optim

#endif  // JYPPX_OPTIM_SGD_ADAPTER_HPP
