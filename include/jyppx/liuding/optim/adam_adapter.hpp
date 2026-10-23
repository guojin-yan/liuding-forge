// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_OPTIM_ADAM_ADAPTER_HPP
#define JYPPX_LIUDING_OPTIM_ADAM_ADAPTER_HPP
/**
 * @file adam_adapter.hpp
 * @brief Defines the delegated Adam adapter and its checkpoint participant.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/optim/optimizer_adapter.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace jyppx::liuding::optim {

/**
 * @brief Configures one delegated Adam parameter group.
 *
 * Values follow `torch::optim::AdamOptions`. All numeric values must be finite.
 * Learning rate and weight decay are non-negative, both beta values are in
 * `[0, 1)`, and epsilon is positive. Only `learning_rate` may change after
 * construction; every other option is part of checkpoint topology identity.
 */
struct AdamAdapterOptions {
  double learning_rate = 0.001;  ///< Non-negative step size for this group.
  double beta1 = 0.9;  ///< First-moment exponential decay factor in `[0, 1)`.
  double beta2 =
      0.999;  ///< Second-moment exponential decay factor in `[0, 1)`.
  double epsilon = 1e-8;      ///< Positive denominator stability constant.
  double weight_decay = 0.0;  ///< Non-negative coupled L2 penalty coefficient.
  bool amsgrad = false;       ///< Whether to retain the AMSGrad maximum moment.
};

/**
 * @brief Defines a stable parameter group delegated to LibTorch Adam.
 */
struct AdamParameterGroup {
  std::string group_id;  ///< Stable nonempty identity within an optimizer.
  std::vector<OptimizerParameterBinding>
      parameters;  ///< Nonempty, uniquely identified model parameters.
  AdamAdapterOptions options;  ///< Validated algorithm options for this group.
};

/**
 * @brief Runs Adam through the supported LibTorch implementation.
 *
 * The adapter owns native optimizer state while parameter tensors remain model
 * owned. It deliberately does not expose the native optimizer, mutable groups,
 * or pointer-keyed state. Stable identity and checkpoint state are provided by
 * the JYPPX descriptor types and `AdamOptimizerCheckpointParticipant`.
 *
 * Instances are not internally synchronized. Callers must serialize gradient,
 * step, learning-rate, and checkpoint operations. Backend failure during
 * `step()` may leave native optimizer state modified; callers must stop or
 * recover from a checkpoint before continuing.
 */
class AdamOptimizerAdapter final : public OptimizerAdapter {
 public:
  /** @brief Preserves the common closure type on the concrete adapter API. */
  using LossClosure = OptimizerAdapter::LossClosure;

  /**
   * @brief Creates an Adam optimizer from stable parameter groups.
   * @param optimizer_id Stable nonempty identity for scheduler/checkpoint
   * links.
   * @param groups Nonempty groups with unique IDs and parameter identities.
   * @throws InvalidArgument If topology, tensors, or options are invalid.
   * @throws BackendError If LibTorch rejects otherwise validated construction.
   */
  explicit AdamOptimizerAdapter(std::string optimizer_id,
                                std::vector<AdamParameterGroup> groups);

  /** @brief Releases the opaque LibTorch optimizer implementation. */
  ~AdamOptimizerAdapter() override;

  /** @brief Prevents duplicating pointer-bound native optimizer state. */
  AdamOptimizerAdapter(const AdamOptimizerAdapter&) = delete;

  /** @brief Prevents assigning pointer-bound native optimizer state. */
  AdamOptimizerAdapter& operator=(const AdamOptimizerAdapter&) = delete;

  /** @brief Prevents moving an object retained by checkpoint participants. */
  AdamOptimizerAdapter(AdamOptimizerAdapter&&) = delete;

  /** @brief Prevents move assignment of participant-retained state. */
  AdamOptimizerAdapter& operator=(AdamOptimizerAdapter&&) = delete;

  /**
   * @brief Applies one LibTorch Adam update and records successful completion.
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
  const std::vector<AdamParameterGroup>& groups() const noexcept;

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
  friend class AdamOptimizerCheckpointParticipant;

  std::unique_ptr<Impl> impl_;  ///< Non-null native implementation owner.
};

/**
 * @brief Configures Adam checkpoint resource limits.
 */
struct AdamOptimizerCheckpointOptions {
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
      1024ULL;  ///< Maximum bytes in one moment tensor.
  std::uint64_t max_total_bytes =
      64ULL * 1024ULL * 1024ULL *
      1024ULL;  ///< Maximum metadata plus bounded tensor fields.
};

/**
 * @brief Exposes delegated Adam state through the checkpoint contract.
 *
 * The current writer emits schema version 2 with immutable stable topology,
 * bounded producer/runtime provenance, completed step count, current learning
 * rates, per-parameter native step counters, first and second moment tensors,
 * and optional AMSGrad maximum tensors. The topology includes stable group and
 * parameter IDs plus dtype, device, shape, stride, immutable Adam options, and
 * the compile-time LibTorch version. The reader accepts the immediately prior
 * schema version 1 as an explicit migration input; schema version 1 has no
 * producer field. Native pointer keys and LibTorch archive formats are never
 * persisted.
 *
 * Loading validates every field and allocates all candidate tensors before
 * swapping native state. Any exception before commit leaves the optimizer
 * unchanged. The participant shares ownership of its adapter and is not
 * internally synchronized.
 */
class AdamOptimizerCheckpointParticipant final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a checkpoint participant retaining an Adam adapter.
   * @param optimizer Non-null delegated optimizer shared with the trainer.
   * @param options Positive resource limits for state preflight and decoding.
   * @throws InvalidArgument If the optimizer, options, or topology is invalid.
   */
  explicit AdamOptimizerCheckpointParticipant(
      std::shared_ptr<AdamOptimizerAdapter> optimizer,
      AdamOptimizerCheckpointOptions options = {});

  /**
   * @brief Returns the optimizer's stable registry identifier.
   * @return The immutable optimizer ID owned by the retained adapter.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes topology, control, and fixed Adam state fields.
   * @return An exact current schema-version-2 optimizer manifest.
   * @throws InvalidArgument If current native state violates the descriptor.
   * @throws ResourceExhausted If state exceeds a configured limit.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes stable topology and current delegated Adam state.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws SerializationError If native state cannot be represented.
   * @throws ResourceExhausted If state exceeds a configured limit.
   * @throws BackendError If LibTorch cannot materialize canonical bytes.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Validates and atomically restores delegated Adam state.
   * @param reader Source containing a current schema-version-2 or N-1
   * schema-version-1 Adam snapshot.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest, policy, or topology differs.
   * @throws SerializationError If a required field is malformed.
   * @throws InvalidArgument If the checkpoint resource options are invalid.
   * @throws ResourceExhausted If decoded state exceeds a configured limit.
   * @throws BackendError If LibTorch cannot allocate restored state.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns mutable access to the retained optimizer adapter.
   * @return A reference valid for the participant lifetime.
   */
  AdamOptimizerAdapter& optimizer() noexcept;

  /**
   * @brief Returns read-only access to the retained optimizer adapter.
   * @return A reference valid for the participant lifetime.
   */
  const AdamOptimizerAdapter& optimizer() const noexcept;

  /**
   * @brief Returns the immutable checkpoint configuration.
   * @return A reference valid for the participant lifetime.
   */
  const AdamOptimizerCheckpointOptions& options() const noexcept;

 private:
  std::shared_ptr<AdamOptimizerAdapter>
      optimizer_;  ///< Shared delegated optimizer whose state is persisted.
  AdamOptimizerCheckpointOptions
      options_;  ///< Immutable checkpoint identity and resource limits.
};

}  // namespace jyppx::liuding::optim

#endif  // JYPPX_LIUDING_OPTIM_ADAM_ADAPTER_HPP