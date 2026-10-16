// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_OPTIM_STEP_LR_SCHEDULER_HPP
#define JYPPX_OPTIM_STEP_LR_SCHEDULER_HPP

/**
 * @file step_lr_scheduler.hpp
 * @brief Defines the descriptor-first StepLR scheduler and exact checkpoint.
 */

#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/optim/optimizer_rate_control.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace jyppx::optim {

/**
 * @brief Identifies the event that advances a scheduler.
 *
 * Numeric values are persisted in scheduler descriptors and must remain
 * stable across compatible releases.
 */
enum class SchedulerStepMode : std::uint8_t {
  batch = 1,   ///< Advances after a successful optimizer update.
  epoch = 2,   ///< Advances once after a completed training epoch.
  metric = 3,  ///< Advances when a caller supplies an evaluation metric.
  manual = 4,  ///< Advances only through an explicit application event.
};

/**
 * @brief Returns the stable lowercase name of a scheduler step mode.
 * @param mode Step mode to render.
 * @return A process-lifetime name, or `"unknown"` for an invalid value.
 */
std::string_view scheduler_step_mode_name(SchedulerStepMode mode) noexcept;

/**
 * @brief Stores one strongly typed scheduler configuration value.
 *
 * Variant alternatives and their order form part of the descriptor schema.
 * Persisted descriptors use explicit tags and never infer a type from text.
 */
using SchedulerConfigValue =
    std::variant<bool, std::int64_t, double, std::string,
                 std::vector<std::int64_t>, std::vector<double>,
                 std::vector<std::string>>;

/**
 * @brief Describes scheduler semantics independently of native objects.
 *
 * A fingerprint is a compact diagnostic derived from the complete canonical
 * descriptor excluding the fingerprint field itself. Compatibility decisions
 * compare the complete canonical descriptor and never trust only the hash.
 */
struct SchedulerDescriptor {
  std::uint32_t schema_version = 1;  ///< Positive descriptor schema version.
  std::string type_id;  ///< Stable scheduler algorithm and semantic identity.
  SchedulerStepMode step_mode =
      SchedulerStepMode::manual;  ///< Event that advances this scheduler.
  std::map<std::string, SchedulerConfigValue>
      config;  ///< Ordered, typed, algorithm-specific immutable options.
  std::string optimizer_id;  ///< Stable identity of the controlled optimizer.
  std::vector<std::string>
      optimizer_group_ids;  ///< Ordered identities of controlled groups.
  std::string optimizer_group_signature;  ///< Diagnostic group-order digest.
  std::optional<std::string>
      callable_id;  ///< Stable callable identity when semantics use a callback.
  std::vector<SchedulerDescriptor>
      children;  ///< Ordered child descriptors for composite schedulers.
  std::string fingerprint;  ///< Canonical descriptor diagnostic fingerprint.
};

/**
 * @brief Configures a descriptor-first StepLR scheduler.
 */
struct StepLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.main";  ///< Stable checkpoint participant identity.
  std::uint64_t step_size =
      1;  ///< Positive number of completed scheduler steps per decay.
  double gamma = 0.1;  ///< Finite non-negative decay multiplier.
};

/**
 * @brief Applies periodic learning-rate decay through a narrow optimizer API.
 *
 * The scheduler captures optimizer and ordered group identities at
 * construction. Every operation rejects later topology drift. Calling
 * `step()` increments the completed-step counter and multiplies all current
 * learning rates by `gamma` whenever the new count is divisible by
 * `step_size`.
 *
 * Instances are not internally synchronized. Callers must serialize
 * scheduler steps, optimizer steps, rate changes, and checkpoint operations.
 */
class StepLrScheduler final {
 public:
  /**
   * @brief Creates a StepLR scheduler for a stable optimizer topology.
   * @param optimizer Non-null rate-control implementation retained by sharing.
   * @param options Stable scheduler identity and validated decay options.
   * @throws InvalidArgument If identity, topology, rates, or options are
   * invalid.
   */
  explicit StepLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                           StepLrSchedulerOptions options = {});

  /** @brief Releases the opaque scheduler implementation. */
  ~StepLrScheduler();

  /** @brief Prevents copying shared mutable scheduling state. */
  StepLrScheduler(const StepLrScheduler&) = delete;

  /** @brief Prevents assigning shared mutable scheduling state. */
  StepLrScheduler& operator=(const StepLrScheduler&) = delete;

  /** @brief Prevents moving an object retained by checkpoint participants. */
  StepLrScheduler(StepLrScheduler&&) = delete;

  /** @brief Prevents move assignment of participant-retained state. */
  StepLrScheduler& operator=(StepLrScheduler&&) = delete;

  /**
   * @brief Advances the schedule by one completed epoch event.
   * @throws CompatibilityError If optimizer identity or group order changed.
   * @throws NumericalError If applying the decay would create a non-finite
   * rate.
   * @throws ResourceExhausted If the step counter is exhausted.
   */
  void step();

  /**
   * @brief Returns the immutable scheduler checkpoint identity.
   * @return A nonempty identifier valid for the scheduler lifetime.
   */
  const std::string& scheduler_id() const noexcept;

  /**
   * @brief Returns the number of successfully completed scheduler steps.
   * @return The exact mutable counter persisted by checkpoint state.
   */
  std::uint64_t completed_steps() const noexcept;

  /**
   * @brief Returns learning rates captured during construction.
   * @return One immutable rate per stable optimizer group.
   */
  const std::vector<double>& base_learning_rates() const noexcept;

  /**
   * @brief Returns current rates after validating optimizer topology.
   * @return One finite non-negative rate per stable optimizer group.
   * @throws CompatibilityError If optimizer identity or group order changed.
   * @throws InvalidArgument If the optimizer exposes invalid current rates.
   */
  std::vector<double> learning_rates() const;

  /**
   * @brief Returns the immutable canonical scheduler descriptor.
   * @return A descriptor valid for the scheduler lifetime.
   */
  const SchedulerDescriptor& descriptor() const noexcept;

 private:
  /** @brief Opaque owner of optimizer linkage and mutable scheduler state. */
  struct Impl;

  /** @brief Allows exact checkpoint state to commit after full preflight. */
  friend class StepLrSchedulerCheckpointParticipant;

  std::unique_ptr<Impl> impl_;  ///< Non-null scheduler implementation owner.
};

/**
 * @brief Configures StepLR checkpoint resource limits.
 */
struct StepLrSchedulerCheckpointOptions {
  std::size_t max_groups = 4096;  ///< Positive maximum optimizer group count.
  std::size_t max_identifier_bytes =
      4096;  ///< Positive maximum bytes in one stable identifier.
  std::size_t max_config_entries =
      1024;  ///< Positive maximum typed entries in one descriptor.
  std::size_t max_config_array_elements =
      1000000;  ///< Positive maximum elements in one typed array value.
  std::size_t max_descriptor_depth =
      64;  ///< Positive maximum nested composite descriptor depth.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL * 1024ULL;  ///< Maximum canonical descriptor bytes.
  std::uint64_t max_control_bytes =
      16ULL * 1024ULL * 1024ULL;  ///< Maximum mutable scheduler state bytes.
  std::uint64_t max_total_bytes =
      80ULL * 1024ULL * 1024ULL;  ///< Maximum topology plus control bytes.
};

/**
 * @brief Exposes StepLR state through the exact checkpoint contract.
 *
 * Schema version 1 stores the complete canonical scheduler descriptor in a
 * topology blob and the completed-step counter plus every current learning
 * rate in a control blob. Restore validates both blobs and all limits before
 * the optimizer or scheduler is mutated. Native archives and legacy scheduler
 * maps are not accepted.
 */
class StepLrSchedulerCheckpointParticipant final
    : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a checkpoint participant retaining a scheduler.
   * @param scheduler Non-null StepLR scheduler shared with the trainer.
   * @param options Positive bounds applied before allocation or mutation.
   * @throws InvalidArgument If scheduler, options, or state is invalid.
   */
  explicit StepLrSchedulerCheckpointParticipant(
      std::shared_ptr<StepLrScheduler> scheduler,
      StepLrSchedulerCheckpointOptions options = {});

  /**
   * @brief Returns the scheduler's stable registry identity.
   * @return An owned identifier matching the published manifest.
   */
  std::string participant_id() const override;

  /**
   * @brief Describes exact topology and control blobs.
   * @return A validated schema-version-1 scheduler manifest.
   */
  StateManifest describe_state() const override;

  /**
   * @brief Writes canonical descriptor and current StepLR control state.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws ResourceExhausted If current state exceeds a configured limit.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Validates and atomically restores exact StepLR state.
   * @param reader Source containing a schema-version-1 scheduler snapshot.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or topology differs.
   * @throws SerializationError If a required field is malformed.
   * @throws ResourceExhausted If decoded state exceeds a configured limit.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns mutable access to the retained scheduler.
   * @return A reference valid for the participant lifetime.
   */
  StepLrScheduler& scheduler() noexcept;

  /**
   * @brief Returns read-only access to the retained scheduler.
   * @return A reference valid for the participant lifetime.
   */
  const StepLrScheduler& scheduler() const noexcept;

  /**
   * @brief Returns immutable checkpoint resource limits.
   * @return A reference valid for the participant lifetime.
   */
  const StepLrSchedulerCheckpointOptions& options() const noexcept;

 private:
  std::shared_ptr<StepLrScheduler>
      scheduler_;  ///< Shared scheduler whose exact state is persisted.
  StepLrSchedulerCheckpointOptions
      options_;  ///< Immutable decode and manifest resource bounds.
};

}  // namespace jyppx::optim

#endif  // JYPPX_OPTIM_STEP_LR_SCHEDULER_HPP
