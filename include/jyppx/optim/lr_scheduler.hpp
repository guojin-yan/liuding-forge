// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_OPTIM_LR_SCHEDULER_HPP
#define JYPPX_OPTIM_LR_SCHEDULER_HPP

/**
 * @file lr_scheduler.hpp
 * @brief Defines descriptor-first learning-rate scheduling and checkpointing.
 */

#include <jyppx/foundation/checkpoint.hpp>
#include <jyppx/optim/optimizer_rate_control.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
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

class LrScheduler;

namespace detail {

/**
 * @brief Stores a scheduler node's mutable state independently of storage.
 *
 * The tree shape must exactly match the immutable descriptor tree. Learning
 * rates are stored once at the root checkpoint because every child in a
 * composite controls the same optimizer.
 */
struct SchedulerState {
  std::uint64_t completed_steps = 0;  ///< Successfully committed step events.
  std::map<std::string, SchedulerConfigValue>
      control;  ///< Ordered typed algorithm-specific mutable control values.
  std::vector<SchedulerState> children;  ///< Ordered mutable child states.
};

/**
 * @brief Holds a fully preflighted scheduler step before mutation.
 */
struct SchedulerTransition {
  SchedulerState state;  ///< Complete replacement state for the scheduler tree.
  std::vector<double>
      learning_rates;  ///< Complete replacement optimizer learning rates.
};

/**
 * @brief Provides controlled framework access to scheduler transaction hooks.
 *
 * This type is an implementation boundary used by composite schedulers and
 * the checkpoint participant. Applications should use `LrScheduler` only.
 */
class SchedulerStateAccess final {
 public:
  /**
   * @brief Previews one step without mutating scheduler or optimizer state.
   * @param scheduler Scheduler tree to evaluate.
   * @param learning_rates Valid current or intermediate group rates.
   * @return A complete candidate state and complete candidate rates.
   */
  static SchedulerTransition preview_step(
      const LrScheduler& scheduler, std::span<const double> learning_rates);

  /**
   * @brief Previews one metric event without mutating visible state.
   * @param scheduler Metric-driven scheduler tree to evaluate.
   * @param learning_rates Valid current group rates.
   * @param metric Finite monitored metric supplied by the caller.
   * @return A complete candidate state and complete candidate rates.
   */
  static SchedulerTransition preview_metric_step(
      const LrScheduler& scheduler, std::span<const double> learning_rates,
      double metric);

  /**
   * @brief Captures the complete mutable scheduler-state tree.
   * @param scheduler Scheduler tree to inspect.
   * @return An owned storage-neutral state tree.
   */
  static SchedulerState capture_state(const LrScheduler& scheduler);

  /**
   * @brief Validates a complete state tree without mutation.
   * @param scheduler Scheduler tree defining the expected shape.
   * @param state Candidate state to validate.
   */
  static void validate_state(const LrScheduler& scheduler,
                             const SchedulerState& state);

  /**
   * @brief Commits a prevalidated state tree without throwing.
   * @param scheduler Scheduler tree to update.
   * @param state Candidate already accepted by `validate_state()`.
   */
  static void apply_state(LrScheduler& scheduler,
                          const SchedulerState& state) noexcept;

  /**
   * @brief Returns the optimizer retained by a scheduler.
   * @param scheduler Scheduler whose optimizer ownership is queried.
   * @return Shared optimizer handle used to verify composite ownership.
   */
  static const std::shared_ptr<OptimizerRateControl>& optimizer(
      const LrScheduler& scheduler) noexcept;
};

}  // namespace detail

/**
 * @brief Base for descriptor-first, transaction-safe LR schedulers.
 *
 * `step()` always computes and validates a complete transition before calling
 * the optimizer's atomic all-group update once. Mutable scheduler state is
 * committed only after that update succeeds. Composite implementations reuse
 * the same path, so no child can be partially advanced.
 *
 * Instances are not internally synchronized. Callers must serialize
 * scheduler steps, optimizer steps, direct rate changes, and checkpoint I/O.
 */
class LrScheduler {
 public:
  /** @brief Enables destruction through the scheduler interface. */
  virtual ~LrScheduler();

  /** @brief Prevents copying mutable scheduler state. */
  LrScheduler(const LrScheduler&) = delete;

  /** @brief Prevents assigning mutable scheduler state. */
  LrScheduler& operator=(const LrScheduler&) = delete;

  /** @brief Prevents moving a scheduler retained by participants or parents. */
  LrScheduler(LrScheduler&&) = delete;

  /** @brief Prevents move assignment of retained scheduler state. */
  LrScheduler& operator=(LrScheduler&&) = delete;

  /**
   * @brief Atomically advances the complete scheduler tree by one event.
   * @throws CompatibilityError If optimizer identity or group order changed.
   * @throws NumericalError If the candidate rates are invalid.
   * @throws ResourceExhausted If a scheduler counter is exhausted.
   */
  void step();

  /**
   * @brief Atomically advances a metric-driven scheduler by one observation.
   * @param metric Finite monitored value, such as validation loss.
   * @throws InvalidArgument If the scheduler is not metric-driven or the
   * metric is non-finite.
   * @throws CompatibilityError If optimizer identity or group order changed.
   * @throws NumericalError If the candidate rates are invalid.
   * @throws ResourceExhausted If a scheduler counter is exhausted.
   */
  void step(double metric);

  /**
   * @brief Returns the immutable checkpoint participant identity.
   * @return A nonempty identifier valid for the scheduler lifetime.
   */
  const std::string& scheduler_id() const noexcept;

  /**
   * @brief Returns the root node's committed step count.
   * @return Number of successfully committed root events.
   */
  std::uint64_t completed_steps() const noexcept;

  /**
   * @brief Returns rates captured when the root scheduler was constructed.
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
   * @brief Returns the immutable canonical descriptor tree.
   * @return A descriptor valid for the scheduler lifetime.
   */
  const SchedulerDescriptor& descriptor() const noexcept;

 protected:
  /**
   * @brief Captures stable optimizer topology for a derived scheduler.
   * @param optimizer Non-null atomic optimizer rate-control surface.
   * @param scheduler_id Stable nonempty checkpoint participant identity.
   * @throws InvalidArgument If identity, topology, or current rates are
   * invalid.
   */
  LrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
              std::string scheduler_id);

  /**
   * @brief Finalizes the immutable descriptor during derived construction.
   * @param type_id Stable algorithm and schema identity.
   * @param step_mode Event semantics shared by the scheduler tree.
   * @param config Complete typed immutable configuration.
   * @param children Ordered child descriptors for a composite scheduler.
   * @param callable_id Optional stable registered callable identity.
   * @throws InvalidArgument If descriptor metadata is invalid.
   */
  void configure_descriptor(
      std::string type_id, SchedulerStepMode step_mode,
      std::map<std::string, SchedulerConfigValue> config,
      std::vector<SchedulerDescriptor> children = {},
      std::optional<std::string> callable_id = std::nullopt);

  /**
   * @brief Applies a validated construction-time rate vector.
   *
   * Some schedulers intentionally begin at a factor of the captured base
   * rates. Derived constructors call this after validating their immutable
   * options and configuring the descriptor; normal training transitions must
   * continue to use `step()` so the scheduler transaction remains atomic.
   *
   * @param learning_rates Complete finite non-negative initial group rates.
   * @throws InvalidArgument If the vector does not match the captured
   * topology or contains an invalid rate.
   */
  void initialize_learning_rates(std::span<const double> learning_rates);

 private:
  /** @brief Opaque owner of stable optimizer linkage and descriptor metadata.
   */
  struct Impl;

  /** @brief Allows transaction helpers to call private virtual state hooks. */
  friend class detail::SchedulerStateAccess;

  /**
   * @brief Computes one complete transition without mutation.
   * @param learning_rates Valid input rates for this scheduler node.
   * @return Fully owned candidate state and output rates.
   */
  virtual detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const = 0;

  /**
   * @brief Computes one metric-driven transition without mutation.
   *
   * The default rejects metric events. Metric-driven derived classes override
   * this hook and normally reject the metric-less hook instead.
   *
   * @param learning_rates Valid input rates for this scheduler node.
   * @param metric Finite monitored metric supplied by the caller.
   * @return Fully owned candidate state and output rates.
   */
  virtual detail::SchedulerTransition preview_metric_step_impl(
      std::span<const double> learning_rates, double metric) const;

  /**
   * @brief Captures this node and all descendant mutable state.
   * @return A fully owned state tree matching the immutable descriptor tree.
   */
  virtual detail::SchedulerState capture_state_impl() const = 0;

  /**
   * @brief Returns this scheduler node's committed event count.
   * @return Number of events committed by this exact scheduler node.
   */
  virtual std::uint64_t completed_steps_impl() const noexcept = 0;

  /**
   * @brief Validates candidate state shape and values without mutation.
   * @param state Candidate node whose descendants must match this scheduler.
   * @throws CompatibilityError If the state tree does not match the scheduler.
   * @throws InvalidArgument If a counter or child state is invalid.
   */
  virtual void validate_state_impl(
      const detail::SchedulerState& state) const = 0;

  /**
   * @brief Applies a prevalidated state tree without throwing.
   * @param state State accepted by `validate_state_impl()` for this scheduler.
   */
  virtual void apply_state_impl(
      const detail::SchedulerState& state) noexcept = 0;

  std::unique_ptr<Impl> impl_;  ///< Stable optimizer and descriptor owner.
};

/**
 * @brief Configures generic scheduler checkpoint resource limits.
 */
struct LrSchedulerCheckpointOptions {
  std::size_t max_groups = 4096;  ///< Positive maximum optimizer group count.
  std::size_t max_identifier_bytes =
      4096;  ///< Positive maximum bytes in one stable identifier.
  std::size_t max_config_entries =
      1024;  ///< Positive maximum typed entries or children per node.
  std::size_t max_config_array_elements =
      1000000;  ///< Positive maximum elements in one typed array value.
  std::size_t max_descriptor_depth =
      64;  ///< Positive maximum descriptor and state-tree depth.
  std::uint64_t max_topology_bytes =
      64ULL * 1024ULL * 1024ULL;  ///< Maximum canonical descriptor bytes.
  std::uint64_t max_control_bytes =
      16ULL * 1024ULL * 1024ULL;  ///< Maximum mutable state-tree bytes.
  std::uint64_t max_total_bytes =
      80ULL * 1024ULL * 1024ULL;  ///< Maximum topology plus control bytes.
};

/**
 * @brief Exposes any descriptor-first scheduler tree as an exact participant.
 *
 * Participant schema version 3 stores the schema-version-1 canonical
 * descriptor in `topology` and the complete mutable state tree plus current
 * rates in `control`. Each state node contains a typed algorithm control map
 * in addition to its event counter and children. Restore
 * performs all parsing, topology, resource, tree-shape, and numeric checks
 * before one atomic optimizer update and a non-throwing scheduler commit.
 */
class LrSchedulerCheckpointParticipant final : public CheckpointParticipant {
 public:
  /**
   * @brief Creates a participant retaining a scheduler tree.
   * @param scheduler Non-null root scheduler shared with the trainer.
   * @param options Positive bounds applied before allocation or mutation.
   * @throws InvalidArgument If scheduler, options, or state is invalid.
   */
  explicit LrSchedulerCheckpointParticipant(
      std::shared_ptr<LrScheduler> scheduler,
      LrSchedulerCheckpointOptions options = {});

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
   * @brief Writes canonical descriptor, state tree, and current rates.
   * @param writer Storage-neutral destination supplied by the coordinator.
   * @throws ResourceExhausted If current state exceeds a configured limit.
   */
  void save_state(StateWriter& writer) const override;

  /**
   * @brief Validates and atomically restores exact scheduler-tree state.
   * @param reader Source containing a schema-version-1 scheduler snapshot.
   * @param policy Restore policy that must accept exact state.
   * @throws CompatibilityError If manifest or topology differs.
   * @throws SerializationError If a required field is malformed.
   * @throws ResourceExhausted If decoded state exceeds a configured limit.
   */
  void load_state(const StateReader& reader, RestorePolicy policy) override;

  /**
   * @brief Returns mutable access to the retained root scheduler.
   * @return A reference valid for the participant lifetime.
   */
  LrScheduler& scheduler() noexcept;

  /**
   * @brief Returns read-only access to the retained root scheduler.
   * @return A reference valid for the participant lifetime.
   */
  const LrScheduler& scheduler() const noexcept;

  /**
   * @brief Returns immutable checkpoint resource limits.
   * @return A reference valid for the participant lifetime.
   */
  const LrSchedulerCheckpointOptions& options() const noexcept;

 private:
  std::shared_ptr<LrScheduler>
      scheduler_;  ///< Shared scheduler tree whose exact state is persisted.
  LrSchedulerCheckpointOptions
      options_;  ///< Immutable decode and manifest resource bounds.
};

}  // namespace jyppx::optim

#endif  // JYPPX_OPTIM_LR_SCHEDULER_HPP
