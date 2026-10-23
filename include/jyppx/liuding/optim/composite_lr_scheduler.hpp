// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_OPTIM_COMPOSITE_LR_SCHEDULER_HPP
#define JYPPX_LIUDING_OPTIM_COMPOSITE_LR_SCHEDULER_HPP
/**
 * @file composite_lr_scheduler.hpp
 * @brief Defines transactional sequential and chained LR schedulers.
 */

#include <jyppx/liuding/optim/lr_scheduler.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::optim {

/**
 * @brief Configures a sequential scheduler tree.
 *
 * Each milestone is the total number of successful parent events assigned to
 * all preceding children. For example, milestone `2` assigns events one and
 * two to child zero and begins child one at event three.
 */
struct SequentialLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.sequential";  ///< Stable checkpoint participant identity.
  std::vector<std::int64_t>
      milestones;  ///< Positive strictly increasing child boundaries.
};

/**
 * @brief Runs one child at a time according to stable milestones.
 *
 * Children are uniquely owned, ordered configuration. They must share the
 * exact parent optimizer object and step mode. A complete child transition is
 * preflighted before the root performs one atomic optimizer update.
 */
class SequentialLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a sequential scheduler with unique child ownership.
   * @param optimizer Non-null optimizer shared by the parent and every child.
   * @param children Nonempty ordered child scheduler list.
   * @param options Stable identity and one fewer milestone than children.
   * @throws InvalidArgument If structure, ownership, modes, or milestones are
   * invalid.
   */
  SequentialLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                        std::vector<std::unique_ptr<LrScheduler>> children,
                        SequentialLrSchedulerOptions options);

  /** @brief Releases the uniquely owned scheduler tree. */
  ~SequentialLrScheduler() override;

  /**
   * @brief Returns the number of ordered child schedulers.
   * @return Positive child count fixed at construction.
   */
  std::size_t child_count() const noexcept;

  /**
   * @brief Returns the child selected for the next parent event.
   * @return Zero-based index within `child_count()`.
   */
  std::size_t next_child_index() const noexcept;

 private:
  /**
   * @brief Previews one active-child transition without mutation.
   * @param learning_rates Valid current or intermediate optimizer group rates.
   * @return Candidate parent/child state tree and active-child output rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures parent and all ordered child state.
   * @return Fully owned state tree in immutable child order.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed parent event count.
   * @return Number of successfully committed sequential events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates parent and every child state before commit.
   * @param state Candidate tree whose ordered children must match this tree.
   * @throws CompatibilityError If child topology or state shape differs.
   * @throws InvalidArgument If a counter or child state is invalid.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits parent and child state without throwing.
   * @param state Tree accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  std::vector<std::unique_ptr<LrScheduler>>
      children_;  ///< Uniquely owned ordered child scheduler tree.
  std::vector<std::int64_t>
      milestones_;  ///< Immutable positive child event boundaries.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed root events.
};

/**
 * @brief Configures a chained scheduler tree.
 */
struct ChainedLrSchedulerOptions {
  std::string scheduler_id =
      "scheduler.chained";  ///< Stable checkpoint participant identity.
};

/**
 * @brief Applies every child in order for each scheduler event.
 *
 * Each child previews against the preceding child's candidate rates. Only the
 * final rates are written to the optimizer, once, after every child succeeds.
 */
class ChainedLrScheduler final : public LrScheduler {
 public:
  /**
   * @brief Creates a chained scheduler with unique child ownership.
   * @param optimizer Non-null optimizer shared by the parent and every child.
   * @param children Nonempty ordered child scheduler list.
   * @param options Stable parent checkpoint identity.
   * @throws InvalidArgument If structure, ownership, or modes are invalid.
   */
  ChainedLrScheduler(std::shared_ptr<OptimizerRateControl> optimizer,
                     std::vector<std::unique_ptr<LrScheduler>> children,
                     ChainedLrSchedulerOptions options = {});

  /** @brief Releases the uniquely owned scheduler tree. */
  ~ChainedLrScheduler() override;

  /**
   * @brief Returns the number of ordered child schedulers.
   * @return Positive child count fixed at construction.
   */
  std::size_t child_count() const noexcept;

 private:
  /**
   * @brief Previews all child transitions in order without mutation.
   * @param learning_rates Valid current or intermediate optimizer group rates.
   * @return Candidate parent/child state tree and final chained output rates.
   */
  detail::SchedulerTransition preview_step_impl(
      std::span<const double> learning_rates) const override;

  /**
   * @brief Captures parent and all ordered child state.
   * @return Fully owned state tree in immutable execution order.
   */
  detail::SchedulerState capture_state_impl() const override;

  /**
   * @brief Returns the committed parent event count.
   * @return Number of successfully committed chained events.
   */
  std::uint64_t completed_steps_impl() const noexcept override;

  /**
   * @brief Validates parent and every child state before commit.
   * @param state Candidate tree whose ordered children must match this tree.
   * @throws CompatibilityError If child topology or state shape differs.
   * @throws InvalidArgument If a counter or child state is invalid.
   */
  void validate_state_impl(const detail::SchedulerState& state) const override;

  /**
   * @brief Commits parent and child state without throwing.
   * @param state Tree accepted by `validate_state_impl()`.
   */
  void apply_state_impl(const detail::SchedulerState& state) noexcept override;

  std::vector<std::unique_ptr<LrScheduler>>
      children_;  ///< Uniquely owned ordered child scheduler tree.
  std::uint64_t completed_steps_ = 0;  ///< Successfully committed root events.
};

}  // namespace jyppx::liuding::optim

#endif  // JYPPX_LIUDING_OPTIM_COMPOSITE_LR_SCHEDULER_HPP