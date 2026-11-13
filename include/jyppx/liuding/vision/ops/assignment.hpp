// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_OPS_ASSIGNMENT_HPP
#define JYPPX_LIUDING_VISION_OPS_ASSIGNMENT_HPP
/** @file assignment.hpp @brief Bounded native rectangular minimum-cost matching. */
#include <jyppx/liuding/core/types.hpp>
#include <cstdint>
#include <optional>
#include <stop_token>

namespace jyppx::liuding::vision_ops {
/** @brief CPU working limits, checked before copying or padding a cost matrix. */
struct AssignmentOptions {
  std::int64_t max_pairs = 1048576;
  std::uint64_t max_work = 100000000;
  /** @brief Optional rejection threshold. Unmatched rows/columns each cost
   * half this value; costs above it are forbidden, rather than matched then
   * discarded. This gives a genuine partial minimum-cost assignment. */
  std::optional<double> cost_limit;
};
/** @brief Matched original row/column indices on CPU, sorted by row. */
struct Assignment {
  Tensor rows;
  Tensor columns;
};
/** @brief Minimizes finite [N,M] costs without Python/SciPy.
 * Copies a detached float64 CPU snapshot. Empty axes are valid. Equal costs
 * use deterministic increasing-index tie breaking. No input mutation or
 * gradients; cancellation and explicit work/memory limits are enforced.
 */
Assignment linear_assignment(const Tensor& costs,
                             const AssignmentOptions& options = {},
                             std::stop_token stop = {});
}  // namespace jyppx::liuding::vision_ops
#endif
