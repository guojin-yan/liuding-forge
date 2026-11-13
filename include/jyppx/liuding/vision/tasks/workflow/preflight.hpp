// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_PREFLIGHT_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_PREFLIGHT_HPP
/** @file preflight.hpp @brief Disposable native AMP/optimizer/batch probes. */
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks {
struct PreflightOptions {
  std::vector<std::size_t> batch_candidates{1, 2, 4, 8, 16, 32};
  std::size_t maximum_probes = 8;
  std::uint64_t max_tensor_bytes = 2ULL << 30U;
  double cuda_memory_fraction = .8;
  double amp_absolute_tolerance = .05;
  double amp_relative_tolerance = .1;
};
/** @brief Exercises supplied real samples on fresh, disposable model states.
 * Each probe computes an actual task loss, backward and optimizer update. AMP
 * compares fixed eval outputs against FP32 before training. Global RNG is
 * restored explicitly on success/failure; no live model, optimizer or data
 * cursor is used. CUDA memory readings are advisory live-device deltas, not a
 * peak-memory or future-fit guarantee. Candidate draws and resolved optimizer
 * persist in JSON.
 */
std::string preflight_vision(const RecipeSpec& recipe,
                             std::span<const data::image::VisionSample> samples,
                             const PreflightOptions& options = {},
                             std::stop_token stop = {});
/** @brief Resolves the versioned optimizer policy before run/checkpoint
 * identity. */
std::string resolve_vision_optimizer(const RecipeSpec& recipe);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_PREFLIGHT_HPP
