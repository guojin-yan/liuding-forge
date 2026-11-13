// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SEARCH_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SEARCH_HPP
/** @file search.hpp @brief Bounded native random/evolution vision recipe
 * search. */
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>
namespace jyppx::liuding::vision::tasks {
enum class SearchDistribution { uniform, log_uniform, integer, discrete };
struct SearchParameter {
  std::string field;  ///< Supported train/augmentation field, never model/data
                      ///< identity.
  SearchDistribution distribution = SearchDistribution::uniform;
  double lower = 0;
  double upper = 1;
  std::vector<double> choices;
  std::string
      condition_field;  ///< Optional earlier parameter, exact finite equality.
  double condition_value = 0;
  bool allow_enable =
      false;  ///< Explicit opt-in to search a disabled augmentation.
};
struct SearchOptions {
  std::vector<SearchParameter> space;
  std::uint64_t seed =
      0;  ///< Search seed; every trial retains the fixed recipe seed.
  std::size_t trials = 20;
  std::size_t elite_size = 5;
  bool evolution = false;
  double mutation_sigma = .2;
  std::string objective = "validation.loss";
  bool maximize = false;
};
struct SearchObservation {
  double objective = 0;
  std::filesystem::path checkpoint;
  std::string run_identity;
};
/** @brief Actual execution callback returning structured metric/artifact
 * identities. A trial gets a stable index/output and a fully checked recipe.
 * Throwing leaves a persisted pending trial for explicit resume; it is never
 * counted completed.
 */
using TrialExecutor = std::function<SearchObservation(
    const RecipeSpec&, const std::filesystem::path&, std::size_t)>;
/** @brief Optional native process launcher; it must write the normal training
 * outputs. Each request uses a new attempt directory and an explicit resume
 * checkpoint. The caller owns process cancellation, timeout, environment and
 * log bounds.
 */
using TrialTrainingExecutor = std::function<void(const TrainRequest&)>;
/** @brief Executes or recovers one trial using the shared attempt/scoring
 * policy. An empty launcher calls train_vision in process. A launcher may reuse
 * the existing native local process runner without changing search/RNG
 * semantics.
 */
SearchObservation execute_vision_trial(const RecipeSpec& recipe,
                                       const std::filesystem::path& output,
                                       const std::string& objective,
                                       TrialTrainingExecutor launcher = {},
                                       std::stop_token stop = {});
struct SearchResult {
  RecipeSpec best_recipe;
  SearchObservation best;
  std::size_t completed = 0;
};
/** @brief Runs/resumes sequential search with atomic state and deterministic
 * draws. Parent selection, pending candidates, observations and sampler
 * revision persist. Native counter draws make the sequence independent of wall
 * time or trial duration. Output contains search.json, per-trial recipes,
 * best.recipe.json and best.json.
 */
SearchResult search_vision(const RecipeSpec& baseline,
                           const SearchOptions& options,
                           const std::filesystem::path& output,
                           TrialExecutor execute = {},
                           std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SEARCH_HPP
