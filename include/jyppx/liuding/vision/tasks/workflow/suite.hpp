// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SUITE_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SUITE_HPP
/** @file suite.hpp @brief Native train, resume, evaluate and prediction facade.
 */
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks {
/** @brief Canonical, disjoint training/validation selections shared by all
 * workflows. */
struct VisionDataSplits {
  std::shared_ptr<data::image::ImageDataset> training, validation;
  std::string training_identity, validation_identity;
  std::vector<data::image::VisionCategory> categories;
};
/** @brief Imports external layouts and resolves pose symmetry into the recipe.
 */
VisionDataSplits load_vision_data(RecipeSpec& recipe,
                                  std::stop_token stop = {});
struct TrainRequest {
  RecipeSpec recipe;
  std::filesystem::path output;
  std::filesystem::path resume;
  std::uint64_t pause_after_updates = 0;
  std::stop_token stop;
};
struct TrainResult {
  training::FitResult fit;
  std::string run_identity;
  std::filesystem::path last_checkpoint;
  std::shared_ptr<nn::ModelAdapter> training_model;
  std::shared_ptr<nn::ModelAdapter> evaluation_model;
};
/** @brief Runs existing Trainer with fixed task/recipe/data identities.
 * last includes full optimizer/RNG/data/EMA state; best follows the declared
 * validation metric and direction.
 * A nonempty resume restores the same recipe and data, never reinitializes
 * weights. Output must be new/empty for fresh training. No global singleton
 * execution state.
 */
TrainResult train_vision(const TrainRequest& request);
/** @brief Predicts canonical samples through deterministic preprocessing and
 * decoding. Caller owns model placement. Results restore to each original
 * source coordinate. Classification returns class probabilities in
 * scores/classes and empty boxes.
 */
std::vector<ImageResult> predict_vision(
    TaskBinding& binding, std::span<const data::image::VisionSample> samples,
    const RecipeSpec& recipe, bool restore_original = true,
    std::stop_token stop = {});
struct PredictionTiming {
  double preprocess_ms = 0, inference_ms = 0, postprocess_ms = 0, warmup_ms = 0;
  std::size_t images = 0, warmup_iterations = 0;
};
struct ProfiledPrediction {
  std::vector<ImageResult> results;
  PredictionTiming timing;
};
/** @brief Synchronized wall times; warmup reuses the prepared input without
 * decoding. Warmup is capped at 100 invocations. Device placement remains
 * caller-owned.
 */
ProfiledPrediction predict_vision_profiled(
    TaskBinding& binding, std::span<const data::image::VisionSample> samples,
    const RecipeSpec& recipe, std::size_t warmup_iterations = 0,
    bool restore_original = true, std::stop_token stop = {});
/** @brief Full-split loss and authoritative task metrics; bounded native JSON
 * report. Uses mask/OKS/exact quadrilateral AP for corresponding tasks; box AP
 * for detect. Does not modify model tensors, consume training sampler/cursor or
 * use train as val.
 */
std::string evaluate_vision(TaskBinding& binding,
                            const data::image::ImageDataset& dataset,
                            const RecipeSpec& recipe,
                            std::string dataset_identity,
                            std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_SUITE_HPP
