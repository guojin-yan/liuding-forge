// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_WORKFLOW_RECIPE_HPP
#define JYPPX_LIUDING_VISION_TASKS_WORKFLOW_RECIPE_HPP
/** @file recipe.hpp @brief Versioned native vision workflow recipes. */
#include <jyppx/liuding/training/diagnostics.hpp>
#include <jyppx/liuding/vision/tasks/data/dataset.hpp>
#include <jyppx/liuding/vision/tasks/data/inspection.hpp>
#include <jyppx/liuding/vision/tasks/data/orientation.hpp>
#include <jyppx/liuding/vision/tasks/data/preparation.hpp>
#include <jyppx/liuding/vision/tasks/results.hpp>
#include <jyppx/liuding/vision/tasks/training/strategy.hpp>
#include <jyppx/liuding/vision/tasks/training/transfer.hpp>

#include <filesystem>
#include <optional>
namespace jyppx::liuding::vision::tasks {
/** @brief Architecture identity independent of the installed optional packs. */
struct ModelSpec {
  std::string architecture = "yolov8n";
  TaskKind task = TaskKind::detect;
  std::string variant = "p5";
  std::int64_t num_classes = 80;
  std::int64_t keypoints = 17;
  std::int64_t keypoint_dimensions = 3;
  std::int64_t mask_channels = 32;
};
/** @brief Version-2 task policy; all defaults are serialized in run identity. */
struct ExtensionRecipe {
  std::filesystem::path training_manifest, validation_manifest;
  bool asymmetric_multilabel=false;
  std::vector<double> multilabel_thresholds;
  std::string threshold_validation_identity;
  bool activation_recomputation=false;
  std::filesystem::path tokenizer_merges,text_encoder_weights;
  std::string tokenizer_sha256,text_encoder_sha256;
  bool feature_outputs=false;
  std::int64_t feature_projection=0;
  std::string teacher_architecture;
  InitializationOptions teacher_initialization;
  double distillation_temperature=2.,logits_distillation_weight=1.,feature_distillation_weight=0.;
  std::int64_t semantic_ignore=255;
  bool relative_inverse_depth=false;
  double depth_gradient_weight=0.;
  std::int64_t model_patch_size=4, mask_patch_size=32;
  double mask_ratio=.6;
  double triplet_margin=.3;
};
/** @brief Fixed training, validation and initialization policy. */
struct RecipeSpec {
  std::uint32_t schema_version = 1;
  ModelSpec model;
  ExtensionRecipe extension;
  DetectionDatasetOptions training;
  std::optional<DetectionDatasetOptions> validation;
  std::filesystem::path classification_root;
  std::filesystem::path classification_validation_root;
  YoloDataOptions data;
  DetectionSgdOptions optimizer;
  WarmupCosineOptions scheduler;
  std::string optimizer_name = "sgd";
  Device device{torch::kCPU};
  training::PrecisionMode precision = training::PrecisionMode::full;
  std::uint64_t seed = 0;
  bool deterministic = true;  ///< Require deterministic training algorithms.
  double initial_loss_scale =
      65536.;  ///< Initial dynamic CUDA FP16 loss scale.
  std::uint64_t epochs = 100;
  std::size_t batch_size = 16;
  std::size_t workers = 0;
  std::size_t prefetch_batches = 2;
  bool pin_memory = false;
  bool rectangular_batches = false;
  DecodeCacheOptions cache;
  ExifPolicy exif_policy = ExifPolicy::reject_non_identity;
  InitializationOptions initialization;
  std::size_t accumulate_steps = 1;
  std::optional<double> gradient_clip_norm = 10.;
  double label_smoothing = 0;
  std::vector<double>
      class_weights;  ///< Fixed training-only weights; empty disables.
  std::vector<double> keypoint_sigmas;  ///< Shared pose loss/OKS scales; empty
                                        ///< selects uniform 1/K.
  bool full_validation_metrics = true;
  bool tensorboard =
      false;  ///< Optional native event output for each run segment.
  bool diagnostics =
      false;  ///< Reuse checkpointed Trainer diagnostics when enabled.
  training::TrainingDiagnosticsOptions diagnostics_options;
  bool verify_image_contents = true;  ///< Hash actual images at startup and
                                      ///< reject cross-split content leakage.
  std::string selection_metric = "auto";
  bool selection_maximize = true;
  bool ema = true;
  double ema_decay = .9999;
  double ema_ramp_updates = 2000.;
  std::uint64_t early_stopping_patience = 50;
  std::uint64_t save_interval_epochs = 1;
  ResultOptions prediction;
  ResultOptions evaluation{.confidence = .001, .nms_iou = .7};
};
/** @brief Strict bounded JSON loader; local paths resolve against recipe
 * location. Unknown fields/unsupported combinations fail before training. No
 * YAML program, subprocess, network access or implicit precision/device
 * fallback.
 */
RecipeSpec load_recipe(const std::filesystem::path& path);
/** @brief Parses bounded JSON; relative paths resolve against the explicit base
 * directory. */
RecipeSpec parse_recipe(std::string_view json,
                        const std::filesystem::path& base_directory);
/** @brief Canonical resolved JSON, including explicit defaults and absolute
 * paths. */
std::string recipe_json(const RecipeSpec& recipe);
/** @brief Validates values and task/augmentation contracts without creating a
 * model. */
void check_recipe(const RecipeSpec& recipe);
/** @brief SHA-256 of resolved recipe; data contents are bound separately by the
 * run. */
std::string recipe_identity(const RecipeSpec& recipe);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_RECIPE_HPP
