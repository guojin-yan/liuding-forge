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
  std::int64_t rtdetr_max_encoder_elements = 16777216;
  ///< RT-DETR encoder allocation bound; nondefault values bind run identity.
};
/** @brief Optional explicit YOLO26 detection criterion; no upstream defaults
 * are inferred from a weight filename. */
struct Yolo26LossRecipe {
  double box_gain = 7.5, class_gain = .5, dfl_gain = 1.5;
  std::int64_t one_to_one_topk = 7;
  bool fixed_branch_gains = false;
  double fixed_many_gain = 1., fixed_one_gain = 1.;
};
/** @brief Explicit per-batch YOLO26 warmup policy compatible with Ultralytics.
 */
struct Yolo26WarmupRecipe {
  bool enabled = false;
  double warmup_epochs = .98124;
  double warmup_bias_lr = .08114;
  double warmup_momentum = .6576;
  bool cosine = false;
};
/** @brief Version-2 task policy; all defaults are serialized in run identity.
 */
struct ExtensionRecipe {
  std::filesystem::path training_manifest, validation_manifest;
  bool asymmetric_multilabel = false;
  std::vector<double> multilabel_thresholds;
  std::string threshold_validation_identity;
  bool activation_recomputation = false;
  std::filesystem::path tokenizer_merges, text_encoder_weights;
  std::string tokenizer_sha256, text_encoder_sha256;
  bool feature_outputs = false;
  std::int64_t feature_projection = 0;
  std::string teacher_architecture;
  InitializationOptions teacher_initialization;
  double distillation_temperature = 2., logits_distillation_weight = 1.,
         feature_distillation_weight = 0.;
  std::int64_t semantic_ignore = 255;
  bool relative_inverse_depth = false;
  double depth_gradient_weight = 0.;
  std::int64_t model_patch_size = 4, mask_patch_size = 32;
  double mask_ratio = .6;
  double triplet_margin = .3;
};
/** @brief Fixed training, validation and initialization policy. */
struct RecipeSpec {
  std::uint32_t schema_version = 1;
  ModelSpec model;
  std::optional<Yolo26LossRecipe> yolo26_loss;
  Yolo26WarmupRecipe yolo26_warmup;
  std::vector<std::int64_t> rtdetr_training_sizes;
  ///< Explicit square model-input sizes, sampled with replacement per batch.
  ExtensionRecipe extension;
  DetectionDatasetOptions training;
  std::optional<DetectionDatasetOptions> validation;
  std::filesystem::path classification_root;
  std::filesystem::path classification_validation_root;
  YoloDataOptions data;
  DetectionSgdOptions optimizer;
  WarmupCosineOptions scheduler;
  std::string optimizer_name = "sgd";
  double musgd_muon_weight = .5, musgd_sgd_weight = .5;
  Device device{torch::kCPU};
  training::PrecisionMode precision = training::PrecisionMode::full;
  std::uint64_t seed = 0;
  bool deterministic = true;  ///< Require deterministic training algorithms.
  double initial_loss_scale =
      65536.;  ///< Initial dynamic CUDA FP16 loss scale.
  std::uint64_t epochs = 100;
  std::size_t batch_size = 16;
  bool drop_last_training_batch =
      false;  ///< Drop only the incomplete training batch.
  std::size_t workers = 0;
  std::size_t prefetch_batches = 2;
  bool pin_memory = false;
  bool non_blocking_transfer =
      false;  ///< Caller-stream device transfer policy.
  bool rectangular_batches = false;
  DecodeCacheOptions cache;
  ExifPolicy exif_policy = ExifPolicy::reject_non_identity;
  InitializationOptions initialization;
  std::size_t accumulate_steps = 1;
  training::AccumulationTailPolicy accumulation_tail =
      training::AccumulationTailPolicy::flush;
  training::AccumulationReduction accumulation_reduction =
      training::AccumulationReduction::mean;
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
/** @brief SHA-256 of resolved recipe and applicable training-numerics revision.
 * YOLO26 OBB binds FP32 v4; its other object tasks bind E2E FP32 v3.
 * Other modern YOLO detection,
 * segmentation, pose and OBB bind FP32 criterion v2;
 * other recipes retain the canonical JSON hash. Data contents and architecture
 * revision are bound separately by the run. */
std::string recipe_identity(const RecipeSpec& recipe);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_WORKFLOW_RECIPE_HPP
