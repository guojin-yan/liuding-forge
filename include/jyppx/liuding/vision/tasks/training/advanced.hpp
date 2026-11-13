// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_TRAINING_ADVANCED_HPP
#define JYPPX_LIUDING_VISION_TASKS_TRAINING_ADVANCED_HPP
/** @file advanced.hpp @brief Native multilabel, distillation and pretraining criteria. */
#include <jyppx/liuding/vision/tasks/task.hpp>
#include <array>
#include <optional>
namespace jyppx::liuding::vision::tasks {
/** @brief Stable logits BCE or asymmetric focusing, unknown labels masked. */
struct MultilabelLossOptions {
  bool asymmetric = false;
  double gamma_positive = 0., gamma_negative = 4., negative_clip = .05;
  Tensor positive_weights;  ///< Optional train-only [C], nonnegative finite.
};
/** @brief Mean over known labels; rejects zero supervision and nonfinite data.
 * ASL detaches focal weights, clips only negative probabilities, and never
 * hides NaN/Inf by substituting a zero loss. */
Tensor multilabel_loss(const Tensor& logits, const Tensor& labels,
                        const Tensor& known,
                        const MultilabelLossOptions& options = {});
/** @brief Temperature-scaled KL for matching single-label class spaces. */
Tensor logits_distillation_loss(const Tensor& student, const Tensor& teacher,
                                double temperature = 4.);
/** @brief Bernoulli KL for multilabel logits, distinct from softmax KD. */
Tensor bernoulli_distillation_loss(const Tensor& student, const Tensor& teacher,
                                   const Tensor& known, double temperature = 1.);
/** @brief Explicit same-shape feature L2; optional channel normalization. */
Tensor feature_distillation_loss(const Tensor& student, const Tensor& teacher,
                                 bool normalize_channels = true);
/** @brief Ignore-aware semantic CE with optional train-only class weights. */
Tensor semantic_segmentation_loss(const Tensor& logits, const Tensor& labels,
                                  std::int64_t ignore_index = 255,
                                  const Tensor& class_weights = {});
/** @brief Depth log-L1 or scale/shift-aligned inverse-depth L1. */
struct DepthLossOptions {
  bool scale_shift_invariant = false;
  double gradient_weight = 0.;
  double epsilon = 1e-6;
};
/** @brief Dense [B,1,H,W] predictions/targets and bool valid mask.
 * Invalid labels are excluded before arithmetic. Metric mode requires positive
 * valid values; relative mode solves per-image scale/shift with a guarded
 * determinant. Empty supervision is rejected. */
Tensor depth_loss(const Tensor& prediction, const Tensor& target,
                  const Tensor& valid, const DepthLossOptions& options = {});
/** @brief Deterministic block masking on the patch-token grid. */
Tensor simmim_mask(std::array<std::int64_t, 2> image_size,
                   std::int64_t model_patch_size, std::int64_t mask_patch_size,
                   double ratio, std::uint64_t seed);
/** @brief Normalized pixel L1 only where [B,Hpatch,Wpatch] mask is true. */
Tensor simmim_reconstruction_loss(const Tensor& reconstructed,
                                  const Tensor& target, const Tensor& patch_mask,
                                  std::int64_t model_patch_size);
/** @brief Identity CE and batch-hard triplet; excludes anchors lacking a
 * positive or negative and rejects a completely invalid triplet batch. */
Tensor batch_hard_triplet_loss(const Tensor& embeddings, const Tensor& identities,
                                double margin = .3, bool normalize = true);
/** @brief Existing Trainer step with targets=(labels,known). */
training::StepFunction make_multilabel_step(MultilabelLossOptions options = {});
/** @brief Dense target=(labels) or (depth,valid); auxiliary logits are explicit. */
training::StepFunction make_semantic_step(std::int64_t ignore_index = 255,
                                          double auxiliary_weight = .4);
training::StepFunction make_depth_step(DepthLossOptions options = {});
/** @brief Reconstruction output tensor and targets=(image,patch_mask). */
training::StepFunction make_simmim_step(std::int64_t patch_size = 4);
}  // namespace jyppx::liuding::vision::tasks
#endif
