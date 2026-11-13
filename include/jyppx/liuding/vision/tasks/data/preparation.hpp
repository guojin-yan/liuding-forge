// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_PREPARATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_PREPARATION_HPP
/** @file preparation.hpp
 * @brief Vision task composition over existing image geometry and batches.
 */

#include <jyppx/liuding/core/types.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/training/trainer.hpp>
#include <jyppx/liuding/vision/tasks/data/augmentation.hpp>
#include <jyppx/liuding/vision/tasks/data/classification.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>
#include <string>

namespace jyppx::liuding::vision::tasks {

/** @brief Stateless sample augmentation; zero probabilities select evaluation.
 */
struct YoloAugmentationOptions {
  double horizontal_flip_probability = 0;  ///< Explicit probability in [0,1].
  double vertical_flip_probability = 0;    ///< Explicit probability in [0,1].
  ScaleTranslateOptions spatial;  ///< Joint fixed-canvas scale and translation.
  PhotometricOptions photo;  ///< Input RGB photo augmentation after geometry.
  ProjectiveOptions projective;
  MultiSourceOptions multi_source;
  std::uint64_t seed =
      0;  ///< Base experiment seed, independent of worker order.
  std::uint64_t epoch = 0;  ///< Logical epoch; set by the pipeline factory.
  std::string dataset_id;   ///< Required when augmentation is enabled.
};

/** @brief Deterministic resize/pad policy in explicitly encoded pixel units. */
struct LetterBoxOptions {
  data::image::ImageExtent canvas{640, 640};  ///< Exact output {width,height}.
  bool scale_up = true;  ///< Permit enlarging small input images.
  bool center = true;  ///< Split padding across both sides, extra pixel at end.
  double padding_value = 114;  ///< Fill in input PixelSpec's value encoding.
  data::image::JointTransformOptions geometry{
      .interpolation = data::image::ImageInterpolation::
          bilinear};  ///< Photo interpolation and separate target policies.
};

/** @brief Explicit task policies for assembling annotated RGB detection data.
 */
struct YoloDataOptions {
  bool rectangular =
      false;  ///< Per-batch minimal stride-aligned LetterBox canvas.
  bool multiscale = false;  ///< Counter-derived fixed canvas per logical batch.
  double multiscale_min = .5, multiscale_max = 1.5;
  std::uint32_t size_divisor = 32, multiscale_max_dimension = 2048;
  std::vector<double>
      normalization_mean;  ///< Empty disables channel normalization.
  std::vector<double> normalization_std;  ///< Three RGB standard deviations.
  ClassificationAugmentationOptions classification;
  LetterBoxOptions
      letterbox;  ///< Fixed canvas; default fill is uint8 value 114.
  YoloAugmentationOptions
      augmentation;                 ///< Shared geometry for pixels and targets.
  std::int64_t num_classes = 80;    ///< Class IDs must lie in [0,num_classes).
  std::int64_t max_targets = 4096;  ///< Positive aggregate row bound.
  bool allow_unannotated_negatives =
      false;                  ///< Missing annotations need opt-in.
  bool exclude_crowd = true;  ///< Drop crowd boxes from training target rows.
  bool exclude_difficult = false;  ///< Explicit VOC difficult training policy.
  data::image::ImageBatchLimits limits;  ///< Existing collation/numeric limits.
};

/** @brief Resizes and pads pixels and rich targets using the existing owner.
 * @param sample CPU image with explicit PixelSpec, extents and annotations.
 * @param options Exact canvas, fill units and shared transform constraints.
 * @param stop Cooperative cancellation forwarded to image transformation.
 * @return Owning sample with annotations and GeometryTrace composed together.
 * @throws InvalidArgument Extents, pixels or geometry policies are invalid.
 * @throws ResourceExhausted Existing geometry resource limits are exceeded.
 * @throws Cancelled Cancellation is requested.
 * @note Does not normalize colors, mutate inputs or consume random state.
 */
data::image::VisionSample letterbox_sample(
    const data::image::VisionSample& sample,
    const LetterBoxOptions& options = {}, std::stop_token stop = {});

/** @brief Composes deterministic LetterBox, stack and existing unit conversion.
 * @param samples Nonempty CPU uint8 RGB samples; explicit annotations required
 * unless negative-sample opt-in is set. Each input must have known orientation.
 * @param keys One existing stable ImageBatchKey per sample.
 * @param options Class/negative/crowd/geometry and existing batch limits.
 * @param stop Cooperative cancellation between samples and existing stages.
 * @return Existing ImageBatch with RGB float32 NCHW [0,1] encoding, annotations
 * and source-coordinate traces retained. No alternate image/batch type exists.
 * @throws InvalidArgument Pixel, annotation, canvas or identity is invalid.
 * @throws ResourceExhausted Existing geometry/batch limits are exceeded.
 * @throws Cancelled Cancellation is requested.
 */
data::image::ImageBatch prepare_yolo_batch(
    std::span<const data::image::VisionSample> samples,
    std::span<const data::image::ImageBatchKey> keys,
    const YoloDataOptions& options = {}, std::stop_token stop = {});

/** @brief Packs existing host box annotations into the detection target tensor.
 * @param batch Non-ragged, structurally valid ImageBatch; images need no copy.
 * @param options Class/negative/crowd rules and aggregate target-row bound.
 * @return [T,6] float32 on batch.images.device(); batch/class/x1/y1/x2/y2.
 * @throws InvalidArgument Annotation/class/negative policy is violated.
 * @throws ResourceExhausted Target rows exceed max_targets before allocation.
 * @note Explicit ignore regions and optionally crowd are omitted. This is a
 * training projection, not an evaluation-ground-truth conversion.
 */
Tensor build_yolo_targets(const data::image::ImageBatch& batch,
                          const YoloDataOptions& options = {});

/** @brief Maps an existing ImageBatch value to one model image invocation.
 * @return Existing Trainer mapper requiring packed images, preserving device.
 */
training::BatchMapper make_yolo_image_mapper();

/** @brief Maps existing ImageBatch annotations to the typed target tensor.
 * @param options Class, missing-annotation and target-row projection policies.
 * @return Existing Trainer target mapper retaining an immutable options copy.
 */
training::TargetMapper make_yolo_target_mapper(
    const YoloDataOptions& options = {});

/** @brief Binds preparation semantics/limits to a stable pipeline factory ID.
 * @return Versioned SHA-256 identity; epoch and runtime budget owner are
 * excluded.
 */
std::string yolo_data_identity(const YoloDataOptions& options = {});

/** @brief Disables every random augmentation while preserving pixel/resource
 * policy.
 * @param training Explicit training data recipe.
 * @return Deterministic data options for evaluation/prediction; retains canvas,
 * fill, target rules, filtering and logical/aggregate resource limits.
 */
YoloDataOptions yolo_evaluation_options(const YoloDataOptions& training);

/** @brief Connects an existing canonical dataset to the existing worker
 * pipeline.
 * @param dataset Shared immutable source, with explicit object annotations.
 * @param pipeline Caller-selected batching, worker, transfer and lineage
 * policy. Dataset size and factory identity are bound by this function;
 * dataset_version must identify the source. Missing augmentation dataset_id
 * uses that version.
 * @param options LetterBox, augmentation and target projection configuration.
 * @param sampler Optional caller sampler; omission uses sequential ordering.
 * @return Pipeline emitting the standard ImageBatch value, with lazy decoding.
 * @throws InvalidArgument Inconsistent source size, configuration or lineage.
 */
std::shared_ptr<data::DataPipeline> make_yolo_data_pipeline(
    std::shared_ptr<const data::image::ImageDataset> dataset,
    data::DataPipelineOptions pipeline, YoloDataOptions options = {},
    std::shared_ptr<data::samplers::Sampler> sampler = {});

}  // namespace jyppx::liuding::vision::tasks

#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_PREPARATION_HPP
