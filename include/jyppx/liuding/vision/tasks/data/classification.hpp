// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_CLASSIFICATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_CLASSIFICATION_HPP
/** @file classification.hpp @brief Classification geometry, color and
 * soft-target mixing. */
#include <jyppx/liuding/training/trainer.hpp>
#include <jyppx/liuding/vision/tasks/data/augmentation.hpp>
namespace jyppx::liuding::vision::tasks {
struct ClassificationAugmentationOptions {
  bool enabled = false;
  bool training = true;
  double crop_min_area = .08;
  double crop_max_area = 1;
  double crop_min_aspect = .75;
  double crop_max_aspect = 4. / 3;
  double validation_crop_fraction = .875;
  double brightness = 0;
  double contrast = 0;
  double saturation = 0;
  std::uint32_t randaugment_operations = 0;  ///< Bounded native set, [0,8].
  double randaugment_magnitude = .5;         ///< [0,1].
  double erasing_probability = 0;
  double mixup_probability = 0;
  double cutmix_probability = 0;
};
data::image::VisionSample prepare_classification_sample(
    const data::image::VisionSample& source, data::image::ImageExtent output,
    const ClassificationAugmentationOptions& options,
    const data::image::SampleRandom& random, std::stop_token stop = {});
std::string classification_augmentation_identity(
    const ClassificationAugmentationOptions& options);
/** @brief Mixes unit NCHW inputs and soft class distributions inside task step.
 * MixUp uses [.35,.65], CutMix uses actual clipped pixel area. Fixed logical
 * batch identity derives partners, coefficients and rectangles without Torch
 * RNG.
 */
training::StepFunction make_augmented_classification_step(
    std::int64_t classes, const ClassificationAugmentationOptions& options,
    std::uint64_t seed = 0, double label_smoothing = 0,
    const std::vector<double>& class_weights = {});
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_CLASSIFICATION_HPP
