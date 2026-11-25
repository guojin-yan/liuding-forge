// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_YOLO_VALIDATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_YOLO_VALIDATION_HPP
#include "coco_box_validation.hpp"

#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks::detail {
class YoloDetectionValidation final : public CocoBoxValidation {
 public:
  YoloDetectionValidation(
      std::shared_ptr<const data::image::AnnotationIndex> targets,
      std::string dataset_identity, const RecipeSpec& recipe)
      : CocoBoxValidation(std::move(targets), std::move(dataset_identity),
                          recipe, "yolo.batched_loss_and_inference/v1") {}
  training::StepOutput step(TaskBinding& binding,
                            const training::StepInput& input);
};
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
