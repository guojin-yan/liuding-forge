// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_PRIVATE_BINDING_BACKENDS_HPP
#define JYPPX_LIUDING_VISION_TASKS_PRIVATE_BINDING_BACKENDS_HPP
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks::detail {
void catalog_extended(std::vector<CapabilityRecord>&);
void bind_extended(const RecipeSpec&,TaskBinding&,const YoloDataOptions&);
void catalog_yolo(std::vector<CapabilityRecord>&);
void bind_yolo(const RecipeSpec&, TaskBinding&, const YoloDataOptions&);
void catalog_rcnn(std::vector<CapabilityRecord>&);
void bind_rcnn(const RecipeSpec&, TaskBinding&, const YoloDataOptions&);
void catalog_classification(std::vector<CapabilityRecord>&);
void bind_classification(const RecipeSpec&, TaskBinding&,
                         const YoloDataOptions&);
}  // namespace jyppx::liuding::vision::tasks::detail
#endif
