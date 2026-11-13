// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "binding_backends.hpp"
namespace jyppx::liuding::vision::tasks::detail {
#if !defined(JYPPX_VISION_WITH_EXTENDED)
void catalog_extended(std::vector<CapabilityRecord>&) {}
void bind_extended(const RecipeSpec&,TaskBinding&,const YoloDataOptions&) {}
#endif
#if !defined(JYPPX_VISION_WITH_YOLO)
void catalog_yolo(std::vector<CapabilityRecord>&) {}
void bind_yolo(const RecipeSpec&, TaskBinding&, const YoloDataOptions&) {}
#endif
#if !defined(JYPPX_VISION_WITH_RCNN)
void catalog_rcnn(std::vector<CapabilityRecord>&) {}
void bind_rcnn(const RecipeSpec&, TaskBinding&, const YoloDataOptions&) {}
#endif
#if !defined(JYPPX_VISION_WITH_MODELS)
void catalog_classification(std::vector<CapabilityRecord>&) {}
void bind_classification(const RecipeSpec&, TaskBinding&,
                         const YoloDataOptions&) {}
#endif
}  // namespace jyppx::liuding::vision::tasks::detail
