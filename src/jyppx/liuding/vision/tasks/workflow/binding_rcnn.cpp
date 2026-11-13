// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "binding_backends.hpp"

#include <jyppx/liuding/vision/models/rcnn/faster_rcnn.hpp>
#include <jyppx/liuding/vision/tasks/data/targets.hpp>
#include <jyppx/liuding/vision/tasks/faster_rcnn.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks {
namespace detail {
void catalog_rcnn(std::vector<CapabilityRecord>& entries) {
  entries.push_back({.architecture = "faster_rcnn_resnet50_fpn",
                     .revision = "native-resnet50-fpn-rpn-roialign/v1",
                     .variant = "p5",
                     .component = "models_rcnn"});
}
void bind_rcnn(const RecipeSpec& r, TaskBinding& b,
               const YoloDataOptions& data) {
  if (b.capability.component == "models_rcnn") {
    models::rcnn::FasterRcnnOptions o;
    o.num_classes = r.model.num_classes;
    b.model = std::make_shared<nn::ModelAdapter>(
        models::rcnn::create_faster_rcnn_resnet50_fpn(o));
    b.image_mapper = make_rcnn_image_mapper(data);
    b.step = make_faster_rcnn_step(
        {.num_classes = r.model.num_classes, .class_weights = r.class_weights});
    b.decode = [](const c10::IValue& x, std::array<std::int64_t, 2>,
                  const ResultOptions& p) {
      return decode_faster_rcnn_results(raw_tensors(x), p);
    };
  }
}
}  // namespace detail
}  // namespace jyppx::liuding::vision::tasks
