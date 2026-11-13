// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "binding_backends.hpp"

#include <jyppx/liuding/vision/tasks/data/targets.hpp>
#include <jyppx/liuding/vision/tasks/faster_rcnn.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <algorithm>
#include <jyppx/liuding/vision/tasks/data/extended.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/training/advanced.hpp>
namespace jyppx::liuding::vision::tasks {

std::vector<Tensor> raw_tensors(const c10::IValue& value) {
  if (value.isTensor()) return {value.toTensor()};
  if (!value.isTuple())
    throw InvalidArgument(
        "Vision raw output must be tensor or a fixed tensor tuple");
  std::vector<Tensor> result;
  for (const auto& v : value.toTupleRef().elements()) {
    if (!v.isTensor())
      throw InvalidArgument("Vision raw tuple contains a non-tensor value");
    result.push_back(v.toTensor());
  }
  if (result.empty())
    throw InvalidArgument("Vision model returned an empty tuple");
  return result;
}
std::vector<CapabilityRecord> vision_catalog() {
  std::vector<CapabilityRecord> entries;
  detail::catalog_yolo(entries);
  detail::catalog_rcnn(entries);
  detail::catalog_classification(entries);
  detail::catalog_extended(entries);
  return entries;
}

CapabilityRecord describe_binding(const ModelSpec& spec) {
  if (spec.num_classes < 1 || spec.num_classes > 10000)
    throw InvalidArgument("Vision class count must be in [1,10000]");
  const auto catalog = vision_catalog();
  auto it = std::find_if(catalog.begin(), catalog.end(), [&](const auto& c) {
    return c.architecture == spec.architecture && c.task == spec.task &&
           c.variant == spec.variant;
  });
  if (it == catalog.end())
    throw InvalidArgument(
        "Model/task/variant is not present in the installed vision catalog: " +
        spec.architecture);

  return *it;
}
TaskBinding bind_task(const RecipeSpec& r) {
  check_recipe(r);
  TaskBinding b;
  b.capability = describe_binding(r.model);
  auto data = r.data;
  data.size_divisor = static_cast<std::uint32_t>(b.capability.input_divisor);
  data.num_classes = r.model.num_classes;
  b.image_mapper = make_yolo_image_mapper();
  b.target_mapper = make_yolo_target_mapper(data);
  if (r.data.letterbox.canvas.width % b.capability.input_divisor ||
      r.data.letterbox.canvas.height % b.capability.input_divisor)
    throw InvalidArgument("Recipe image size violates model input divisor");
  detail::bind_yolo(r, b, data);
  detail::bind_rcnn(r, b, data);
  if(b.capability.fixed_input_size[0] && (r.data.letterbox.canvas.height!=b.capability.fixed_input_size[0]||r.data.letterbox.canvas.width!=b.capability.fixed_input_size[1]))
    throw InvalidArgument("Model requires its catalog fixed input size");
  detail::bind_classification(r, b, data);
  detail::bind_extended(r,b,data);
  if(static_cast<unsigned>(r.model.task)>4) {
    const auto& e=r.extension;
    b.image_mapper=make_extended_image_mapper(r.model.task,data,false,e.model_patch_size,e.mask_patch_size,e.mask_ratio,r.seed);
    b.target_mapper=make_extended_target_mapper(r.model.task,r.model.num_classes,e.model_patch_size,e.mask_patch_size,e.mask_ratio,r.seed);
    if(r.model.task==TaskKind::classify_multilabel)b.step=make_multilabel_step({.asymmetric=e.asymmetric_multilabel});
    if(r.model.task==TaskKind::semantic_segment)b.step=make_semantic_step(e.semantic_ignore);
    if(r.model.task==TaskKind::depth)b.step=make_depth_step({.scale_shift_invariant=e.relative_inverse_depth,.gradient_weight=e.depth_gradient_weight});
    if(r.model.task==TaskKind::pretrain_simmim) {
      if((r.model.architecture=="simmim_swin_tiny"?4:16)!=e.model_patch_size)throw InvalidArgument("SimMIM model_patch_size differs from encoder");
      b.step=make_simmim_step(e.model_patch_size);
    }
    if(r.model.task==TaskKind::reid)b.step=make_reid_step(e.triplet_margin);
    b.decode=[task=r.model.task,thresholds=r.extension.multilabel_thresholds](const c10::IValue& v,auto size,const ResultOptions& o){return decode_extended_results(raw_tensors(v),size,task,o,thresholds);};
  }
  if (r.model.task == TaskKind::classify) {
    b.target_mapper = make_classification_target_mapper();
    b.step = make_augmented_classification_step(
        r.model.num_classes, r.data.classification, r.seed, r.label_smoothing,
        r.class_weights);
    b.decode = [](const c10::IValue& x, std::array<std::int64_t, 2> size,
                  const ResultOptions& p) {
      return decode_classification_results(raw_tensors(x).at(0), size, p);
    };
  }
  if (!b.model || !b.step || !b.decode)
    throw NotImplemented(
        "Installed model selection has no native task binding");
  return b;
}
}  // namespace jyppx::liuding::vision::tasks
