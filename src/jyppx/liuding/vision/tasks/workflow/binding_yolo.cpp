// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "binding_backends.hpp"

#include <jyppx/liuding/vision/models/yolo/model.hpp>
#include <jyppx/liuding/vision/models/yolo/yolov5.hpp>
#include <jyppx/liuding/vision/tasks/data/targets.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>
namespace jyppx::liuding::vision::tasks {
namespace {
namespace y = models::yolo;
constexpr std::array<std::string_view, 4> families{"yolov8", "yolo11", "yolo12",
                                                   "yolo26"};
constexpr std::array<std::string_view, 6> variants{
    "p5", "p2", "p6", "ghost_p5", "ghost_p2", "ghost_p6"};
y::ModelOptions Options(const ModelSpec& s) {
  y::ModelOptions o;
  bool found = false;
  for (std::size_t i = 0; i < families.size(); ++i)
    if (s.architecture.size() == families[i].size() + 1 &&
        s.architecture.starts_with(families[i])) {
      auto index = std::string_view("nsmlx").find(s.architecture.back());
      if (index == std::string_view::npos) break;
      o.family = static_cast<y::Family>(i);
      o.capacity = static_cast<y::Capacity>(index);
      found = true;
      break;
    }
  if (!found)
    throw InvalidArgument("Unknown native YOLO architecture: " +
                          s.architecture);
  auto v = std::find(variants.begin(), variants.end(), s.variant);
  if (v == variants.end()) throw InvalidArgument("Unknown YOLO variant");
  o.variant = static_cast<y::Variant>(v - variants.begin());
  o.task = static_cast<y::Task>(s.task);
  o.num_classes = s.num_classes;
  o.keypoints = s.keypoints;
  o.keypoint_dimensions = s.keypoint_dimensions;
  o.mask_channels = s.mask_channels;
  return o;
}
std::string Name(const y::ModelOptions& o) {
  return std::string(families.at(static_cast<std::size_t>(o.family))) +
         "nsmlx"[static_cast<std::size_t>(o.capacity)];
}
bool V5(const ModelSpec& s) {
  return s.architecture.size() == 7 && s.architecture.starts_with("yolov5") &&
         std::string_view("nsmlx").find(s.architecture.back()) !=
             std::string_view::npos;
}
y::OriginalV5Options V5Options(const ModelSpec& s) {
  if (s.task != TaskKind::detect || (s.variant != "p5" && s.variant != "p6"))
    throw InvalidArgument("Original YOLOv5 binding requires detect and p5/p6");
  y::OriginalV5Options o;
  o.capacity = static_cast<y::Capacity>(
      std::string_view("nsmlx").find(s.architecture.back()));
  o.num_classes = s.num_classes;
  o.p6 = s.variant == "p6";
  return o;
}
}  // namespace
namespace detail {
void catalog_yolo(std::vector<CapabilityRecord>& entries) {
  for (const auto& o : y::model_catalog()) {
    auto d = y::describe_model(o);
    entries.push_back({.architecture = Name(o),
                       .revision = d.revision,
                       .task = static_cast<TaskKind>(o.task),
                       .variant = std::string(
                           variants.at(static_cast<std::size_t>(o.variant))),
                       .component = "models_yolo",
                       .input_divisor = d.input_divisor,
                       .end_to_end = d.end_to_end});
  }
  for (char capacity : std::string_view("nsmlx"))
    for (auto variant : {"p5", "p6"})
      entries.push_back(
          {.architecture = std::string("yolov5") + capacity,
           .revision = "original-yolov5/"
                       "915bbf294bb74c859f0b41f1c23bc395014ea679/native-v1",
           .variant = variant,
           .component = "models_yolo",
           .input_divisor = std::string_view(variant) == "p6" ? 64 : 32});
}
void bind_yolo(const RecipeSpec& r, TaskBinding& b,
               const YoloDataOptions& data) {
  if (b.capability.component == "models_yolo") {
    if (V5(r.model)) {
      auto o = V5Options(r.model);
      b.model = std::make_shared<nn::ModelAdapter>(y::create_original_v5(o));
      OriginalV5TaskOptions loss;
      loss.num_classes = r.model.num_classes;
      loss.anchors = y::original_v5_anchors(o.p6);
      loss.label_smoothing = r.label_smoothing;
      loss.class_weights = r.class_weights;
      if (o.p6) {
        loss.strides = {8, 16, 32, 64};
        loss.objectness_balance = {4., 1., .25, .06};
      }
      b.step = make_original_v5_step(loss);
      b.decode = [loss](const c10::IValue& x, std::array<std::int64_t, 2> size,
                        const ResultOptions& p) {
        return decode_detection_results(
            decode_original_v5(x.toTensor(), size, loss), size, p);
      };
    } else {
      auto o = Options(r.model);
      auto d = y::describe_model(o);
      b.model = std::make_shared<nn::ModelAdapter>(y::create_model(o));
      b.yolo.detection.class_weights = r.class_weights;
      b.yolo.task = r.model.task;
      b.yolo.detection.num_classes = r.model.num_classes;
      b.yolo.detection.reg_max = d.reg_max;
      b.yolo.detection.strides = d.strides;
      b.yolo.end_to_end = d.end_to_end;
      b.yolo.generation26 = o.family == y::Family::yolo26;
      b.yolo.mask_channels = o.mask_channels;
      b.yolo.keypoints = o.keypoints;
      b.yolo.keypoint_dimensions = o.keypoint_dimensions;
      b.yolo.schedule_epochs = r.epochs;
      b.yolo.keypoint_sigmas = r.keypoint_sigmas;
      if (r.model.task != TaskKind::classify) {
        PoseLogProbability flow;
        if (b.yolo.generation26 && r.model.task == TaskKind::pose) {
          auto module = std::dynamic_pointer_cast<y::ModelImpl>(
              b.model->state().shared_module());
          flow = [module](const Tensor& input) {
            return module->pose_log_probability(input);
          };
        }
        b.step = make_yolo_task_step(b.yolo, flow);
        b.target_mapper = make_task_target_mapper(b.yolo, data);
        b.decode = [task = b.yolo](const c10::IValue& x,
                                   std::array<std::int64_t, 2> size,
                                   const ResultOptions& p) {
          return decode_yolo_results(raw_tensors(x), size, task, p);
        };
      }
    }
    b.fuse_inference = [model = b.model]() {
      return y::fuse_for_inference(model->state().module());
    };
  }
}
}  // namespace detail
}  // namespace jyppx::liuding::vision::tasks
