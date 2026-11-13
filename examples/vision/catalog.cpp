// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
/** @file catalog.cpp @brief Bounded native model catalog and execution example.
 */
#include <jyppx/liuding/vision/models/yolo/model.hpp>
#include <jyppx/liuding/vision/models/yolo/yolov5.hpp>
#include <jyppx/liuding/vision/tasks/faster_rcnn.hpp>
#include <jyppx/liuding/vision/tasks/original_v5.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <ATen/Parallel.h>
#include <iostream>
namespace f = jyppx::liuding;
namespace v = f::vision::tasks;
namespace y = f::models::yolo;
int main(int argc, char** argv) {
  try {
    at::set_num_threads(2);
    torch::manual_seed(7);
    const bool all = argc > 1 && std::string_view(argv[1]) == "--catalog";
    std::cout << "modern selections=" << y::model_catalog().size()
              << " installed bindings=" << v::vision_catalog().size() << '\n';
    for (auto o : y::model_catalog()) {
      if (o.capacity != y::Capacity::nano_capacity) continue;
      if (!all &&
          (o.variant != y::Variant::p5 ||
           (o.task != y::Task::detect && o.family != y::Family::yolo26)))
        continue;
      o.num_classes = 2;
      o.keypoints = 3;
      auto model = std::make_shared<y::ModelImpl>(o);
      auto input = torch::rand({2, 3, 64, 64});
      auto output = model->forward(input);
      torch::Tensor loss;
      if (o.task == y::Task::classify)
        loss = v::classification_loss(output[0],
                                      torch::tensor({0, 1}, torch::kInt64));
      else {
        v::YoloTaskOptions task;
        task.task = static_cast<v::TaskKind>(o.task);
        task.detection.num_classes = 2;
        task.detection.reg_max = model->definition().reg_max;
        task.detection.strides = model->definition().strides;
        task.end_to_end = model->definition().end_to_end;
        task.generation26 = o.family == y::Family::yolo26;
        task.keypoints = 3;
        v::TaskTargets target;
        target.boxes = torch::tensor(std::vector<float>{0, 0, 8, 8, 40, 40, 1,
                                                        1, 16, 16, 48, 48})
                           .view({2, 6});
        target.masks = torch::zeros({2, 64, 64});
        target.masks.slice(1, 16, 40).slice(2, 16, 40).fill_(1);
        target.keypoints =
            torch::tensor(std::vector<float>{20, 20, 2, 24, 24, 2, 30, 30, 2,
                                             24, 24, 2, 28, 28, 2, 32, 32, 2})
                .view({2, 3, 3});
        target.oriented_boxes =
            torch::tensor(
                std::vector<float>{24, 24, 32, 32, .1f, 32, 32, 32, 32, .2f})
                .view({2, 5});
        v::PoseLogProbability flow;
        if (task.generation26 && o.task == y::Task::pose)
          flow = [model](const auto& x) {
            return model->pose_log_probability(x);
          };
        loss = v::yolo_task_loss(output, target, {64, 64}, task, 0, flow).total;
        auto result =
            v::decode_yolo_results(output, {64, 64}, task, {.confidence = .8});
        if (result.size() != 2) throw std::runtime_error("wrong result batch");
      }
      if (!torch::isfinite(loss).item<bool>())
        throw std::runtime_error("nonfinite model loss");
      loss.backward();
      std::cout << model->definition().id << " outputs=" << output.size()
                << " loss=" << loss.item<double>() << '\n';
    }
    for (bool p6 : {false, true}) {
      y::OriginalV5Options o;
      o.num_classes = 2;
      o.p6 = p6;
      auto model = y::OriginalV5(o);
      auto raw = model->forward(torch::rand({2, 3, 64, 64}));
      v::OriginalV5TaskOptions task;
      task.num_classes = 2;
      task.anchors = y::original_v5_anchors(p6);
      if (p6) {
        task.strides = {8, 16, 32, 64};
        task.objectness_balance = {4, 1, .25, .06};
      }
      auto target =
          torch::tensor(std::vector<float>{0, 0, 8, 8, 40, 40}).view({1, 6});
      auto loss = v::original_v5_loss(raw, target, {64, 64}, task).total;
      if (!torch::isfinite(loss).item<bool>())
        throw std::runtime_error("nonfinite original5 loss");
      loss.backward();
      std::cout << y::original_v5_model_id(o) << " loss=" << loss.item<double>()
                << '\n';
    }
    std::cout << "Native forward/backward example finished; no accuracy or "
                 "device qualification.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
