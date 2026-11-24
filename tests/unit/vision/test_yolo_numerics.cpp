// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "yolo_numerics_reference.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>
#include <jyppx/liuding/vision/tasks/yolo_detection.hpp>

#include <torch/torch.h>

#include <ATen/autocast_mode.h>
#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <span>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace reference = yolo_numerics_reference;

template <std::size_t N>
Tensor Expected(const std::array<float, N>& values, torch::IntArrayRef shape,
                const Device& device) {
  return torch::from_blob(const_cast<float*>(values.data()), shape,
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone()
      .to(device);
}

void CheckNumerics(const Device& device, bool autocast, bool half_input) {
  torch::set_num_threads(1);
  auto raw = (((torch::arange(2 * 18 * 21, torch::kInt64) * 37) % 251 - 125)
                  .to(torch::kFloat32) /
              32)
                 .reshape({2, 18, 21})
                 .to(device, half_input ? torch::kFloat16 : torch::kFloat32)
                 .requires_grad_(true);
  const auto targets =
      torch::tensor({0., 0., 4., 4., 28., 28., 1., 1., 2., 6., 30., 30.})
          .reshape({2, 6})
          .to(device);
  const t::YoloDetectionOptions options{.num_classes = 2, .reg_max = 4};
  t::YoloDetectionLoss loss;
  {
    std::unique_ptr<training::amp::AutocastGuard> guard;
    if (autocast)
      guard = std::make_unique<training::amp::AutocastGuard>(
          device.type(), device.is_cpu() ? torch::kBFloat16 : torch::kFloat16);
    const auto enabled = at::autocast::is_autocast_enabled(device.type());
    const auto decoded = t::decode_yolo_detection(raw, {32, 32}, options);
    EXPECT_EQ(decoded.boxes.scalar_type(), torch::kFloat32);
    EXPECT_EQ(decoded.class_scores.scalar_type(), torch::kFloat32);
    loss = t::yolo_detection_loss(raw, targets, {32, 32}, options);
    EXPECT_EQ(at::autocast::is_autocast_enabled(device.type()), enabled);
    // A later eligible operation must still see the caller's autocast scope.
    auto matrix = torch::ones({2, 2}, raw.options().dtype(torch::kFloat32));
    EXPECT_EQ(torch::mm(matrix, matrix).scalar_type(),
              autocast ? (device.is_cpu() ? torch::kBFloat16 : torch::kFloat16)
                       : torch::kFloat32);
  }
  const auto actual =
      torch::stack({loss.total, loss.box * 7.5, loss.classification * .5,
                    loss.distribution * 1.5});
  EXPECT_TRUE(torch::allclose(actual, Expected(reference::loss, {4}, device),
                              1e-5, 1e-7));
  loss.total.backward();
  const auto expected_gradient =
      Expected(reference::gradient, {2, 18, 21}, device).to(raw.scalar_type());
  EXPECT_TRUE(torch::isfinite(raw.grad()).all().item<bool>());
  EXPECT_TRUE(torch::allclose(raw.grad(), expected_gradient,
                              half_input ? 1e-3 : 1e-5,
                              half_input ? 1e-5 : 1e-7));
  if (half_input) {
    auto fp32 = raw.detach().to(torch::kFloat32).requires_grad_(true);
    t::yolo_detection_loss(fp32, targets, {32, 32}, options).total.backward();
    EXPECT_TRUE(torch::equal(raw.grad(), fp32.grad().to(torch::kFloat16)));
    auto task_raw = raw.detach().clone().requires_grad_(true);
    const std::array<Tensor, 1> outputs{task_raw};
    t::YoloTaskOptions task;
    task.detection = options;
    t::yolo_task_loss(outputs, {.boxes = targets}, {32, 32}, task)
        .total.backward();
    EXPECT_TRUE(torch::equal(task_raw.grad(), raw.grad()));
  }

  const auto tiny =
      torch::tensor({3.995, 3.995, 4.005, 4.005}).reshape({1, 1, 4}).to(device);
  t::YoloDecodedPrediction prediction{
      tiny, torch::tensor({.5, .25}).reshape({1, 1, 2}).to(device),
      torch::tensor({.5, .5}).reshape({1, 2}).to(device),
      torch::full({1, 1}, 8., tiny.options())};
  const auto tiny_targets = torch::cat(
      {torch::zeros({1, 2}, tiny.options()), tiny.reshape({1, 4})}, 1);
  auto policy = options;
  policy.assignment_topk = 1;
  const auto assigned =
      t::assign_yolo_targets(prediction, tiny_targets, policy);
  EXPECT_TRUE(assigned.foreground.all().item<bool>());
  EXPECT_TRUE(torch::allclose(
      assigned.scores, Expected(reference::tiny_scores, {1, 1, 2}, device),
      1e-6, 1e-8));
}

TEST(VisionYoloNumerics, CpuReferenceAndAutocastPreserveFp32) {
  for (bool autocast : {false, true}) {
    SCOPED_TRACE(autocast);
    CheckNumerics(Device(torch::kCPU), autocast, false);
  }
}

TEST(VisionYoloNumerics, RejectsInvalidPredictionsBeforePrecisionPromotion) {
  const t::YoloDetectionOptions options{.num_classes = 2, .reg_max = 4};
  const auto targets = torch::empty({0, 6});
  EXPECT_THROW(t::yolo_detection_loss({}, targets, {32, 32}, options),
               InvalidArgument);
  EXPECT_THROW(t::yolo_detection_loss(torch::zeros({2, 18, 21}, torch::kInt64),
                                      targets, {32, 32}, options),
               InvalidArgument);
}

TEST(VisionYoloNumerics, CudaReferenceAndAutocastPreserveFp32) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "No CUDA-compatible GPU";
  for (bool autocast : {false, true}) {
    SCOPED_TRACE(autocast);
    CheckNumerics(Device(torch::kCUDA, 0), autocast, false);
  }
  CheckNumerics(Device(torch::kCUDA, 0), true, true);
}

TEST(VisionYoloNumerics, RecipeIdentityBindsOnlyAffectedTrainingSemantics) {
  const auto hash = [](const std::string& text) {
    return sha256_hex(std::as_bytes(std::span(text)));
  };
  for (const auto& scenario :
       {std::pair{"yolov8n", t::TaskKind::detect},
        std::pair{"yolo26n", t::TaskKind::pose},
        std::pair{"yoloe_v8s_text", t::TaskKind::detect},
        std::pair{"yoloe_v8s_text_segment", t::TaskKind::instance_segment}}) {
    t::RecipeSpec recipe;
    recipe.schema_version = 2;
    recipe.extension.training_manifest =
        std::filesystem::absolute("fixture.json");
    recipe.model.architecture = scenario.first;
    recipe.model.task = scenario.second;
    const auto canonical = t::recipe_json(recipe);
    EXPECT_NE(t::recipe_identity(recipe), hash(canonical));
    EXPECT_EQ(t::recipe_identity(recipe),
              hash(canonical + "\nyolo-detection-fp32/v2"));
    const auto restored =
        t::parse_recipe(canonical, std::filesystem::current_path());
    EXPECT_EQ(t::recipe_identity(restored), t::recipe_identity(recipe));
  }
  for (const auto& scenario :
       {std::pair{"yolov8n", t::TaskKind::classify},
        std::pair{"yolov5n", t::TaskKind::detect},
        std::pair{"resnet18", t::TaskKind::classify},
        std::pair{"fcos_resnet50_fpn", t::TaskKind::detect}}) {
    t::RecipeSpec recipe;
    recipe.schema_version = 2;
    recipe.extension.training_manifest = "__fixture__";
    recipe.model.architecture = scenario.first;
    recipe.model.task = scenario.second;
    EXPECT_EQ(t::recipe_identity(recipe), hash(t::recipe_json(recipe)));
  }
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
