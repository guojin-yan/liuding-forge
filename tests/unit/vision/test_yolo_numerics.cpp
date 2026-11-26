// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "yolo26_coco245_reference.hpp"
#include "yolo26_numerics_reference.hpp"
#include "yolo26_obb_cpu2141_reference.hpp"
#include "yolo26_obb_numerics_reference.hpp"
#include "yolo_numerics_reference.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/vision/tasks/task.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>
#include <jyppx/liuding/vision/tasks/yolo_detection.hpp>

#include <torch/torch.h>
#include <torch/version.h>

#include <ATen/autocast_mode.h>
#include <array>
#include <gtest/gtest.h>
#include <limits>
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

void CheckBatchedAssignment(const Device& device) {
  torch::set_num_threads(1);
  torch::manual_seed(26);
  t::YoloDetectionOptions options;
  options.num_classes = 2;
  options.reg_max = 1;
  options.assignment_minimum_side = 16;
  options.assignment_topk2 = 1;
  auto raw = torch::randn({4, 6, 21}, torch::kFloat32).to(device);
  raw.slice(1, 0, 4).abs_().add_(1);
  auto targets =
      torch::tensor({3.f, 1.f,  3.f,  2.f, 25.f, 26.f, 1.f, 0.f,  4.f,
                     5.f, 23.f, 28.f, 2.f, 1.f,  1.f,  2.f, 17.f, 25.f,
                     3.f, 0.f,  2.f,  4.f, 20.f, 30.f, 2.f, 0.f,  8.f,
                     7.f, 29.f, 31.f, 3.f, 1.f,  6.f,  3.f, 27.f, 28.f})
          .view({6, 6})
          .to(device);
  auto prediction = t::decode_yolo_detection(raw, {32, 32}, options);
  // Exercise both the grouped path and the one-image chunk boundary.
  for (const auto budget : {1048576, 63}) {
    options.max_assignment_pairs = budget;
    auto actual = t::assign_yolo_targets(prediction, targets, options);
    for (std::int64_t b = 0; b < 4; ++b) {
      auto rows = (targets.select(1, 0) == b).nonzero().flatten();
      auto gt = targets.index_select(0, rows);
      gt.select(1, 0).zero_();
      auto one =
          t::decode_yolo_detection(raw.narrow(0, b, 1), {32, 32}, options);
      auto expected = t::assign_yolo_targets(one, gt, options);
      EXPECT_TRUE(torch::equal(actual.foreground[b], expected.foreground[0]));
      EXPECT_TRUE(
          torch::allclose(actual.scores[b], expected.scores[0], 1e-6, 1e-7));
      auto mask = expected.foreground[0];
      EXPECT_TRUE(torch::equal(actual.boxes[b].index({mask}),
                               expected.boxes[0].index({mask})));
      if (rows.numel())
        EXPECT_TRUE(torch::equal(
            actual.target_indices[b].index({mask}),
            rows.index_select(0, expected.target_indices[0].index({mask}))));
      EXPECT_TRUE(
          (actual.target_indices[b].index({~mask}) == -1).all().item<bool>());
    }
  }
  options.max_assignment_pairs = 62;
  EXPECT_THROW(t::assign_yolo_targets(prediction, targets, options),
               ResourceExhausted);
}

TEST(VisionYoloNumerics, BatchedAssignmentPreservesImageRowsAndChunkBoundsCpu) {
  CheckBatchedAssignment(Device(torch::kCPU));
}

TEST(VisionYoloNumerics,
     BatchedAssignmentPreservesImageRowsAndChunkBoundsCuda) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA device unavailable";
  CheckBatchedAssignment(Device(torch::kCUDA, 0));
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

void CheckYolo26Numerics(const Device& device, bool autocast,
                         bool original_coco = false) {
  torch::set_num_threads(1);
  namespace e2e = yolo26_numerics_reference;
  namespace coco = yolo26_coco245_reference;
  const auto base =
      torch::arange(2 * 7 * 85, torch::kInt64).reshape({2, 7, 85});
  auto many = ((base * 37 % 251 - 125).to(torch::kFloat32) / 32).to(device);
  auto one = ((base * 19 % 229 - 114).to(torch::kFloat32) / 32).to(device);
  many.slice(1, 0, 4).abs_().add_(.25);
  one.slice(1, 0, 4).abs_().add_(.5);
  many.requires_grad_(true);
  one.requires_grad_(true);
  const auto targets =
      torch::tensor({0.,  0.,  10., 10., 14., 14., 0.,  1.,  8.,  8.,
                     30., 29., 0.,  2.,  24., 16., 50., 54., 1.,  1.,
                     29., 29., 35., 35., 1.,  0.,  6.,  10., 42., 50.})
          .reshape({5, 6})
          .to(device);
  t::YoloTaskOptions options;
  options.end_to_end = options.generation26 = true;
  options.one_to_many_initial = .8;
  options.detection.num_classes = 3;
  options.detection.reg_max = 1;
  options.detection.strides = {8, 16, 32, 64};
  options.detection.assignment_minimum_side = 16;
  if (original_coco) {
    options.detection.box_gain = 5.62767;
    options.detection.class_gain = .56099;
    options.detection.dfl_gain = 9.03871;
    options.one_to_one_topk = 8;
    options.fixed_branch_gains = true;
    options.schedule_epochs = 245;
  }
  const std::array<Tensor, 2> raw{many, one};
  const auto saved_many = many.detach().clone(),
             saved_one = one.detach().clone();
  const auto saved_targets = targets.clone();
  const auto expected_loss = original_coco ? Expected(coco::losses, {4}, device)
                                           : Expected(e2e::losses, {4}, device);
  std::unique_ptr<training::amp::AutocastGuard> guard;
  if (autocast)
    guard = std::make_unique<training::amp::AutocastGuard>(
        device.type(), device.is_cpu() ? torch::kBFloat16 : torch::kFloat16);
  const auto epochs = original_coco
                          ? std::array<std::uint64_t, 4>{0, 1, 244, 245}
                          : std::array<std::uint64_t, 4>{0, 1, 99, 100};
  for (std::size_t i = 0; i < epochs.size(); ++i) {
    SCOPED_TRACE(epochs[i]);
    auto loss = t::yolo_task_loss(raw, {.boxes = targets}, {64, 64}, options,
                                  epochs[i]);
    EXPECT_EQ(loss.total.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(loss.total, expected_loss[i], 1e-5, 1e-6));
    if (i == 0) loss.total.backward();
  }
  EXPECT_TRUE(torch::allclose(
      many.grad(),
      original_coco ? Expected(coco::many_gradient, {2, 7, 85}, device)
                    : Expected(e2e::many_gradient, {2, 7, 85}, device),
      1e-5, 1e-7));
  EXPECT_TRUE(torch::allclose(
      one.grad(),
      original_coco ? Expected(coco::one_gradient, {2, 7, 85}, device)
                    : Expected(e2e::one_gradient, {2, 7, 85}, device),
      1e-5, 1e-7));
  options.schedule_epochs = 1;
  EXPECT_TRUE(torch::allclose(
      t::yolo_task_loss(raw, {.boxes = targets}, {64, 64}, options, 1).total,
      (original_coco ? Expected(coco::single_epoch_loss, {1}, device)
                     : Expected(e2e::single_epoch_loss, {1}, device))
          .squeeze(),
      1e-5, 1e-6));
  for (int branch = 0; branch < 2; ++branch) {
    auto policy = options.detection;
    if (branch) {
      policy.assignment_topk = original_coco ? 8 : 7;
      policy.assignment_topk2 = 1;
    }
    const auto decoded =
        t::decode_yolo_detection(raw[branch], {64, 64}, policy);
    const auto assigned = t::assign_yolo_targets(decoded, targets, policy);
    const auto scores =
        original_coco
            ? (branch ? Expected(coco::one_scores, {2, 85, 3}, device)
                      : Expected(coco::many_scores, {2, 85, 3}, device))
            : (branch ? Expected(e2e::one_scores, {2, 85, 3}, device)
                      : Expected(e2e::many_scores, {2, 85, 3}, device));
    const auto fg =
        original_coco
            ? (branch ? Expected(coco::one_foreground, {2, 85}, device)
                      : Expected(coco::many_foreground, {2, 85}, device))
            : (branch ? Expected(e2e::one_foreground, {2, 85}, device)
                      : Expected(e2e::many_foreground, {2, 85}, device));
    const auto ids =
        original_coco ? (branch ? Expected(coco::one_indices, {2, 85}, device)
                                : Expected(coco::many_indices, {2, 85}, device))
                      : (branch ? Expected(e2e::one_indices, {2, 85}, device)
                                : Expected(e2e::many_indices, {2, 85}, device));
    EXPECT_TRUE(torch::allclose(assigned.scores, scores, 1e-5, 1e-7));
    EXPECT_TRUE(torch::equal(assigned.foreground, fg.to(torch::kBool)));
    EXPECT_TRUE(torch::equal(assigned.target_indices, ids.to(torch::kInt64)));
  }
  EXPECT_TRUE(torch::equal(many, saved_many));
  EXPECT_TRUE(torch::equal(one, saved_one));
  EXPECT_TRUE(torch::equal(targets, saved_targets));
}

TEST(VisionYoloNumerics, Yolo26CpuMatchesIndependentE2eScheduleAndAssignments) {
  for (bool autocast : {false, true})
    CheckYolo26Numerics(Device(torch::kCPU), autocast);
}

TEST(VisionYoloNumerics,
     Yolo26CudaMatchesIndependentE2eScheduleAndAssignments) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "No CUDA-compatible GPU";
  for (bool autocast : {false, true})
    CheckYolo26Numerics(Device(torch::kCUDA, 0), autocast);
}

TEST(VisionYoloNumerics, Yolo26Coco245CpuMatchesOriginalLossAndGradients) {
  for (bool autocast : {false, true})
    CheckYolo26Numerics(Device(torch::kCPU), autocast, true);
}

TEST(VisionYoloNumerics, Yolo26Coco245CudaMatchesOriginalLossAndGradients) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "No CUDA-compatible GPU";
  for (bool autocast : {false, true})
    CheckYolo26Numerics(Device(torch::kCUDA, 0), autocast, true);
}

void CheckYolo26ObbNumerics(const Device& device, bool autocast,
                            bool half_input) {
  torch::set_num_threads(1);
  namespace obb = yolo26_obb_numerics_reference;
  const auto dtype = half_input ? torch::kFloat16 : torch::kFloat32;
  auto many = Expected(obb::raw_many, {2, 8, 85}, device)
                  .to(dtype)
                  .requires_grad_(true);
  auto one =
      Expected(obb::raw_one, {2, 8, 85}, device).to(dtype).requires_grad_(true);
  const auto rows = Expected(obb::rows, {6, 7}, device);
  const auto oriented = rows.slice(1, 2, 7).contiguous();
  const auto center = oriented.slice(1, 0, 2);
  const auto half = oriented.slice(1, 2, 4) * .5;
  const auto boxes =
      torch::cat({rows.slice(1, 0, 2), center - half, center + half}, 1);
  const auto saved_many = many.detach().clone(),
             saved_one = one.detach().clone();
  const auto saved_boxes = boxes.clone(), saved_oriented = oriented.clone();
  t::YoloTaskOptions options;
  options.task = t::TaskKind::obb;
  options.end_to_end = options.generation26 = true;
  options.one_to_many_initial = .8;
  options.detection.num_classes = 3;
  options.detection.reg_max = 1;
  options.detection.strides = {8, 16, 32, 64};
  options.detection.assignment_minimum_side = 16;
  const std::array<Tensor, 2> raw{many, one};
  std::unique_ptr<training::amp::AutocastGuard> guard;
  if (autocast)
    guard = std::make_unique<training::amp::AutocastGuard>(
        device.type(), device.is_cpu() ? torch::kBFloat16 : torch::kFloat16);
  const std::array<std::uint64_t, 4> epochs{0, 1, 99, 100};
  for (std::size_t i = 0; i < epochs.size(); ++i) {
    SCOPED_TRACE(epochs[i]);
    auto loss =
        t::yolo_task_loss(raw, {.boxes = boxes, .oriented_boxes = oriented},
                          {64, 64}, options, epochs[i]);
    EXPECT_EQ(loss.total.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(torch::allclose(
        loss.total, Expected(obb::losses, {4}, device)[i], 1e-5, 1e-6))
        << loss.total;
    if (i == 0) {
      loss.total.backward();
      const std::array<std::string, 4> components{"box", "classification",
                                                  "distribution", "angle"};
      for (int branch = 0; branch < 2; ++branch) {
        const auto expected = branch
                                  ? Expected(obb::one_components, {4}, device)
                                  : Expected(obb::many_components, {4}, device);
        for (std::size_t j = 0; j < components.size(); ++j) {
          const auto key =
              std::string(branch ? "one_to_one." : "one_to_many.") +
              components[j];
          const auto found = loss.components.find(key);
          EXPECT_NE(found, loss.components.end()) << key;
          if (found != loss.components.end())
            EXPECT_TRUE(torch::allclose(found->second, expected[j], 1e-5, 1e-6))
                << key;
        }
      }
    }
  }
  // CPU 2.14.1's unchanged official criterion differs from the CPU 2.14.0
  // carrier near zero gradients. Bind that exact backend fixture, preserving
  // the original tolerance; GPU references retain the AMD 2.14.0 fixture.
  const bool cpu2141 = device.is_cpu() && TORCH_VERSION_MAJOR == 2 &&
                       TORCH_VERSION_MINOR == 14 && TORCH_VERSION_PATCH == 1;
  const auto expected_many =
      cpu2141 ? Expected(yolo26_obb_cpu2141_reference::many_gradient,
                         {2, 8, 85}, device)
              : Expected(obb::many_gradient, {2, 8, 85}, device);
  const auto expected_one =
      cpu2141 ? Expected(yolo26_obb_cpu2141_reference::one_gradient, {2, 8, 85},
                         device)
              : Expected(obb::one_gradient, {2, 8, 85}, device);
  EXPECT_TRUE(
      torch::allclose(many.grad(), expected_many.to(dtype), 1e-5, 1e-7));
  EXPECT_TRUE(torch::allclose(one.grad(), expected_one.to(dtype), 1e-5, 1e-7));
  for (bool tiny : {false, true}) {
    const auto b = tiny ? boxes.slice(0, 3, 4) : boxes.slice(0, 0, 0);
    const auto r = tiny ? oriented.slice(0, 3, 4) : oriented.slice(0, 0, 0);
    auto loss = t::yolo_task_loss(raw, {.boxes = b, .oriented_boxes = r},
                                  {64, 64}, options);
    EXPECT_TRUE(torch::allclose(
        loss.total, Expected(obb::empty_loss, {1}, device).squeeze(), 1e-5,
        1e-6));
  }
  EXPECT_TRUE(torch::equal(many, saved_many));
  EXPECT_TRUE(torch::equal(one, saved_one));
  EXPECT_TRUE(torch::equal(boxes, saved_boxes));
  EXPECT_TRUE(torch::equal(oriented, saved_oriented));
}

TEST(VisionYoloNumerics, Yolo26ObbCpuMatchesIndependentCriterion) {
  for (bool autocast : {false, true})
    for (bool half_input : {false, true})
      CheckYolo26ObbNumerics(Device(torch::kCPU), autocast, half_input);
}

TEST(VisionYoloNumerics, Yolo26ObbCudaMatchesIndependentCriterion) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "No CUDA-compatible GPU";
  for (bool autocast : {false, true})
    for (bool half_input : {false, true})
      CheckYolo26ObbNumerics(Device(torch::kCUDA, 0), autocast, half_input);
}

TEST(VisionYoloNumerics,
     RejectsInvalidSecondaryAssignmentAndComplementaryGains) {
  const auto raw = torch::zeros({1, 7, 85});
  const auto targets = torch::empty({0, 6});
  t::YoloDetectionOptions options{
      .num_classes = 3, .reg_max = 1, .strides = {8, 16, 32, 64}};
  for (auto topk : {std::int64_t{-1}, std::int64_t{11}}) {
    options.assignment_topk2 = topk;
    EXPECT_THROW(t::yolo_detection_loss(raw, targets, {64, 64}, options),
                 InvalidArgument);
  }
  options.assignment_topk2 = 0;
  for (auto side : {-1., 65537., std::numeric_limits<double>::infinity()}) {
    options.assignment_minimum_side = side;
    EXPECT_THROW(t::yolo_detection_loss(raw, targets, {64, 64}, options),
                 InvalidArgument);
  }
  t::YoloTaskOptions task;
  task.generation26 = task.end_to_end = true;
  task.detection.num_classes = 3;
  task.detection.reg_max = 1;
  task.detection.strides = {8, 16, 32, 64};
  const std::array<Tensor, 2> outputs{raw, raw};
  task.one_to_many_initial = .8;
  task.one_to_many_final = .1;
  EXPECT_NO_THROW(
      t::yolo_task_loss(outputs, {.boxes = targets}, {64, 64}, task));
  task.one_to_many_initial = 1.01;
  EXPECT_THROW(t::yolo_task_loss(outputs, {.boxes = targets}, {64, 64}, task),
               InvalidArgument);
  task.one_to_many_initial = .8;
  task.one_to_many_final = 1.01;
  EXPECT_THROW(t::yolo_task_loss(outputs, {.boxes = targets}, {64, 64}, task),
               InvalidArgument);
  task.one_to_many_final = .1;
  for (auto gain : {-1., std::numeric_limits<double>::infinity()}) {
    task.angle_gain = gain;
    EXPECT_THROW(t::yolo_task_loss(outputs, {.boxes = targets}, {64, 64}, task),
                 InvalidArgument);
  }
}

TEST(VisionYoloNumerics, RecipeIdentityBindsOnlyAffectedTrainingSemantics) {
  const auto hash = [](const std::string& text) {
    return sha256_hex(std::as_bytes(std::span(text)));
  };
  for (const auto& scenario :
       {std::pair{"yolov8n", t::TaskKind::detect},
        std::pair{"yolo26n", t::TaskKind::pose},
        std::pair{"yolo26n", t::TaskKind::obb},
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
              hash(canonical + (recipe.model.architecture.starts_with("yolo26")
                                    ? (recipe.model.task == t::TaskKind::obb
                                           ? "\nyolo26-obb-fp32/v4"
                                           : "\nyolo26-e2e-fp32/v3")
                                    : "\nyolo-detection-fp32/v2")));
    if (scenario.second == t::TaskKind::obb)
      EXPECT_NE(t::recipe_identity(recipe),
                hash(canonical + "\nyolo26-e2e-fp32/v3"));
    const auto restored =
        t::parse_recipe(canonical, std::filesystem::current_path());
    EXPECT_EQ(t::recipe_identity(restored), t::recipe_identity(recipe));
  }
  for (const auto& scenario :
       {std::pair{"yolov8n", t::TaskKind::classify},
        std::pair{"yolo26n", t::TaskKind::classify},
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
