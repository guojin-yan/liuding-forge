// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;

struct RecoveryCase {
  std::string architecture;
  std::string variant;
};

void PrintTo(const RecoveryCase& value, std::ostream* output) {
  *output << value.architecture << '_' << value.variant;
}

constexpr int kMaximumCycles = 32;

std::string ScaleText(double scale) {
  std::ostringstream output;
  output << std::setprecision(17) << scale;
  return output.str();
}

const std::array<RecoveryCase, 11> kCases{{
    {"yolov8s", "ghost_p6"},
    {"yolov8m", "ghost_p6"},
    {"yolov8l", "ghost_p6"},
    {"yolov8x", "ghost_p6"},
    {"yolo26n", "p6"},
    {"yolo26s", "p6"},
    {"yolo26m", "p6"},
    {"yolo26l", "p6"},
    {"yolo26x", "p6"},
    {"yolov5l", "p6"},
    {"yolov5x", "p6"},
}};

class VisionAmpOverflowRecovery
    : public ::testing::TestWithParam<RecoveryCase> {};

TEST_P(VisionAmpOverflowRecovery, FiniteUpdateAfterDynamicBackoff) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "No CUDA-compatible GPU available";
  const auto& test = GetParam();
  const auto catalog = t::vision_catalog();
  const auto found =
      std::find_if(catalog.begin(), catalog.end(), [&](const auto& entry) {
        return entry.architecture == test.architecture &&
               entry.variant == test.variant &&
               entry.task == t::TaskKind::detect;
      });
  ASSERT_NE(found, catalog.end());
  const auto& entry = *found;
  torch::set_num_threads(1);
  torch::manual_seed(7);
  const Device device(torch::kCUDA, 0);
  t::RecipeSpec recipe;
  recipe.schema_version = 2;
  recipe.model.architecture = entry.architecture;
  recipe.model.task = entry.task;
  recipe.model.variant = entry.variant;
  recipe.model.num_classes = 3;
  recipe.device = device;
  recipe.training.source = "__execution_fixture__";
  recipe.classification_root = "__execution_fixture__";
  recipe.extension.training_manifest = "__execution_fixture__";
  const auto side = (std::max)(std::int64_t{64}, entry.minimum_input_size);
  const auto height =
      entry.fixed_input_size[0] ? entry.fixed_input_size[0] : side;
  const auto width =
      entry.fixed_input_size[1] ? entry.fixed_input_size[1] : side;
  recipe.data.letterbox.canvas = {static_cast<std::uint32_t>(width),
                                  static_cast<std::uint32_t>(height)};
  auto binding = t::bind_task(recipe);
  auto image = torch::randn({2, 3, height, width});
  auto boxes =
      torch::tensor({0., 0., 8., 8., 32., 32., 1., 1., 12., 12., 40., 40.})
          .view({2, 6});
  binding.model->state().to(device);
  image = image.to(device);
  boxes = boxes.to(device);
  binding.model->state().train();
  nn::Invocation invocation{{image}, {}};
  data::BatchEnvelope envelope;
  training::StepContext context;
  std::vector<optim::OptimizerParameterBinding> parameters;
  std::vector<std::pair<Tensor, Tensor>> initial;
  for (const auto& item : binding.model->state().named_parameters()) {
    ASSERT_EQ(item.value().scalar_type(), torch::kFloat32);
    initial.emplace_back(item.value(), item.value().detach().clone());
    if (item.value().requires_grad())
      parameters.push_back(
          {.parameter_id = item.key(), .parameter = item.value()});
  }
  auto optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
      "vision.optimizer", std::vector<optim::SgdParameterGroup>{
                              {.group_id = "main",
                               .parameters = std::move(parameters),
                               .options = {.learning_rate = .001}}});
  training::amp::GradScaler scaler({.init_scale = 8});
  int overflows = 0;
  bool updated = false;
  RecordProperty("input_height", std::to_string(height));
  RecordProperty("input_width", std::to_string(width));
  // The strict one-step catalog contract is kept separately. This contract
  // exercises ordinary dynamic backoff on the same fixed synthetic batch;
  // BatchNorm buffers keep their normal training-forward updates.
  for (int cycle = 0; cycle < kMaximumCycles; ++cycle) {
    SCOPED_TRACE("cycle=" + std::to_string(cycle));
    optimizer->zero_grad();
    const auto scale = scaler.get_scale();
    training::StepOutput output;
    {
      training::amp::AutocastGuard autocast(c10::DeviceType::CUDA,
                                            torch::kFloat16);
      output =
          binding.step({*binding.model, envelope, invocation, boxes, context});
    }
    ASSERT_TRUE(output.loss.defined());
    ASSERT_EQ(output.loss.numel(), 1);
    ASSERT_EQ(output.loss.device(), device);
    ASSERT_EQ(output.sample_count, 2);
    ASSERT_TRUE(torch::isfinite(output.loss).item<bool>());
    scaler.scale(output.loss).backward();
    scaler.unscale(optimizer);
    bool finite = true;
    Tensor strongest;
    double maximum = 0;
    for (const auto& parameter : binding.model->state().parameters()) {
      ASSERT_EQ(parameter.device(), device);
      ASSERT_EQ(parameter.scalar_type(), torch::kFloat32);
      if (!parameter.grad().defined()) continue;
      if (!torch::isfinite(parameter.grad()).all().item<bool>()) {
        finite = false;
        continue;
      }
      const auto magnitude = parameter.grad().abs().max().item<double>();
      if (magnitude > maximum) {
        maximum = magnitude;
        strongest = parameter;
      }
    }
    const bool allowed = scaler.cycle_allowed();
    ASSERT_EQ(allowed, finite);
    ASSERT_EQ(scaler.step_allowed(optimizer), allowed);
    std::cout << "AMP_RECOVERY_CYCLE " << cycle << " scale=" << ScaleText(scale)
              << " allowed=" << allowed << '\n';
    RecordProperty("attempted_cycles", cycle + 1);
    if (!allowed) {
      ASSERT_EQ(optimizer->completed_steps(), 0U);
      for (const auto& [parameter, original] : initial)
        ASSERT_TRUE(torch::equal(parameter, original));
      scaler.mark_step(optimizer, false);
      scaler.update();
      ASSERT_DOUBLE_EQ(scaler.get_scale(), scale * .5);
      ++overflows;
      RecordProperty("overflow_skips", overflows);
      RecordProperty("final_scale", ScaleText(scaler.get_scale()));
      continue;
    }
    ASSERT_GT(maximum, 0);
    ASSERT_TRUE(strongest.defined());
    const auto before = strongest.detach().clone();
    optimizer->step();
    scaler.mark_step(optimizer, true);
    scaler.update();
    ASSERT_EQ(optimizer->completed_steps(), 1U);
    ASSERT_FALSE(torch::equal(before, strongest));
    for (const auto& [parameter, original] : initial) {
      ASSERT_TRUE(torch::isfinite(parameter).all().item<bool>());
      if (!parameter.requires_grad()) {
        ASSERT_FALSE(parameter.grad().defined());
        ASSERT_TRUE(torch::equal(parameter, original));
      }
    }
    ASSERT_DOUBLE_EQ(scaler.get_scale(), scale);
    RecordProperty("overflow_skips", overflows);
    RecordProperty("final_scale", ScaleText(scale));
    RecordProperty("completed_optimizer_steps", 1);
    std::cout << "AMP_RECOVERY_RESULT attempts=" << cycle + 1
              << " overflow_skips=" << overflows
              << " final_scale=" << ScaleText(scale)
              << " optimizer_steps=" << optimizer->completed_steps() << '\n';
    updated = true;
    break;
  }
  ASSERT_TRUE(updated) << "No finite update within " << kMaximumCycles
                       << " original dynamic-scaling cycles";
}

INSTANTIATE_TEST_SUITE_P(P6Models, VisionAmpOverflowRecovery,
                         ::testing::ValuesIn(kCases), [](const auto& info) {
                           auto variant = info.param.variant;
                           std::replace(variant.begin(), variant.end(), '-',
                                        '_');
                           return info.param.architecture + "_" + variant;
                         });
}  // namespace
}  // namespace jyppx::liuding::vision::tests
