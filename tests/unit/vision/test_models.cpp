// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/torch.h>

#include <cstdlib>
#include <gtest/gtest.h>

namespace jyppx::liuding::vision::tasks {
void PrintTo(const CapabilityRecord& entry, std::ostream* output) {
  *output << entry.architecture << '_' << task_name(entry.task) << '_'
          << entry.variant;
}
}  // namespace jyppx::liuding::vision::tasks

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
using K = t::TaskKind;
using I = std::int64_t;

class VisionModelTraining
    : public ::testing::TestWithParam<t::CapabilityRecord> {};

TEST_P(VisionModelTraining, CpuLossBackwardAndActualUpdate) {
  const auto& entry = GetParam();
  // The classic 4096-wide VGG region head needs a separate memory admission.
  if (entry.architecture == "fast_rcnn_vgg16" &&
      std::getenv("JYPPX_LIUDING_VISION_VGG_RESOURCE_GATE") == nullptr)
    GTEST_SKIP()
        << "Classic VGG region training requires explicit memory admission";
  torch::set_num_threads(1);
  torch::manual_seed(7);
  t::RecipeSpec recipe;
  recipe.schema_version = 2;
  recipe.model.architecture = entry.architecture;
  recipe.model.task = entry.task;
  recipe.model.variant = entry.variant;
  recipe.model.num_classes = 3;
  recipe.training.source = "__execution_fixture__";
  recipe.classification_root = "__execution_fixture__";
  recipe.extension.training_manifest = "__execution_fixture__";
  if (entry.architecture.starts_with("simmim_vit"))
    recipe.extension.model_patch_size = 16;
  const I side = (std::max)(I{64}, entry.minimum_input_size);
  const I height = entry.fixed_input_size[0] ? entry.fixed_input_size[0] : side;
  const I width = entry.fixed_input_size[1] ? entry.fixed_input_size[1] : side;
  recipe.data.letterbox.canvas = {static_cast<std::uint32_t>(width),
                                  static_cast<std::uint32_t>(height)};
  auto binding = t::bind_task(recipe);
  auto image = torch::randn({2, 3, height, width});
  nn::Invocation invocation{{image}, {}};
  const auto boxes =
      torch::tensor({0., 0., 8., 8., 32., 32., 1., 1., 12., 12., 40., 40.})
          .view({2, 6});
  std::optional<c10::IValue> target;
  if (entry.component == "models_rcnn") {
    invocation.positional.insert(
        invocation.positional.end(),
        {boxes, torch::tensor({float(height), float(width)})
                    .view({1, 2})
                    .expand({2, 2})});
    target = boxes;
  } else if (entry.architecture.starts_with("rtdetr_")) {
    invocation.positional.push_back(boxes);
    target = boxes;
  } else if (entry.requires_external_proposals) {
    auto rois = torch::cat({boxes.slice(1, 0, 1), boxes.slice(1, 2, 6)}, 1);
    invocation.positional.insert(invocation.positional.end(),
                                 {rois, boxes,
                                  torch::tensor({float(height), float(width)})
                                      .view({1, 2})
                                      .expand({2, 2})});
    target = boxes;
  } else if (entry.architecture.starts_with("yoloe_") ||
             entry.architecture.starts_with("yoloworld_")) {
    invocation.positional.push_back(torch::randn({2, 3, 512}));
    invocation.positional.push_back(torch::ones({2, 3, height / 8, width / 8}));
    auto masks = torch::zeros({2, height, width});
    masks[0].slice(0, 8, 32).slice(1, 8, 32).fill_(1);
    masks[1].slice(0, 12, 40).slice(1, 12, 40).fill_(1);
    target = c10::ivalue::Tuple::create(
        {boxes, masks, torch::empty({0}), torch::empty({0})});
  } else if (entry.task == K::pretrain_simmim) {
    auto mask = torch::ones({2, height / recipe.extension.model_patch_size,
                             width / recipe.extension.model_patch_size},
                            torch::kBool);
    invocation.positional.push_back(mask);
    target = c10::ivalue::Tuple::create({image, mask});
  } else if (entry.task == K::semantic_segment) {
    target = torch::zeros({2, height, width}, torch::kInt64);
  } else if (entry.task == K::depth) {
    target = c10::ivalue::Tuple::create(
        {torch::ones({2, 1, height, width}),
         torch::ones({2, 1, height, width}, torch::kBool)});
  } else if (entry.task == K::classify_multilabel) {
    target = c10::ivalue::Tuple::create(
        {torch::zeros({2, 3}), torch::ones({2, 3}, torch::kBool)});
  } else if (entry.task == K::classify) {
    target = torch::tensor({0, 1}, torch::kInt64);
  } else if (entry.task == K::reid) {
    // A valid P=2,K=2 batch exercises the triplet term as well as CE.
    image = torch::randn({4, 3, height, width});
    invocation.positional[0] = image;
    target = torch::tensor({0, 0, 1, 1}, torch::kInt64);
  } else if (entry.task == K::instance_segment) {
    auto masks = torch::zeros({2, height, width});
    masks[0].slice(0, 8, 32).slice(1, 8, 32).fill_(1);
    masks[1].slice(0, 12, 40).slice(1, 12, 40).fill_(1);
    target = c10::ivalue::Tuple::create(
        {boxes, masks, torch::empty({0}), torch::empty({0})});
  } else if (entry.task == K::pose) {
    auto keypoints = torch::full({2, 17, 3}, 20.);
    keypoints.select(2, 2).fill_(2);
    target = c10::ivalue::Tuple::create(
        {boxes, torch::empty({0}), keypoints, torch::empty({0})});
  } else if (entry.task == K::obb) {
    auto oriented =
        torch::tensor({20., 20., 24., 24., .2, 26., 26., 28., 28., -.3})
            .view({2, 5});
    target = c10::ivalue::Tuple::create(
        {boxes, torch::empty({0}), torch::empty({0}), oriented});
  } else {
    target = boxes;
  }
  binding.model->state().train();
  data::BatchEnvelope envelope;
  training::StepContext context;
  auto output =
      binding.step({*binding.model, envelope, invocation, target, context});
  ASSERT_TRUE(output.loss.defined());
  ASSERT_EQ(output.loss.numel(), 1);
  ASSERT_TRUE(torch::isfinite(output.loss).item<bool>());
  output.loss.backward();
  Tensor strongest;
  double maximum = 0;
  for (const auto& parameter : binding.model->state().parameters()) {
    if (!parameter.grad().defined()) continue;
    ASSERT_TRUE(torch::isfinite(parameter.grad()).all().item<bool>());
    const auto magnitude = parameter.grad().abs().max().item<double>();
    if (magnitude > maximum) {
      maximum = magnitude;
      strongest = parameter;
    }
  }
  ASSERT_GT(maximum, 0);
  const auto before = strongest.detach().clone();
  torch::optim::SGD optimizer(binding.model->state().parameters(),
                              torch::optim::SGDOptions(.001));
  optimizer.step();
  EXPECT_FALSE(torch::equal(before, strongest));
}

INSTANTIATE_TEST_SUITE_P(InstalledCatalog, VisionModelTraining,
                         ::testing::ValuesIn(t::vision_catalog()),
                         [](const auto& info) {
                           return info.param.architecture + "_" +
                                  std::string(t::task_name(info.param.task)) +
                                  "_" + info.param.variant;
                         });
}  // namespace
}  // namespace jyppx::liuding::vision::tests
