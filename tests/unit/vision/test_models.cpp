// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "../../../src/jyppx/liuding/vision/models/yolo/blocks.hpp"
#include "yolo_attention_reference.hpp"

#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/amp/autocast.hpp>
#include <jyppx/liuding/training/amp/grad_scaler.hpp>
#include <jyppx/liuding/vision/models/extended/rtdetr.hpp>
#include <jyppx/liuding/vision/models/yolo/model.hpp>
#include <jyppx/liuding/vision/tasks/extended.hpp>
#include <jyppx/liuding/vision/tasks/workflow/binding.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

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

TEST(VisionModelGraph, SppfActivationFollowsFrozenLegacyAndExtendedConfigs) {
  namespace y = models::yolo;
  namespace d = models::yolo::detail;
  torch::set_num_threads(1);
  struct Case {
    y::Family family;
    y::Variant variant;
    bool first_active;
  };
  for (const auto& scenario :
       {Case{y::Family::v8, y::Variant::p5, true},
        Case{y::Family::v8, y::Variant::ghost_p6, true},
        Case{y::Family::yolo26, y::Variant::p5, false}}) {
    const y::ModelOptions options{.family = scenario.family,
                                  .variant = scenario.variant,
                                  .num_classes = 3};
    EXPECT_TRUE(y::describe_model(options).revision.starts_with(
        scenario.first_active ? "native-yolo-graph/v2/"
                              : "native-yolo-graph/v3/"));
    auto model = y::create_model(options);
    bool found = false;
    for (const auto& item : model.state().module().named_modules()) {
      auto sppf = std::dynamic_pointer_cast<d::Sppf>(item.value());
      if (!sppf) continue;
      found = true;
      SCOPED_TRACE(item.key());
      auto first = std::dynamic_pointer_cast<d::Conv>(sppf->a);
      ASSERT_TRUE(first);
      EXPECT_EQ(first->activation, scenario.first_active);
      // With zeroed projections and a negative BN bias, the first output is
      // SiLU(-1) for the legacy graph and exactly -1 for the extended graph.
      sppf->eval();
      {
        torch::NoGradGuard guard;
        first->conv->weight.zero_();
        first->bn->weight.zero_();
        first->bn->bias.fill_(-1);
      }
      const auto input =
          torch::zeros({1, first->conv->options.in_channels(), 3, 3});
      const auto output = sppf->a->run(input);
      const auto expected =
          scenario.first_active ? -1.0 / (1.0 + std::exp(1.0)) : -1.0;
      EXPECT_TRUE(torch::allclose(output, torch::full_like(output, expected),
                                  1e-6, 1e-7));
    }
    EXPECT_TRUE(found);
  }
}

void check_spatial_attention_reference(const Device& device,
                                       c10::ScalarType dtype) {
  namespace d = models::yolo::detail;
  namespace r = yolo_attention_reference;
  struct Packed final : d::Unit {
    Tensor value;
    explicit Packed(const Tensor& tensor) {
      value = register_parameter("value", tensor.clone());
    }
    Tensor run(const Tensor&) override { return value; }
  };
  struct Zero final : d::Unit {
    Tensor run(const Tensor& x) override { return torch::zeros_like(x); }
  };
  const auto tensor = [&](const auto& values, std::vector<I> shape) {
    return torch::tensor(std::vector<float>(values.begin(), values.end()))
        .view(shape)
        .to(device, dtype);
  };
  const auto packed =
      dtype == torch::kFloat32 ? r::fp32_packed : r::fp16_packed;
  auto projection = std::make_shared<Packed>(tensor(packed, {1, 24, 2, 3}));
  d::Attention attention(12, 2);
  attention.qkv = projection;
  attention.projection = std::make_shared<d::Identity>();
  attention.positional = std::make_shared<Zero>();
  // The fixed official forward isolates attention from convolution kernels.
  // A non-power-of-two key width exposes premature FP16 query rounding.
  const auto output = attention.run(torch::zeros(
      {1, 12, 2, 3}, torch::TensorOptions().device(device).dtype(dtype)));
  output.to(torch::kFloat32).square().mean().backward();
  const auto expected_output =
      tensor(dtype == torch::kFloat32 ? r::fp32_output : r::fp16_output,
             {1, 12, 2, 3});
  const auto expected_gradient =
      tensor(dtype == torch::kFloat32 ? r::fp32_gradient : r::fp16_gradient,
             {1, 24, 2, 3});
  const auto rtol = dtype == torch::kFloat32 ? 2e-5 : 2e-4;
  const auto atol = dtype == torch::kFloat32 ? 2e-6 : 2e-5;
  EXPECT_TRUE(torch::allclose(output, expected_output, rtol, atol))
      << (output - expected_output).abs().max().item<double>();
  EXPECT_TRUE(
      torch::allclose(projection->value.grad(), expected_gradient, rtol, atol))
      << (projection->value.grad() - expected_gradient)
             .abs()
             .max()
             .item<double>();
}

TEST(VisionModelGraph, SpatialAttentionCpuMatchesOfficialOutputAndGradients) {
  check_spatial_attention_reference(Device(torch::kCPU), torch::kFloat32);
  check_spatial_attention_reference(Device(torch::kCPU), torch::kFloat16);
}

TEST(VisionModelGraph, SpatialAttentionCudaMatchesOfficialOutputAndGradients) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA device unavailable";
  check_spatial_attention_reference(Device(torch::kCUDA, 0), torch::kFloat32);
  check_spatial_attention_reference(Device(torch::kCUDA, 0), torch::kFloat16);
}

TEST(VisionModelGraph, SpatialAttentionRevisionPreservesUnaffectedFamilies) {
  namespace y = models::yolo;
  for (const auto family : {y::Family::v8, y::Family::yolo11, y::Family::yolo12,
                            y::Family::yolo26}) {
    const auto definition = y::describe_model({.family = family});
    EXPECT_TRUE(definition.revision.starts_with(
        family == y::Family::yolo11 || family == y::Family::yolo26
            ? "native-yolo-graph/v3/"
        : family == y::Family::v8 ? "native-yolo-graph/v2/"
                                  : "native-yolo-graph/v1/"));
  }
}

c10::IValue on_device(const c10::IValue& value, const Device& device) {
  if (value.isTensor()) return value.toTensor().to(device);
  if (value.isTuple()) {
    std::vector<c10::IValue> elements;
    for (const auto& element : value.toTupleRef().elements())
      elements.push_back(on_device(element, device));
    return c10::ivalue::Tuple::create(std::move(elements));
  }
  throw std::runtime_error("Unexpected task fixture target type");
}

void train(const t::CapabilityRecord& entry, const Device& device,
           c10::ScalarType dtype) {
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
  recipe.device = device;
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
  binding.model->state().to(device);
  for (auto& value : invocation.positional) value = on_device(value, device);
  target = on_device(*target, device);
  binding.model->state().train();
  data::BatchEnvelope envelope;
  training::StepContext context;
  training::StepOutput output;
  {
    std::unique_ptr<training::amp::AutocastGuard> autocast;
    if (dtype != torch::kFloat32)
      autocast = std::make_unique<training::amp::AutocastGuard>(
          c10::DeviceType::CUDA, dtype);
    output =
        binding.step({*binding.model, envelope, invocation, target, context});
  }
  ASSERT_TRUE(output.loss.defined());
  ASSERT_EQ(output.loss.numel(), 1);
  ASSERT_EQ(output.loss.device(), device);
  ASSERT_EQ(output.sample_count, image.size(0));
  ASSERT_TRUE(torch::isfinite(output.loss).item<bool>());
  training::amp::GradScaler scaler(
      {.init_scale = 8, .enabled = dtype == torch::kFloat16});
  scaler.scale(output.loss).backward();
  std::vector<optim::OptimizerParameterBinding> parameters;
  std::vector<std::pair<Tensor, Tensor>> frozen;
  for (const auto& item : binding.model->state().named_parameters())
    if (item.value().requires_grad())
      parameters.push_back(
          {.parameter_id = item.key(), .parameter = item.value()});
    else
      frozen.emplace_back(item.value(), item.value().detach().clone());
  auto optimizer = std::make_shared<optim::SgdOptimizerAdapter>(
      "vision.optimizer", std::vector<optim::SgdParameterGroup>{
                              {.group_id = "main",
                               .parameters = std::move(parameters),
                               .options = {.learning_rate = .001}}});
  scaler.unscale(optimizer);
  ASSERT_TRUE(scaler.cycle_allowed());
  Tensor strongest;
  double maximum = 0;
  for (const auto& parameter : binding.model->state().parameters()) {
    ASSERT_EQ(parameter.device(), device);
    ASSERT_EQ(parameter.scalar_type(), torch::kFloat32);
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
  optimizer->step();
  scaler.mark_step(optimizer, true);
  scaler.update();
  EXPECT_EQ(optimizer->completed_steps(), 1U);
  EXPECT_FALSE(torch::equal(before, strongest));
  for (const auto& [parameter, original] : frozen) {
    EXPECT_FALSE(parameter.grad().defined());
    EXPECT_TRUE(torch::equal(parameter, original));
  }
}

TEST_P(VisionModelTraining, CpuLossBackwardAndActualUpdate) {
  ASSERT_NO_FATAL_FAILURE(
      train(GetParam(), Device(torch::kCPU), torch::kFloat32));
}

struct AcceleratorCase {
  t::CapabilityRecord capability;
  c10::ScalarType dtype;
};
void PrintTo(const AcceleratorCase& value, std::ostream* stream) {
  *stream << value.capability.architecture << '_'
          << t::task_name(value.capability.task) << '_'
          << value.capability.variant << '_'
          << (value.dtype == torch::kFloat32   ? "FP32"
              : value.dtype == torch::kFloat16 ? "FP16"
                                               : "BF16");
}
std::vector<AcceleratorCase> accelerator_cases() {
  std::vector<AcceleratorCase> result;
  for (const auto& entry : t::vision_catalog()) {
    const bool selected =
        entry.variant == "p5" &&
        (entry.architecture == "resnet18" || entry.architecture == "yolo26n" ||
         entry.architecture == "fcos_resnet50_fpn" ||
         entry.architecture == "fcos_vit_adapter_tiny_fpn" ||
         entry.architecture == "rtdetr_resnet18_vd" ||
         entry.architecture == "rtdetr_resnet18_vd_coco72" ||
         entry.architecture == "fcn_resnet50" ||
         entry.architecture == "depth_resnet50_fpn" ||
         entry.architecture == "simmim_vit_tiny" ||
         entry.architecture == "reid_resnet50");
    if (selected)
      for (const auto dtype :
           {torch::kFloat32, torch::kFloat16, torch::kBFloat16})
        result.push_back({entry, dtype});
  }
  return result;
}
TEST(VisionRtDetrProfile,
     R18CocoTopologyHasThreeDecodersAndTrainableBackboneNorm) {
  torch::set_num_threads(1);
  namespace ex = models::extended;
  ex::RtDetr model(
      ex::RtDetrOptions{.backbone = ex::RtDetrBackbone::resnet18_vd,
                        .num_classes = 3,
                        .profile = ex::RtDetrProfile::r18_coco72});
  const auto parameters = model->named_parameters();
  const auto buffers = model->named_buffers();
  EXPECT_EQ(parameters.size() + buffers.size(), 533);
  std::int64_t count = 0;
  for (const auto& parameter : parameters) count += parameter.value().numel();
  EXPECT_EQ(count, 20085596);
  EXPECT_TRUE(parameters.contains("model.backbone.stem.0.norm.weight"));
  EXPECT_TRUE(parameters["model.backbone.stem.0.norm.weight"].requires_grad());
  EXPECT_TRUE(
      buffers.contains("model.backbone.stem.0.norm.num_batches_tracked"));
  EXPECT_TRUE(parameters.contains(
      "model.decoder2.self_attention.packed.in_proj_weight"));
  EXPECT_FALSE(parameters.contains(
      "model.decoder3.self_attention.packed.in_proj_weight"));
  EXPECT_EQ(parameters["model.encoder.fpn0.conv1.conv.weight"].size(0), 128);
  EXPECT_TRUE(
      parameters.contains("model.encoder.fpn0.blocks.2.conv3.conv.weight"));
  t::ModelSpec spec;
  spec.architecture = "rtdetr_resnet18_vd_coco72";
  const auto profile = t::describe_binding(spec);
  spec.architecture = "rtdetr_resnet18_vd";
  EXPECT_NE(profile.revision, t::describe_binding(spec).revision);
  EXPECT_FALSE(profile.reference_qualified);
  EXPECT_FALSE(profile.quality_qualified);
}

TEST(VisionRtDetrProfile, RejectsInvalidBackboneAndUnknownProfile) {
  namespace ex = models::extended;
  ex::RtDetrOptions options;
  options.profile = ex::RtDetrProfile::r18_coco72;
  EXPECT_THROW(ex::RtDetr model(options), InvalidArgument);
  options.backbone = ex::RtDetrBackbone::resnet18_vd;
  options.profile = static_cast<ex::RtDetrProfile>(99);
  EXPECT_THROW(ex::RtDetr model(options), InvalidArgument);
}

TEST(VisionRtDetrProfile, ExplicitEncoderBudgetControlsActualForward) {
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  namespace ex = models::extended;
  ex::RtDetrOptions options;
  options.backbone = ex::RtDetrBackbone::resnet18_vd;
  options.profile = ex::RtDetrProfile::r18_coco72;
  options.max_encoder_elements = 1024;
  ex::RtDetr limited(options);
  limited->eval();
  const auto images = torch::zeros({2, 3, 256, 256});
  const auto targets = torch::empty({0, 6});
  EXPECT_THROW(limited->forward(images, targets), ResourceExhausted);
  options.max_encoder_elements = 1048576;
  ex::RtDetr allowed(options);
  allowed->eval();
  const auto result = allowed->forward(images, targets);
  ASSERT_EQ(result.size(), 8U);
  EXPECT_EQ(result[0].size(1), 2);
  EXPECT_TRUE(torch::isfinite(result[0]).all().item<bool>());
  for (const auto invalid :
       std::initializer_list<std::int64_t>{0, -1, (1LL << 30) + 1}) {
    options.max_encoder_elements = invalid;
    EXPECT_THROW(ex::RtDetr model(options), InvalidArgument);
  }
}

class VisionAcceleratorTraining
    : public ::testing::TestWithParam<AcceleratorCase> {};
TEST_P(VisionAcceleratorTraining, FiniteTaskLossGradientsAndActualUpdate) {
  if (!torch::cuda::is_available())
    GTEST_SKIP() << "No CUDA-compatible GPU available";
  const auto& test = GetParam();
  ASSERT_NO_FATAL_FAILURE(
      train(test.capability, Device(torch::kCUDA, 0), test.dtype));
}
INSTANTIATE_TEST_SUITE_P(RepresentativeTasks, VisionAcceleratorTraining,
                         ::testing::ValuesIn(accelerator_cases()),
                         [](const auto& info) {
                           return ::testing::PrintToString(info.param);
                         });

INSTANTIATE_TEST_SUITE_P(InstalledCatalog, VisionModelTraining,
                         ::testing::ValuesIn(t::vision_catalog()),
                         [](const auto& info) {
                           return info.param.architecture + "_" +
                                  std::string(t::task_name(info.param.task)) +
                                  "_" + info.param.variant;
                         });
}  // namespace
}  // namespace jyppx::liuding::vision::tests
