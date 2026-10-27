// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/models/models.hpp>

#include <torch/optim.h>
#include <torch/torch.h>

#include <array>
#include <gtest/gtest.h>
#include <iostream>
#include <locale>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;

ModelInfo DummyInfo() {
  return {.name = "caller_features",
          .family = "test",
          .task = "image_features",
          .reference = "caller-owned deterministic fixture",
          .features = {{.name = "spatial",
                        .layout = FeatureLayout::nchw,
                        .channels = 3,
                        .reduction = 1},
                       {.name = "global",
                        .layout = FeatureLayout::nc,
                        .channels = 3}},
          .feature_input = FeatureInputSpec{.spatial_multiple = 2,
                                            .crop_to_multiple = true}};
}

nn::ModelAdapter DummyAdapter(std::shared_ptr<int> calls = {},
                              bool tuple = true) {
  auto module = std::make_shared<torch::nn::Module>();
  return nn::adapt_module(
      module,
      [calls, tuple](const nn::Invocation& invocation) -> c10::IValue {
        if (calls) ++*calls;
        const auto images = invocation.positional.front().toTensor();
        if (!tuple) return images;
        return c10::ivalue::Tuple::create({images, images.mean({2, 3})});
      },
      {.positional = {ValueSchema::tensor({.rank = 4}, "images")}},
      tuple ? ValueSchema::tuple({ValueSchema::tensor({.rank = 4}),
                                  ValueSchema::tensor({.rank = 2})})
            : ValueSchema::tensor({.rank = 4}));
}

TEST(ModelsFeatures, CatalogMetadataDoesNotConstructModelsOrConsumeRng) {
  torch::manual_seed(8128);
  const auto expected_next_random = torch::rand({8});
  torch::manual_seed(8128);
  const auto entries = list_models();
  EXPECT_EQ(entries.size(), 34U);
  std::size_t feature_entries = 0;
  for (const auto& entry : entries) {
    const auto image = entry.name != "transformer_encoder";
    EXPECT_EQ(entry.feature_input.has_value(), image) << entry.name;
    EXPECT_EQ(entry.features.empty(), !image) << entry.name;
    if (!image) continue;
    ++feature_entries;
    const auto spec = model_feature_spec(entry.name);
    ASSERT_EQ(spec.outputs.size(), entry.features.size());
    for (std::size_t i = 0; i < spec.outputs.size(); ++i) {
      EXPECT_EQ(spec.outputs[i].name, entry.features[i].name);
      EXPECT_EQ(spec.outputs[i].channels, entry.features[i].channels);
    }
    if (!entry.family.starts_with("dinov")) continue;
    ASSERT_EQ(entry.features.size(), 4U);
    if (entry.family == "dinov3_convnext") {
      EXPECT_EQ(entry.feature_input->spatial_multiple, 32);
      EXPECT_TRUE(entry.feature_input->crop_to_multiple);
      EXPECT_EQ(entry.features.back().channels,
                entry.features.front().channels * 8);
      EXPECT_EQ(entry.features.back().reduction, 32);
    } else {
      const auto v2 = entry.family == "dinov2";
      EXPECT_EQ(entry.feature_input->spatial_multiple, v2 ? 14 : 16);
      EXPECT_EQ(entry.feature_input->crop_to_multiple, !v2);
      EXPECT_EQ(entry.features[0].layout, FeatureLayout::nc);
      EXPECT_EQ(entry.features[1].layout, FeatureLayout::ntc);
      EXPECT_EQ(entry.features[2].layout, FeatureLayout::nchw);
      EXPECT_EQ(entry.features[3].prefix_tokens, v2 ? 1 : 5);
    }
  }
  EXPECT_EQ(feature_entries, 33U);
  EXPECT_EQ(model_info("dinov3_vith16plus").features[0].channels, 1280);
  EXPECT_EQ(model_info("dinov3_vit7b16").features[0].channels, 4096);
  EXPECT_EQ(model_info("vit_giant_patch14_dinov2").features[0].channels, 1536);
  EXPECT_TRUE(torch::equal(expected_next_random, torch::rand({8})));
  std::cout << "FEATURE_MANIFEST " << models_preview_manifest_json() << '\n';
}

TEST(ModelsFeatures,
     RegistryRejectsInconsistentFeatureContractsBeforeConstruction) {
  const ModelFactory normal = [](const ModelOptions&) {
    return DummyAdapter();
  };
  const auto good = DummyInfo();
  EXPECT_THROW(ModelRegistry().add(good, normal), InvalidArgument);
  auto info = good;
  info.features.clear();
  info.feature_input.reset();
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.features[1].name = info.features[0].name;
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.features[0].channels = 0;
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.features[0].prefix_tokens = 1;
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.features[1].reduction = 1;
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.features[0].layout = static_cast<FeatureLayout>(255);
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.feature_input->spatial_multiple = 0;
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.feature_input->minimum_spatial = 0;
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.feature_input->fixed_size = {4, 0};
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
  info = good;
  info.feature_input->crop_to_multiple = false;
  info.feature_input->fixed_size = {3, 4};
  EXPECT_THROW(ModelRegistry().add(info, normal, normal), InvalidArgument);
}

TEST(ModelsFeatures, InvalidSelectionsFailBeforeFactoryAndNeverConstructGiant) {
  auto constructions = std::make_shared<int>(0);
  ModelRegistry registry;
  registry.add(
      DummyInfo(), [](const ModelOptions&) { return DummyAdapter(); },
      [constructions](const ModelOptions&) {
        ++*constructions;
        return DummyAdapter();
      });
  EXPECT_THROW(registry.create_features("caller_features", {}, {0, 0}),
               InvalidArgument);
  EXPECT_THROW(registry.create_features("caller_features", {}, {2}),
               InvalidArgument);
  EXPECT_THROW(registry.create_features("caller_features", {}, {0, 1, 0}),
               InvalidArgument);
  EXPECT_EQ(*constructions, 0);
  EXPECT_THROW(create_feature_extractor("vit_giant_patch14_dinov2", {}, {4}),
               InvalidArgument);
  EXPECT_THROW(create_feature_extractor("transformer_encoder"), NotImplemented);
  EXPECT_THROW(model_feature_spec("transformer_encoder"), NotImplemented);
  EXPECT_THROW(create_feature_extractor("missing"), InvalidArgument);
}

TEST(ModelsFeatures, SelectionPreservesOrderStorageAndRunsFactoryForwardOnce) {
  auto calls = std::make_shared<int>(0);
  ModelRegistry registry;
  registry.add(
      DummyInfo(), [](const ModelOptions&) { return DummyAdapter(); },
      [calls](const ModelOptions&) { return DummyAdapter(calls); });
  auto extractor = registry.create_features("caller_features", {}, {1, 0});
  EXPECT_EQ(extractor.output_schema().elements[0].name, "global");
  EXPECT_EQ(extractor.output_schema().elements[1].name, "spatial");
  auto input = torch::arange(3 * 3 * 5, torch::kFloat32).reshape({1, 3, 3, 5});
  const auto output = extractor.invoke({.positional = {input}}).value.toTuple();
  const auto cropped = input.narrow(2, 0, 2).narrow(3, 0, 4);
  ASSERT_EQ(output->elements().size(), 2U);
  EXPECT_TRUE(
      torch::equal(output->elements()[0].toTensor(), cropped.mean({2, 3})));
  EXPECT_TRUE(torch::equal(output->elements()[1].toTensor(), cropped));
  EXPECT_TRUE(output->elements()[1].toTensor().is_alias_of(input));
  EXPECT_EQ(*calls, 1);
  EXPECT_THROW(extractor.invoke({.positional = {torch::zeros({1, 4, 2, 2})}}),
               InvalidArgument);
  EXPECT_THROW(extractor.invoke({.positional = {torch::zeros({1, 3, 1, 2})}}),
               InvalidArgument);
  EXPECT_THROW(extractor.invoke({.positional = {torch::zeros({1, 3, 2})}}),
               InvalidArgument);
}

TEST(ModelsFeatures, BadFactoryOutputContractIsRejected) {
  ModelRegistry registry;
  registry.add(
      DummyInfo(), [](const ModelOptions&) { return DummyAdapter(); },
      [](const ModelOptions&) { return DummyAdapter({}, false); });
  EXPECT_THROW(registry.create_features("caller_features"), InvalidArgument);
}

class GroupedNumbers final : public std::numpunct<char> {
 protected:
  char do_thousands_sep() const override { return ','; }
  std::string do_grouping() const override { return "\3"; }
};

TEST(ModelsFeatures, ManifestNumbersAreIndependentOfTheGlobalLocale) {
  const auto previous = std::locale();
  struct RestoreLocale {
    std::locale previous;
    ~RestoreLocale() { std::locale::global(previous); }
  } restore{previous};
  std::locale::global(std::locale(previous, new GroupedNumbers));
  const auto manifest = models_preview_manifest_json();
  EXPECT_NE(manifest.find("\"channels\":4096"), std::string::npos);
  EXPECT_EQ(manifest.find("\"channels\":4,096"), std::string::npos);
}

struct ExecutionCase {
  const char* name;
  bool cuda;
};

void PrintTo(const ExecutionCase& scenario, std::ostream* stream) {
  *stream << scenario.name << (scenario.cuda ? "_Cuda" : "_Cpu");
}

class ModelsFeaturesExecution : public ::testing::TestWithParam<ExecutionCase> {
};

TEST_P(ModelsFeaturesExecution,
       LayoutViewsAutogradAndOriginalStateArePreserved) {
  const auto scenario = GetParam();
  if (scenario.cuda && !torch::cuda::is_available())
    GTEST_SKIP() << "CUDA unavailable";
  torch::manual_seed(65537);
  auto extractor = create_feature_extractor(scenario.name);
  const auto device = torch::Device(scenario.cuda ? torch::kCUDA : torch::kCPU);
  extractor.state().to(device);
  extractor.state().eval();
  EXPECT_FALSE(extractor.state().shared_module()->is_training());
  const auto info = model_info(scenario.name);
  EXPECT_EQ(parameter_count(extractor.state().module()),
            info.default_parameters);
  const auto convnext = info.family == "dinov3_convnext";
  const auto v2 = info.family == "dinov2";
  const auto height = v2 ? 28 : 35;
  const auto width = v2 ? 42 : convnext ? 66 : 50;
  auto images = torch::randn({2, 3, height, width},
                             torch::TensorOptions().device(device));
  const auto values =
      extractor.invoke({.positional = {images}}).value.toTuple();
  ASSERT_EQ(values->elements().size(), 4U);
  std::vector<Tensor> expected;
  {
    torch::NoGradGuard no_grad;
    if (convnext) {
      const auto concrete = std::dynamic_pointer_cast<DinoV3ConvNeXtImpl>(
          extractor.state().shared_module());
      ASSERT_TRUE(concrete);
      expected = concrete->forward_features(images);
    } else {
      Tensor tokens;
      if (v2) {
        const auto concrete = std::dynamic_pointer_cast<DinoV2Impl>(
            extractor.state().shared_module());
        ASSERT_TRUE(concrete);
        tokens = concrete->forward_features(images);
      } else {
        const auto concrete = std::dynamic_pointer_cast<DinoV3Impl>(
            extractor.state().shared_module());
        ASSERT_TRUE(concrete);
        tokens = concrete->forward_features(images);
      }
      const auto prefix = v2 ? 1 : 5;
      const auto patch = v2 ? 14 : 16;
      const auto patches = tokens.slice(1, prefix);
      expected = {
          tokens.select(1, 0), patches,
          patches.view({2, height / patch, width / patch, tokens.size(2)})
              .permute({0, 3, 1, 2}),
          tokens};
    }
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto feature = values->elements()[index].toTensor();
    EXPECT_EQ(feature.device(), images.device());
    EXPECT_EQ(feature.scalar_type(), torch::kFloat32);
    EXPECT_TRUE(feature.requires_grad());
    EXPECT_TRUE(torch::equal(feature, expected[index]))
        << info.features[index].name;
  }
  if (!convnext) {
    const auto tokens = values->elements()[3].toTensor();
    for (std::size_t index = 0; index < 3; ++index)
      EXPECT_TRUE(values->elements()[index].toTensor().is_alias_of(tokens));
    EXPECT_EQ(values->elements()[1].toTensor().size(1), 6);
  }
  const auto key =
      convnext ? "downsample_layers.0.0.weight" : "patch_embed.proj.weight";
  const auto parameter = extractor.state().named_parameters()[key];
  const auto before = parameter.detach().clone();
  torch::optim::SGD optimizer(extractor.state().parameters(),
                              torch::optim::SGDOptions(0.01));
  auto loss = values->elements()[0].toTensor().select(1, 0).mean();
  for (const auto& value : values->elements())
    loss = loss + value.toTensor().flatten().slice(0, 0, 128).square().mean();
  loss.backward();
  ASSERT_TRUE(parameter.grad().defined());
  EXPECT_TRUE(torch::isfinite(parameter.grad()).all().item<bool>());
  EXPECT_GT(parameter.grad().abs().sum().item<double>(), 0.0);
  optimizer.step();
  EXPECT_FALSE(torch::equal(parameter, before));
  extractor.state().train();
  EXPECT_TRUE(extractor.state().shared_module()->is_training());
  EXPECT_THROW(
      extractor.invoke({.positional = {torch::zeros({1, 1, height, width},
                                                    images.options())}}),
      InvalidArgument);
  if (v2)
    EXPECT_THROW(extractor.invoke({.positional = {torch::zeros(
                                       {1, 3, 29, 42}, images.options())}}),
                 InvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(
    DINO, ModelsFeaturesExecution,
    ::testing::Values(ExecutionCase{"vit_small_patch14_dinov2", false},
                      ExecutionCase{"dinov3_vits16", false},
                      ExecutionCase{"dinov3_convnext_tiny", false},
                      ExecutionCase{"vit_small_patch14_dinov2", true},
                      ExecutionCase{"dinov3_vits16", true},
                      ExecutionCase{"dinov3_convnext_tiny", true}),
    [](const ::testing::TestParamInfo<ExecutionCase>& info) {
      return std::string(info.param.name) +
             (info.param.cuda ? "_Cuda" : "_Cpu");
    });

std::vector<Tensor> ConcreteFeatures(
    const std::shared_ptr<torch::nn::Module>& module, const Tensor& images,
    const std::string& family) {
  if (family == "resnet" || family == "resnext")
    return std::dynamic_pointer_cast<ResNetImpl>(module)->forward_features(
        images);
  if (family == "densenet")
    return std::dynamic_pointer_cast<DenseNetImpl>(module)->forward_features(
        images);
  if (family == "mobilenetv2")
    return std::dynamic_pointer_cast<MobileNetV2Impl>(module)->forward_features(
        images);
  if (family == "mobilenet")
    return {
        std::dynamic_pointer_cast<MobileNetV3Impl>(module)->forward_features(
            images)};
  if (family == "efficientnet")
    return std::dynamic_pointer_cast<EfficientNetImpl>(module)
        ->forward_features(images);
  if (family == "efficientnetv2")
    return std::dynamic_pointer_cast<EfficientNetV2Impl>(module)
        ->forward_features(images);
  if (family == "convnext")
    return std::dynamic_pointer_cast<ConvNeXtImpl>(module)->forward_features(
        images);
  if (family == "swin")
    return std::dynamic_pointer_cast<SwinTinyImpl>(module)->forward_features(
        images);
  const auto tokens =
      std::dynamic_pointer_cast<ViTImpl>(module)->forward_features(images);
  const auto patches = tokens.slice(1, 1);
  return {tokens.select(1, 0), patches,
          patches
              .view({images.size(0), images.size(2) / 16, images.size(3) / 16,
                     tokens.size(2)})
              .permute({0, 3, 1, 2}),
          tokens};
}

class VisionFeaturesExecution : public ::testing::TestWithParam<ExecutionCase> {
};

TEST_P(VisionFeaturesExecution, ConcreteOutputsGradientsAndOddSpatialRules) {
  const auto scenario = GetParam();
  if (scenario.cuda && !torch::cuda::is_available())
    GTEST_SKIP() << "CUDA unavailable";
  torch::set_num_threads(1);
  torch::manual_seed(257);
  ModelOptions options;
  options.vit.dynamic_image_size = true;
  options.swin.dynamic_image_size = true;
  const auto info = model_info(scenario.name);
  const auto spec = model_feature_spec(scenario.name, options);
  auto extractor = create_feature_extractor(scenario.name, options);
  const auto device = torch::Device(scenario.cuda ? torch::kCUDA : torch::kCPU);
  extractor.state().to(device);
  extractor.state().eval();
  EXPECT_EQ(parameter_count(extractor.state().module()),
            info.default_parameters);
  const auto height = info.family == "vit"    ? 32
                      : info.family == "swin" ? 36
                                              : 35;
  const auto width = info.family == "vit"    ? 48
                     : info.family == "swin" ? 52
                                             : 67;
  auto images = torch::randn({2, 3, height, width},
                             torch::TensorOptions().device(device));
  const auto values =
      extractor.invoke({.positional = {images}}).value.toTuple();
  ASSERT_EQ(values->elements().size(), spec.outputs.size());
  {
    torch::NoGradGuard no_grad;
    const auto expected = ConcreteFeatures(extractor.state().shared_module(),
                                           images, info.family);
    ASSERT_EQ(expected.size(), spec.outputs.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
      const auto actual = values->elements()[i].toTensor();
      EXPECT_TRUE(torch::equal(actual, expected[i])) << spec.outputs[i].name;
      EXPECT_EQ(actual.device(), images.device());
      EXPECT_EQ(actual.scalar_type(), images.scalar_type());
      EXPECT_EQ(
          actual.size(spec.outputs[i].layout == FeatureLayout::ntc ? 2 : 1),
          spec.outputs[i].channels);
      EXPECT_TRUE(actual.requires_grad());
      if (info.family == "vit") continue;
      const auto stride = spec.outputs[i].reduction;
      const auto floor = info.family == "convnext";
      const auto dense = info.family == "densenet";
      const auto expected_h = dense   ? ((height + 3) / 4) / (stride / 4)
                              : floor ? height / stride
                                      : (height + stride - 1) / stride;
      const auto expected_w = dense   ? ((width + 3) / 4) / (stride / 4)
                              : floor ? width / stride
                                      : (width + stride - 1) / stride;
      EXPECT_EQ(actual.size(2), expected_h);
      EXPECT_EQ(actual.size(3), expected_w);
    }
  }
  if (info.family == "vit") {
    const auto tokens = values->elements()[3].toTensor();
    EXPECT_EQ(tokens.size(1), 7);
    for (std::size_t i = 0; i < 3; ++i)
      EXPECT_TRUE(values->elements()[i].toTensor().is_alias_of(tokens));
  }
  const auto parameter = extractor.state().parameters().front();
  const auto before = parameter.detach().clone();
  torch::optim::SGD optimizer(extractor.state().parameters(),
                              torch::optim::SGDOptions(0.1));
  auto loss = values->elements()[0].toTensor().select(1, 0).mean();
  for (const auto& value : values->elements())
    loss = loss + value.toTensor().flatten().slice(0, 0, 128).square().mean();
  loss.backward();
  ASSERT_TRUE(parameter.grad().defined());
  EXPECT_TRUE(torch::isfinite(parameter.grad()).all().item<bool>());
  EXPECT_GT(parameter.grad().abs().sum().item<double>(), 0.0);
  optimizer.step();
  EXPECT_FALSE(torch::equal(before, parameter));
  extractor.state().train();
  EXPECT_TRUE(extractor.state().shared_module()->is_training());
}

std::vector<ExecutionCase> VisionCases() {
  std::vector<ExecutionCase> result;
  for (const auto name :
       {"resnet18", "resnet34", "resnet50", "resnet101", "resnext50_32x4d",
        "densenet121", "mobilenetv2_100", "mobilenet_v3_small",
        "mobilenet_v3_large", "efficientnet_b0", "efficientnet_b2",
        "tf_efficientnetv2_s", "convnext_tiny", "convnext_small",
        "vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224"}) {
    result.push_back({name, false});
    result.push_back({name, true});
  }
  return result;
}

INSTANTIATE_TEST_SUITE_P(
    Vision, VisionFeaturesExecution, ::testing::ValuesIn(VisionCases()),
    [](const ::testing::TestParamInfo<ExecutionCase>& info) {
      return std::string(info.param.name) +
             (info.param.cuda ? "_Cuda" : "_Cpu");
    });

TEST(ModelsFeatures,
     ConfiguredMetadataDoesNotAllocateModelsAndMatchesWidthRounding) {
  torch::manual_seed(8191);
  const auto expected = torch::rand({8});
  torch::manual_seed(8191);
  for (const auto name : {"mobilenet_v3_small", "mobilenet_v3_large"}) {
    ModelOptions options;
    options.mobile.width_multiplier = 0.37;
    const auto spec = model_feature_spec(name, options);
    EXPECT_EQ(spec.outputs[0].channels,
              std::string(name).ends_with("small") ? 192 : 336);
  }
  for (const auto name :
       {"vit_tiny_patch16_224", "vit_small_patch16_224", "vit_base_patch16_224",
        "swin_tiny_patch4_window7_224"}) {
    EXPECT_EQ(model_feature_spec(name).input.fixed_size,
              (std::array<std::int64_t, 2>{224, 224}));
    ModelOptions options;
    options.vit.dynamic_image_size = true;
    options.swin.dynamic_image_size = true;
    EXPECT_EQ(model_feature_spec(name, options).input.fixed_size,
              (std::array<std::int64_t, 2>{0, 0}));
  }
  EXPECT_TRUE(torch::equal(expected, torch::rand({8})));
  EXPECT_THROW(model_feature_spec("mobilenet_v3_small",
                                  {.mobile = {.width_multiplier = 0.2}}),
               InvalidArgument);
}

TEST(ModelsFeatures, ResolvedContractsRejectInvalidSizesBeforeForward) {
  auto calls = std::make_shared<int>(0);
  auto constructions = std::make_shared<int>(0);
  ModelRegistry registry;
  registry.add(
      DummyInfo(), [](const ModelOptions&) { return DummyAdapter(); },
      [calls, constructions](const ModelOptions&) {
        ++*constructions;
        return DummyAdapter(calls);
      },
      [](const ModelOptions& options) {
        return FeatureSpec{
            .outputs = DummyInfo().features,
            .input = {.minimum_spatial = options.resnet.num_classes,
                      .fixed_size = {4, 6}}};
      });
  EXPECT_THROW(registry.create_features("caller_features",
                                        {.resnet = {.num_classes = 5}}),
               InvalidArgument);
  EXPECT_EQ(*constructions, 0);
  auto extractor = registry.create_features("caller_features",
                                            {.resnet = {.num_classes = 4}});
  EXPECT_THROW(extractor.invoke({.positional = {torch::ones({1, 3, 3, 6})}}),
               InvalidArgument);
  EXPECT_THROW(extractor.invoke({.positional = {torch::ones({1, 3, 4, 8})}}),
               InvalidArgument);
  EXPECT_EQ(*calls, 0);
  EXPECT_NO_THROW(
      extractor.invoke({.positional = {torch::ones({1, 3, 4, 6})}}));
  EXPECT_EQ(*calls, 1);
}

TEST(ModelsFeatures, NondefaultMobileWidthAndInputChannelsUseResolvedSchema) {
  torch::set_num_threads(1);
  for (const auto name : {"mobilenet_v3_small", "mobilenet_v3_large"}) {
    ModelOptions options;
    options.mobile.input_channels = 1;
    options.mobile.width_multiplier = 0.37;
    const auto spec = model_feature_spec(name, options);
    auto extractor = create_feature_extractor(name, options);
    extractor.state().eval();
    extractor.state().to(torch::kFloat64);
    auto images = torch::randn({1, 1, 33, 49}, torch::kFloat64);
    const auto output = extractor.invoke({.positional = {images}})
                            .value.toTuple()
                            ->elements()[0]
                            .toTensor();
    EXPECT_EQ(output.size(1), spec.outputs[0].channels);
    EXPECT_EQ(output.scalar_type(), torch::kFloat64);
    EXPECT_TRUE(output.requires_grad());
    EXPECT_EQ(
        extractor.output_schema().elements[0].tensor_constraints->shape->at(1),
        spec.outputs[0].channels);
    EXPECT_THROW(extractor.invoke({.positional = {torch::zeros(
                                       {1, 3, 33, 49}, torch::kFloat64)}}),
                 InvalidArgument);
  }
}

TEST(ModelsFeatures, BuiltinMinimumAndFixedSpatialRulesAreEnforced) {
  torch::set_num_threads(1);
  for (const auto name :
       {"densenet121", "convnext_tiny", "vit_tiny_patch16_224",
        "swin_tiny_patch4_window7_224"}) {
    const auto spec = model_feature_spec(name);
    auto extractor = create_feature_extractor(name);
    extractor.state().eval();
    const auto fixed = spec.input.fixed_size[0];
    EXPECT_THROW(
        extractor.invoke(
            {.positional = {torch::zeros(
                 {1, 3, fixed ? 32 : spec.input.minimum_spatial - 1, 48})}}),
        InvalidArgument);
    const auto height = fixed ? fixed : spec.input.minimum_spatial;
    torch::NoGradGuard no_grad;
    EXPECT_NO_THROW(extractor.invoke(
        {.positional = {torch::zeros({1, 3, height, height})}}));
  }
}
}  // namespace
