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
    const auto dino = entry.family.starts_with("dinov");
    EXPECT_EQ(entry.feature_input.has_value(), dino) << entry.name;
    EXPECT_EQ(entry.features.empty(), !dino) << entry.name;
    if (!dino) continue;
    ++feature_entries;
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
  EXPECT_EQ(feature_entries, 15U);
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
  EXPECT_THROW(create_feature_extractor("resnet18"), NotImplemented);
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
}  // namespace
