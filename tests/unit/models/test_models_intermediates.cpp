// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/models/models.hpp>

#include <torch/optim.h>
#include <torch/torch.h>

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;
constexpr std::array<const char*, 2> kNames{"vit_small_patch14_dinov2",
                                            "dinov3_vits16"};

Tensor Images(bool v2) {
  return torch::linspace(-1.0, 1.0, 2 * 3 * (v2 ? 28 : 35) * (v2 ? 42 : 50))
      .view({2, 3, v2 ? 28 : 35, v2 ? 42 : 50});
}

std::vector<Tensor> Invoke(nn::ModelAdapter& adapter, const Tensor& images) {
  const auto values = adapter.invoke({.positional = {images}}).value.toTuple();
  std::vector<Tensor> outputs;
  for (const auto& value : values->elements())
    outputs.push_back(value.toTensor());
  return outputs;
}

TEST(ModelsDinoIntermediates, ExplicitSelectionsRejectBeforeTensorWork) {
  auto model = create_model(kNames[0]);
  for (const auto& selection : std::vector<DinoIntermediateOptions>{
           {.block_indices = {}},
           {.block_indices = {1, 1}},
           {.block_indices = {12}},
           {.block_indices = {0}, .out_indices = {}},
           {.block_indices = {0}, .out_indices = {2, 2}},
           {.block_indices = {0}, .out_indices = {4}},
           {.block_indices = {std::numeric_limits<std::size_t>::max()}}}) {
    EXPECT_THROW(model_intermediate_feature_spec(model, selection),
                 InvalidArgument);
    EXPECT_THROW(create_intermediate_feature_extractor(model, selection),
                 InvalidArgument);
  }
  torch::manual_seed(267);
  const auto next = torch::rand({4});
  torch::manual_seed(267);
  const auto spec =
      model_intermediate_feature_spec(model, {.block_indices = {8, 2}});
  ASSERT_EQ(spec.outputs.size(), 2U);
  EXPECT_EQ(spec.outputs[0].name, "block_8_patch_grid");
  EXPECT_TRUE(torch::equal(next, torch::rand({4})));
}

TEST(ModelsDinoIntermediates, BlockAndViewOrderPreserveStateModeAndAliasing) {
  for (std::size_t family = 0; family < kNames.size(); ++family) {
    const auto v2 = family == 0;
    auto model = create_model(kNames[family]);
    model.state().eval();
    const auto original = model.state().shared_module();
    const auto parameters = model.state().named_parameters();
    const DinoIntermediateOptions selection{.block_indices = {8, 2},
                                            .out_indices = {3, 1, 2, 0}};
    const auto spec = model_intermediate_feature_spec(model, selection);
    auto extractor = create_intermediate_feature_extractor(model, selection);
    EXPECT_EQ(extractor.state().shared_module(), original);
    EXPECT_FALSE(extractor.state().is_training());
    EXPECT_EQ(extractor.state().named_parameters().size(), parameters.size());
    const auto shared_parameters = extractor.state().named_parameters();
    for (const auto& parameter : parameters)
      EXPECT_EQ(shared_parameters[parameter.key()].unsafeGetTensorImpl(),
                parameter.value().unsafeGetTensorImpl());
    torch::NoGradGuard guard;
    const auto outputs = Invoke(extractor, Images(v2));
    ASSERT_EQ(outputs.size(), 8U);
    EXPECT_EQ(spec.outputs[0].name, "block_8_tokens");
    EXPECT_EQ(spec.outputs[4].name, "block_2_tokens");
    for (std::size_t base = 0; base < outputs.size(); base += 4) {
      EXPECT_EQ(outputs[base].sizes(),
                torch::IntArrayRef({2, v2 ? 7 : 11, 384}));
      EXPECT_EQ(outputs[base + 1].sizes(), torch::IntArrayRef({2, 6, 384}));
      EXPECT_EQ(outputs[base + 2].sizes(), torch::IntArrayRef({2, 384, 2, 3}));
      EXPECT_EQ(outputs[base + 3].sizes(), torch::IntArrayRef({2, 384}));
      for (std::size_t view = 1; view < 4; ++view)
        EXPECT_TRUE(outputs[base].is_alias_of(outputs[base + view]));
      EXPECT_TRUE(torch::isfinite(outputs[base]).all().item<bool>());
      EXPECT_FALSE(outputs[base].requires_grad());
    }
    extractor.state().train();
    EXPECT_TRUE(model.state().is_training());
  }
}

TEST(ModelsDinoIntermediates,
     EachLayerUsesFamilyNormWithoutChangingTheBlockStream) {
  for (const auto* name : kNames) {
    auto model = create_model(name);
    model.state().eval();
    const auto images = Images(std::string(name) == kNames[0]);
    auto raw = create_intermediate_feature_extractor(
        model,
        {.block_indices = {8, 2}, .out_indices = {3}, .normalize = false});
    auto normalized = create_intermediate_feature_extractor(
        model, {.block_indices = {8, 2}, .out_indices = {3}});
    const auto norm = model.state()
                          .module()
                          .named_modules()["norm"]
                          ->as<torch::nn::LayerNorm>();
    torch::NoGradGuard guard;
    const auto raw_outputs = Invoke(raw, images);
    const auto norm_outputs = Invoke(normalized, images);
    ASSERT_EQ(raw_outputs.size(), 2U);
    for (std::size_t index = 0; index < raw_outputs.size(); ++index)
      EXPECT_TRUE(
          torch::equal(norm->forward(raw_outputs[index]), norm_outputs[index]));
    auto last = create_intermediate_feature_extractor(
        model, {.block_indices = {11}, .out_indices = {0}});
    EXPECT_TRUE(
        torch::equal(Invoke(last, images)[0],
                     model.invoke({.positional = {images}}).value.toTensor()));
  }
}

TEST(ModelsDinoIntermediates,
     EarlyStopSkipsPoisonedLaterBlocksAndPreservesGradients) {
  for (const auto* name : kNames) {
    ModelOptions options;
    options.dinov2.num_classes = 3;
    options.dinov3.num_classes = 3;
    auto model = create_model(name, options);
    model.state().eval();
    auto extractor = create_intermediate_feature_extractor(
        model, {.block_indices = {2}, .out_indices = {2}});
    auto parameters = model.state().named_parameters();
    auto patch = parameters["patch_embed.proj.weight"];
    const auto before = patch.detach().clone();
    {
      torch::NoGradGuard guard;
      parameters["blocks.3.norm1.weight"].fill_(
          std::numeric_limits<float>::quiet_NaN());
    }
    auto images = Images(std::string(name) == kNames[0]).requires_grad_(true);
    const auto features = Invoke(extractor, images).front();
    ASSERT_TRUE(torch::isfinite(features).all().item<bool>());
    const auto coefficients =
        torch::linspace(-0.3, 0.7, features.numel()).view(features.sizes());
    (features * coefficients).mean().backward();
    ASSERT_TRUE(images.grad().defined());
    ASSERT_TRUE(patch.grad().defined());
    EXPECT_TRUE(torch::isfinite(images.grad()).all().item<bool>());
    EXPECT_GT(images.grad().abs().sum().item<double>(), 0.0);
    EXPECT_TRUE(torch::isfinite(patch.grad()).all().item<bool>());
    EXPECT_GT(patch.grad().abs().sum().item<double>(), 0.0);
    EXPECT_FALSE(parameters["blocks.3.norm1.weight"].grad().defined());
    EXPECT_FALSE(parameters["blocks.11.norm1.weight"].grad().defined());
    EXPECT_FALSE(parameters["head.weight"].grad().defined());
    torch::optim::SGD optimizer(model.state().parameters(),
                                torch::optim::SGDOptions(0.01));
    optimizer.step();
    EXPECT_GT((patch - before).abs().sum().item<double>(), 0.0);
  }
}

TEST(ModelsDinoIntermediates,
     ConcreteMethodsMatchSharedAdaptersAndValidateImages) {
  auto v2 = dinov2_vits14();
  auto v3 = dinov3_vits16();
  v2->eval();
  v3->eval();
  const DinoIntermediateOptions selection{.block_indices = {0},
                                          .out_indices = {0, 2}};
  auto v2_adapter = create_intermediate_feature_extractor(adapt(v2), selection);
  auto v3_adapter = create_intermediate_feature_extractor(adapt(v3), selection);
  torch::NoGradGuard guard;
  const auto direct_v2 = v2->forward_intermediates(Images(true), selection);
  const auto direct_v3 = v3->forward_intermediates(Images(false), selection);
  const auto adapter_v2 = Invoke(v2_adapter, Images(true));
  const auto adapter_v3 = Invoke(v3_adapter, Images(false));
  for (std::size_t index = 0; index < direct_v2.size(); ++index) {
    EXPECT_TRUE(torch::equal(direct_v2[index], adapter_v2[index]));
    EXPECT_TRUE(torch::equal(direct_v3[index], adapter_v3[index]));
  }
  EXPECT_THROW(Invoke(v2_adapter, torch::zeros({1, 3, 29, 42})),
               InvalidArgument);
  EXPECT_THROW(Invoke(v3_adapter, torch::zeros({1, 3, 15, 32})),
               InvalidArgument);
  EXPECT_THROW(Invoke(v2_adapter, torch::zeros({1, 1, 28, 42})),
               InvalidArgument);
  try {
    static_cast<void>(v2->forward_intermediates({}, {.block_indices = {12}}));
    FAIL() << "Invalid selection was accepted";
  } catch (const InvalidArgument& error) {
    EXPECT_EQ(error.details().operation, "dino.intermediates.blocks");
  }
  try {
    static_cast<void>(v3->forward_intermediates({}, {.block_indices = {12}}));
    FAIL() << "Invalid selection was accepted";
  } catch (const InvalidArgument& error) {
    EXPECT_EQ(error.details().operation, "dino.intermediates.blocks");
  }
}

TEST(ModelsDinoIntermediates, UnsupportedModuleHasAnExplicitBoundary) {
  auto cnn = create_model("resnet18");
  EXPECT_THROW(model_intermediate_feature_spec(cnn, {.block_indices = {0}}),
               NotImplemented);
  EXPECT_THROW(
      create_intermediate_feature_extractor(cnn, {.block_indices = {0}}),
      NotImplemented);
}
}  // namespace
