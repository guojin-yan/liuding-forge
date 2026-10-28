// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/models/models.hpp>

#include <torch/optim.h>
#include <torch/torch.h>

#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;

bool IsHead(const std::string& key, const ClassifierInfo& info) {
  return key.starts_with(info.parameter_prefix + '.');
}

ModelOptions HeadlessOptions() {
  ModelOptions options;
  options.resnet.num_classes = 0;
  options.densenet.num_classes = 0;
  options.mobile_v2.num_classes = 0;
  options.mobile.num_classes = 0;
  options.efficientnet.num_classes = 0;
  options.efficientnet_v2.num_classes = 0;
  options.convnext.num_classes = 0;
  options.vit.num_classes = 0;
  options.vit.dynamic_image_size = true;
  options.swin.num_classes = 0;
  options.swin.dynamic_image_size = true;
  return options;
}

struct HeadCase {
  const char* name;
  bool cuda;
};
void PrintTo(const HeadCase& item, std::ostream* stream) {
  *stream << item.name << (item.cuda ? "_Cuda" : "_Cpu");
}
class ModelsHeadsExecution : public ::testing::TestWithParam<HeadCase> {};

TEST_P(ModelsHeadsExecution, ReplaceRemoveFreezeAndTrainOnOriginalBackbone) {
  const auto scenario = GetParam();
  if (scenario.cuda && !torch::cuda::is_available())
    GTEST_SKIP() << "CUDA unavailable";
  torch::set_num_threads(1);
  torch::manual_seed(51437);
  auto model = create_model(scenario.name, HeadlessOptions());
  const auto module = model.state().shared_module();
  const auto device = torch::Device(scenario.cuda ? torch::kCUDA : torch::kCPU);
  model.state().to(device);
  model.state().eval();
  const auto original = classifier_info(model);
  ASSERT_EQ(original.num_classes, 0);
  ASSERT_GT(original.embedding_features, 0);
  const auto parameters = model.state().named_parameters();
  const auto buffers = model.state().named_buffers();
  for (const auto& item : parameters) {
    EXPECT_FALSE(IsHead(item.key(), original));
    item.value().set_requires_grad(false);
  }
  const auto first = parameters.begin()->value();
  const auto snapshot = first.detach().clone();
  const auto family = model_info(scenario.name).family;
  const auto height = family == "dinov2" ? 28
                      : family == "vit"  ? 32
                      : family == "swin" ? 36
                                         : 35;
  const auto width = family == "dinov2" ? 42
                     : family == "vit"  ? 48
                     : family == "swin" ? 52
                                        : 67;
  auto images = torch::randn({2, 3, height, width},
                             torch::TensorOptions().device(device));
  Tensor embedding;
  {
    torch::NoGradGuard no_grad;
    embedding = model.invoke({.positional = {images}}).value.toTensor();
  }
  EXPECT_EQ(embedding.sizes(),
            torch::IntArrayRef({2, original.embedding_features}));
  torch::manual_seed(19);
  const auto next = torch::rand({8});
  torch::manual_seed(19);
  EXPECT_THROW(reset_classifier(model, -1), InvalidArgument);
  EXPECT_THROW(reset_classifier(model, 1'000'001), InvalidArgument);
  EXPECT_EQ(classifier_info(model).num_classes, 0);
  EXPECT_TRUE(torch::equal(torch::rand({8}), next));

  reset_classifier(model, 3);
  EXPECT_EQ(model.state().shared_module(), module);
  EXPECT_EQ(classifier_info(model).num_classes, 3);
  EXPECT_EQ(model.output_schema().tensor_constraints->shape->at(1), 3);
  for (const auto& child : module->modules(false))
    EXPECT_FALSE(child->is_training());
  for (const auto& item : parameters) {
    const auto current = model.state().named_parameters()[item.key()];
    EXPECT_EQ(current.unsafeGetTensorImpl(),
              item.value().unsafeGetTensorImpl());
    EXPECT_FALSE(current.requires_grad());
  }
  for (const auto& item : buffers)
    EXPECT_EQ(model.state().named_buffers()[item.key()].unsafeGetTensorImpl(),
              item.value().unsafeGetTensorImpl());
  {
    torch::NoGradGuard no_grad;
    const auto logits = model.invoke({.positional = {images}}).value.toTensor();
    EXPECT_EQ(logits.sizes(), torch::IntArrayRef({2, 3}));
    EXPECT_EQ(logits.device(), images.device());
    if (original.kind == ClassifierKind::linear) {
      const auto weights = model.state().named_parameters();
      const auto expected = torch::linear(
          embedding, weights[original.parameter_prefix + ".weight"],
          weights[original.parameter_prefix + ".bias"]);
      EXPECT_TRUE(torch::allclose(logits, expected, 1e-5, 1e-6));
    }
  }
  reset_classifier(model, 0);
  EXPECT_EQ(model.output_schema().tensor_constraints->shape->at(1),
            original.embedding_features);
  {
    torch::NoGradGuard no_grad;
    EXPECT_TRUE(torch::equal(
        embedding, model.invoke({.positional = {images}}).value.toTensor()));
  }
  for (const auto& item : model.state().named_parameters())
    EXPECT_FALSE(IsHead(item.key(), original));
  reset_classifier(model, 5);
  std::vector<Tensor> head;
  for (const auto& item : model.state().named_parameters()) {
    if (!IsHead(item.key(), original)) continue;
    EXPECT_TRUE(item.value().requires_grad());
    EXPECT_EQ(item.value().device(), images.device());
    EXPECT_EQ(item.value().scalar_type(), images.scalar_type());
    head.push_back(item.value());
  }
  ASSERT_FALSE(head.empty());
  const auto key =
      original.parameter_prefix +
      (original.kind == ClassifierKind::mobile_mlp ? ".3.bias" : ".bias");
  const auto trained = model.state().named_parameters()[key];
  const auto before = trained.detach().clone();
  torch::optim::SGD optimizer(head, torch::optim::SGDOptions(0.1));
  auto logits = model.invoke({.positional = {images}}).value.toTensor();
  auto labels = torch::tensor(
      {0, 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto loss = torch::nn::functional::cross_entropy(logits, labels);
  loss.backward();
  ASSERT_TRUE(trained.grad().defined());
  EXPECT_TRUE(torch::isfinite(trained.grad()).all().item<bool>());
  EXPECT_GT(trained.grad().abs().sum().item<double>(), 0.0);
  optimizer.step();
  EXPECT_FALSE(torch::equal(trained, before));
  EXPECT_TRUE(torch::equal(first, snapshot));
  EXPECT_FALSE(first.grad().defined());
  model.state().train();
  reset_classifier(model, 2);
  for (const auto& child : module->modules(false))
    EXPECT_TRUE(child->is_training());
}

std::vector<HeadCase> Cases() {
  std::vector<HeadCase> result;
  for (const auto name : {"resnet18",
                          "resnet34",
                          "resnet50",
                          "resnet101",
                          "resnext50_32x4d",
                          "densenet121",
                          "mobilenetv2_100",
                          "mobilenet_v3_small",
                          "mobilenet_v3_large",
                          "efficientnet_b0",
                          "efficientnet_b2",
                          "tf_efficientnetv2_s",
                          "convnext_tiny",
                          "convnext_small",
                          "vit_tiny_patch16_224",
                          "vit_small_patch16_224",
                          "vit_base_patch16_224",
                          "swin_tiny_patch4_window7_224",
                          "vit_small_patch14_dinov2",
                          "dinov3_vits16",
                          "dinov3_convnext_tiny"}) {
    result.push_back({name, false});
    result.push_back({name, true});
  }
  return result;
}
INSTANTIATE_TEST_SUITE_P(Vision, ModelsHeadsExecution,
                         ::testing::ValuesIn(Cases()),
                         [](const ::testing::TestParamInfo<HeadCase>& info) {
                           return std::string(info.param.name) +
                                  (info.param.cuda ? "_Cuda" : "_Cpu");
                         });

TEST(ModelsHeads,
     UnsupportedModulesAndFeatureAdaptersFailBeforeStateOrRngChanges) {
  auto encoder = create_model("transformer_encoder",
                              {.transformer = {.model_dimension = 16,
                                               .attention_heads = 4,
                                               .feedforward_dimension = 32,
                                               .layers = 1}});
  EXPECT_THROW(classifier_info(encoder), NotImplemented);
  EXPECT_THROW(reset_classifier(encoder, 3), NotImplemented);
  auto features = create_feature_extractor("resnet18");
  const auto count = parameter_count(features.state().module());
  EXPECT_EQ(classifier_info(features).num_classes, 1000);
  torch::manual_seed(23);
  auto next = torch::rand({8});
  torch::manual_seed(23);
  EXPECT_THROW(reset_classifier(features, 3), InvalidArgument);
  EXPECT_TRUE(torch::equal(torch::rand({8}), next));
  EXPECT_EQ(parameter_count(features.state().module()), count);
  EXPECT_EQ(classifier_info(features).num_classes, 1000);
}

TEST(ModelsHeads, NondefaultMobileWidthAndFloatingDtypeArePreserved) {
  torch::set_num_threads(1);
  auto model = create_model(
      "mobilenet_v3_small",
      {.mobile = {
           .num_classes = 0, .input_channels = 1, .width_multiplier = 0.37}});
  model.state().to(torch::kFloat64);
  model.state().eval();
  EXPECT_EQ(classifier_info(model).embedding_features, 192);
  reset_classifier(model, 4);
  for (const auto& p : model.state().parameters())
    EXPECT_EQ(p.scalar_type(), torch::kFloat64);
  torch::NoGradGuard no_grad;
  const auto logits =
      model
          .invoke(
              {.positional = {torch::ones({1, 1, 33, 49}, torch::kFloat64)}})
          .value.toTensor();
  EXPECT_EQ(logits.sizes(), torch::IntArrayRef({1, 4}));
  EXPECT_TRUE(
      torch::equal(logits, model
                               .invoke({.positional = {torch::ones(
                                            {1, 1, 33, 49}, torch::kFloat64)}})
                               .value.toTensor()));
}
}  // namespace
