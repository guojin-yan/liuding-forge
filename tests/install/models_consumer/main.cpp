// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");

#include <jyppx/liuding/models/models.hpp>

#include <exception>
#include <iostream>
#include <string>

namespace {
void Require(bool condition, const char* operation) {
  if (!condition) throw std::runtime_error(operation);
}
}  // namespace

int main() {
  try {
    const auto entries = jyppx::liuding::models::list_models();
    Require(entries.size() == 34U, "installed-model-catalog");
    {
      using namespace jyppx::liuding::models;
      auto model = create_model("resnet18", {.resnet = {.num_classes = 0}});
      model.state().eval();
      const auto original = model.state().shared_module();
      const auto info = classifier_info(model);
      Require(info.embedding_features == 512 && info.num_classes == 0 &&
                  info.parameter_prefix == "fc",
              "installed-headless-classifier-info");
      reset_classifier(model, 3);
      Require(classifier_info(model).num_classes == 3 &&
                  model.state().shared_module() == original,
              "installed-reset-classifier-info");
      torch::NoGradGuard no_grad;
      const auto images = torch::zeros({1, 3, 35, 67});
      Require(
          model.invoke({.positional = {images}}).value.toTensor().size(1) == 3,
          "installed-reset-classifier-schema");
      reset_classifier(model, 0);
      Require(model.invoke({.positional = {images}}).value.toTensor().size(1) ==
                  512,
              "installed-remove-classifier-schema");
    }
    {
      using namespace jyppx::liuding::models;
      std::size_t feature_count = 0;
      for (const auto& entry : entries)
        if (!entry.features.empty()) ++feature_count;
      Require(feature_count == 33U, "installed-visual-feature-catalog");
      auto features = create_feature_extractor("resnet18", {}, {3, 0});
      features.state().eval();
      torch::NoGradGuard no_grad;
      const auto outputs =
          features.invoke({.positional = {torch::zeros({1, 3, 35, 67})}})
              .value.toTuple();
      Require(outputs->elements()[0].toTensor().sizes() ==
                      torch::IntArrayRef({1, 512, 2, 3}) &&
                  outputs->elements()[1].toTensor().sizes() ==
                      torch::IntArrayRef({1, 64, 9, 17}),
              "installed-resnet-features-and-order");
    }
    {
      using namespace jyppx::liuding::models;
      ModelOptions options;
      options.mobile.width_multiplier = 0.37;
      options.mobile.input_channels = 1;
      const auto spec = model_feature_spec("mobilenet_v3_small", options);
      Require(spec.outputs[0].channels == 192,
              "installed-configured-feature-spec");
      auto features = create_feature_extractor("mobilenet_v3_small", options);
      features.state().eval();
      torch::NoGradGuard no_grad;
      const auto outputs =
          features.invoke({.positional = {torch::zeros({1, 1, 35, 67})}})
              .value.toTuple();
      Require(outputs->elements()[0].toTensor().sizes() ==
                  torch::IntArrayRef({1, 192, 2, 3}),
              "installed-configured-mobile-features");
      const auto fixed = model_feature_spec("vit_tiny_patch16_224");
      options.vit.dynamic_image_size = true;
      const auto dynamic = model_feature_spec("vit_tiny_patch16_224", options);
      Require(
          fixed.input.fixed_size[0] == 224 && dynamic.input.fixed_size[0] == 0,
          "installed-configured-spatial-spec");
    }
    {
      const auto info = jyppx::liuding::models::model_info("dinov3_vits16");
      Require(info.features.size() == 4 && info.features[3].prefix_tokens == 5,
              "installed-feature-metadata");
      auto features = jyppx::liuding::models::create_feature_extractor(
          "dinov3_vits16", {}, {0, 2});
      features.state().eval();
      torch::NoGradGuard no_grad;
      const auto outputs =
          features.invoke({.positional = {torch::zeros({2, 3, 35, 50})}})
              .value.toTuple();
      Require(outputs->elements()[0].toTensor().sizes() ==
                      torch::IntArrayRef({2, 384}) &&
                  outputs->elements()[1].toTensor().sizes() ==
                      torch::IntArrayRef({2, 384, 2, 3}),
              "installed-selected-features");
    }
    for (const auto* name : {"convnext_tiny", "convnext_small"}) {
      auto convnext = jyppx::liuding::models::create_model(
          name, {.convnext = {.num_classes = 4}});
      convnext.state().eval();
      Require(convnext.invoke({.positional = {torch::zeros({1, 3, 32, 32})},
                               .keyword = {}})
                      .value.toTensor()
                      .sizes() == torch::IntArrayRef({1, 4}),
              "installed-convnext-adapter");
    }
    for (const auto* name : {"vit_tiny_patch16_224", "vit_small_patch16_224",
                             "vit_base_patch16_224"}) {
      auto vit = jyppx::liuding::models::create_model(
          name, {.vit = {.num_classes = 4, .dynamic_image_size = true}});
      vit.state().eval();
      Require(vit.invoke({.positional = {torch::zeros({1, 3, 16, 32})},
                          .keyword = {}})
                      .value.toTensor()
                      .sizes() == torch::IntArrayRef({1, 4}),
              "installed-vit-adapter");
    }
    auto swin = jyppx::liuding::models::create_model(
        "swin_tiny_patch4_window7_224",
        {.swin = {.num_classes = 4, .dynamic_image_size = true}});
    swin.state().eval();
    Require(swin.invoke({.positional = {torch::zeros({1, 3, 32, 32})},
                         .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-swin-adapter");
    auto dense = jyppx::liuding::models::create_model(
        "densenet121", {.densenet = {.num_classes = 4}});
    auto efficient_b0 = jyppx::liuding::models::create_model(
        "efficientnet_b0", {.efficientnet = {.num_classes = 4}});
    auto efficient_b2 = jyppx::liuding::models::create_model(
        "efficientnet_b2", {.efficientnet = {.num_classes = 4}});
    auto efficient_v2 = jyppx::liuding::models::create_model(
        "tf_efficientnetv2_s", {.efficientnet_v2 = {.num_classes = 4}});
    auto resnet = jyppx::liuding::models::create_model(
        "resnet18", {.resnet = {.num_classes = 4}});
    auto resnet34 = jyppx::liuding::models::create_model(
        "resnet34", {.resnet = {.num_classes = 4}});
    auto resnet101 = jyppx::liuding::models::create_model(
        "resnet101", {.resnet = {.num_classes = 4}});
    auto resnext = jyppx::liuding::models::create_model(
        "resnext50_32x4d", {.resnet = {.num_classes = 4}});
    auto mobile_v2 = jyppx::liuding::models::create_model(
        "mobilenetv2_100", {.mobile_v2 = {.num_classes = 4}});
    auto mobile = jyppx::liuding::models::create_model(
        "mobilenet_v3_small", {.mobile = {.num_classes = 4}});
    auto encoder = jyppx::liuding::models::create_model(
        "transformer_encoder", {.transformer = {.model_dimension = 16,
                                                .attention_heads = 4,
                                                .feedforward_dimension = 32,
                                                .layers = 1,
                                                .dropout = 0.0}});
    auto dino =
        jyppx::liuding::models::create_model("vit_small_patch14_dinov2");
    dense.state().eval();
    efficient_b0.state().eval();
    efficient_b2.state().eval();
    efficient_v2.state().eval();
    resnet.state().eval();
    resnet34.state().eval();
    resnet101.state().eval();
    resnext.state().eval();
    mobile_v2.state().eval();
    mobile.state().eval();
    encoder.state().eval();
    dino.state().eval();
    const auto images = torch::zeros({1, 3, 64, 64});
    Require(dense.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-densenet121-adapter");
    Require(efficient_b0.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-efficientnet-b0-adapter");
    Require(efficient_b2.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-efficientnet-b2-adapter");
    Require(efficient_v2.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-tf-efficientnetv2-s-adapter");
    Require(resnet.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-resnet-adapter");
    Require(resnet34.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-resnet34-adapter");
    Require(resnet101.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-resnet101-adapter");
    Require(resnext.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-resnext50-adapter");
    Require(mobile_v2.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-mobilenetv2-adapter");
    Require(mobile.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-mobile-adapter");
    const auto encoded = encoder.invoke(
        {.positional = {torch::zeros({1, 3, 16})}, .keyword = {}});
    Require(encoded.value.toTensor().sizes() == torch::IntArrayRef({1, 3, 16}),
            "installed-transformer-adapter");
    const auto dino_output = dino.invoke(
        {.positional = {torch::zeros({1, 3, 14, 28})}, .keyword = {}});
    Require(
        dino_output.value.toTensor().sizes() == torch::IntArrayRef({1, 384}),
        "installed-dinov2-adapter");
    const auto weights = jyppx::liuding::models::available_weights("resnet18");
    Require(
        weights.size() == 1U &&
            weights.front().availability ==
                jyppx::liuding::models::WeightAvailability::not_provisioned &&
            !weights.front().artifact.has_value(),
        "installed-weight-provenance-status");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
