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
    Require(entries.size() == 27U, "installed-model-catalog");
    auto dense = jyppx::liuding::models::create_model(
        "densenet121", {.densenet = {.num_classes = 4}});
    auto efficient_b0 = jyppx::liuding::models::create_model(
        "efficientnet_b0", {.efficientnet = {.num_classes = 4}});
    auto efficient_b2 = jyppx::liuding::models::create_model(
        "efficientnet_b2", {.efficientnet = {.num_classes = 4}});
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
