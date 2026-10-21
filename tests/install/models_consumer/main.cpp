// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");

#include <jyppx/models/models.hpp>

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
    const auto entries = jyppx::models::list_models();
    Require(entries.size() == 5U, "installed-model-catalog");
    auto resnet =
        jyppx::models::create_model("resnet18", {.resnet = {.num_classes = 4}});
    auto mobile = jyppx::models::create_model("mobilenet_v3_small",
                                              {.mobile = {.num_classes = 4}});
    auto encoder = jyppx::models::create_model(
        "transformer_encoder", {.transformer = {.model_dimension = 16,
                                                .attention_heads = 4,
                                                .feedforward_dimension = 32,
                                                .layers = 1,
                                                .dropout = 0.0}});
    resnet.state().eval();
    mobile.state().eval();
    encoder.state().eval();
    const auto images = torch::zeros({1, 3, 64, 64});
    Require(resnet.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-resnet-adapter");
    Require(mobile.invoke({.positional = {images}, .keyword = {}})
                    .value.toTensor()
                    .sizes() == torch::IntArrayRef({1, 4}),
            "installed-mobile-adapter");
    const auto encoded = encoder.invoke(
        {.positional = {torch::zeros({1, 3, 16})}, .keyword = {}});
    Require(encoded.value.toTensor().sizes() == torch::IntArrayRef({1, 3, 16}),
            "installed-transformer-adapter");
    const auto weights = jyppx::models::available_weights("resnet18");
    Require(weights.size() == 1U &&
                weights.front().availability ==
                    jyppx::models::WeightAvailability::not_provisioned &&
                !weights.front().artifact.has_value(),
            "installed-weight-provenance-status");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
