// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "feature_adapters.hpp"

#include <utility>

namespace jyppx::liuding::models::detail {
namespace {
ValueSchema FeatureTensor(std::int64_t rank, std::string name) {
  TensorSchema schema;
  schema.rank = rank;
  return ValueSchema::tensor(std::move(schema), std::move(name));
}

ValueSchema TokenSchema() {
  return ValueSchema::tuple(
      {FeatureTensor(2, "cls"), FeatureTensor(3, "patch_tokens"),
       FeatureTensor(4, "patch_grid"), FeatureTensor(3, "tokens")},
      "features", "image_features");
}

c10::IValue TokenViews(const Tensor& tokens, const Tensor& input,
                       std::int64_t patch, std::int64_t prefix) {
  auto patches = tokens.slice(1, prefix);
  auto grid = patches
                  .view({tokens.size(0), input.size(2) / patch,
                         input.size(3) / patch, tokens.size(2)})
                  .permute({0, 3, 1, 2});
  return c10::ivalue::Tuple::create(
      {tokens.select(1, 0), std::move(patches), std::move(grid), tokens});
}
}  // namespace

std::vector<FeatureInfo> TokenFeatureInfo(std::int64_t channels,
                                          std::int64_t patch,
                                          std::int64_t prefix) {
  return {{.name = "cls", .layout = FeatureLayout::nc, .channels = channels},
          {.name = "patch_tokens",
           .layout = FeatureLayout::ntc,
           .channels = channels,
           .reduction = patch},
          {.name = "patch_grid",
           .layout = FeatureLayout::nchw,
           .channels = channels,
           .reduction = patch},
          {.name = "tokens",
           .layout = FeatureLayout::ntc,
           .channels = channels,
           .reduction = patch,
           .prefix_tokens = prefix}};
}

std::vector<FeatureInfo> StageFeatureInfo(
    const std::array<std::int64_t, 4>& channels) {
  std::vector<FeatureInfo> result;
  for (std::size_t index = 0; index < channels.size(); ++index)
    result.push_back({.name = "stage" + std::to_string(index + 1),
                      .layout = FeatureLayout::nchw,
                      .channels = channels[index],
                      .reduction = 4LL << index});
  return result;
}

nn::ModelAdapter AdaptFeatures(DinoV2 model) {
  auto input_schema = adapt(model).input_schema();
  return nn::adapt_holder(
      model,
      [](DinoV2Impl& module, const nn::Invocation& invocation) {
        const auto input = invocation.positional.front().toTensor();
        return TokenViews(module.forward_features(input), input, 14, 1);
      },
      std::move(input_schema), TokenSchema());
}

nn::ModelAdapter AdaptFeatures(DinoV3 model) {
  auto input_schema = adapt(model).input_schema();
  return nn::adapt_holder(
      model,
      [](DinoV3Impl& module, const nn::Invocation& invocation) {
        const auto input = invocation.positional.front().toTensor();
        return TokenViews(module.forward_features(input), input, 16, 5);
      },
      std::move(input_schema), TokenSchema());
}

nn::ModelAdapter AdaptFeatures(DinoV3ConvNeXt model) {
  auto input_schema = adapt(model).input_schema();
  std::vector<ValueSchema> outputs;
  for (std::size_t index = 0; index < 4; ++index)
    outputs.push_back(FeatureTensor(4, "stage" + std::to_string(index + 1)));
  return nn::adapt_holder(
      model,
      [](DinoV3ConvNeXtImpl& module, const nn::Invocation& invocation) {
        std::vector<c10::IValue> result;
        for (auto& feature :
             module.forward_features(invocation.positional.front().toTensor()))
          result.emplace_back(std::move(feature));
        return c10::ivalue::Tuple::create(std::move(result));
      },
      std::move(input_schema),
      ValueSchema::tuple(std::move(outputs), "features", "image_features"));
}
}  // namespace jyppx::liuding::models::detail
