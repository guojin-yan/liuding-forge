// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"
#include "feature_adapters.hpp"

#include <cmath>
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

template <class Holder>
nn::ModelAdapter AdaptStages(Holder model, std::size_t count) {
  auto input_schema = adapt(model).input_schema();
  std::vector<ValueSchema> outputs;
  for (std::size_t index = 0; index < count; ++index)
    outputs.push_back(FeatureTensor(4, "stage" + std::to_string(index + 1)));
  return nn::adapt_holder(
      model,
      [](typename Holder::ContainedType& module,
         const nn::Invocation& invocation) {
        std::vector<c10::IValue> result;
        for (auto& feature :
             module.forward_features(invocation.positional.front().toTensor()))
          result.emplace_back(std::move(feature));
        return c10::ivalue::Tuple::create(std::move(result));
      },
      std::move(input_schema),
      ValueSchema::tuple(std::move(outputs), "features", "image_features"));
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
    std::initializer_list<std::int64_t> channels, std::int64_t first_stride) {
  std::vector<FeatureInfo> result;
  auto stride = first_stride;
  for (const auto channel : channels) {
    result.push_back({.name = "stage" + std::to_string(result.size() + 1),
                      .layout = FeatureLayout::nchw,
                      .channels = channel,
                      .reduction = stride});
    stride *= 2;
  }
  return result;
}

FeatureSpec ViTFeatureSpec(std::int64_t channels, bool dynamic) {
  return {
      .outputs = TokenFeatureInfo(channels, 16, 1),
      .input = {.spatial_multiple = 16,
                .fixed_size = dynamic ? std::array<std::int64_t, 2>{0, 0}
                                      : std::array<std::int64_t, 2>{224, 224}}};
}

FeatureSpec SwinFeatureSpec(bool dynamic) {
  return {
      .outputs = StageFeatureInfo({96, 192, 384, 768}),
      .input = {.spatial_multiple = 4,
                .fixed_size = dynamic ? std::array<std::int64_t, 2>{0, 0}
                                      : std::array<std::int64_t, 2>{224, 224}}};
}

FeatureSpec MobileFeatureSpec(double width, bool large) {
  if (!std::isfinite(width) || width < 0.25 || width > 2.0)
    Invalid("mobilenet.feature_spec", "finite width in [0.25,2]");
  return {
      .outputs = {{.name = "spatial",
                   .layout = FeatureLayout::nchw,
                   .channels = 6 * Divisible((large ? 160.0 : 96.0) * width),
                   .reduction = 32}},
      .input = {}};
}

nn::ModelAdapter AdaptFeatures(ResNet model) { return AdaptStages(model, 4); }
nn::ModelAdapter AdaptFeatures(DenseNet model) { return AdaptStages(model, 4); }
nn::ModelAdapter AdaptFeatures(MobileNetV2 model) {
  return AdaptStages(model, 5);
}
nn::ModelAdapter AdaptFeatures(EfficientNet model) {
  return AdaptStages(model, 5);
}
nn::ModelAdapter AdaptFeatures(EfficientNetV2 model) {
  return AdaptStages(model, 5);
}
nn::ModelAdapter AdaptFeatures(ConvNeXt model) { return AdaptStages(model, 4); }
nn::ModelAdapter AdaptFeatures(SwinTiny model) { return AdaptStages(model, 4); }

nn::ModelAdapter AdaptFeatures(MobileNetV3 model) {
  auto input_schema = adapt(model).input_schema();
  return nn::adapt_holder(
      model,
      [](MobileNetV3Impl& module, const nn::Invocation& invocation) {
        return c10::ivalue::Tuple::create({module.forward_features(
            invocation.positional.front().toTensor())});
      },
      std::move(input_schema),
      ValueSchema::tuple({FeatureTensor(4, "spatial")}, "features",
                         "image_features"));
}

nn::ModelAdapter AdaptFeatures(ViT model) {
  auto input_schema = adapt(model).input_schema();
  return nn::adapt_holder(
      model,
      [](ViTImpl& module, const nn::Invocation& invocation) {
        const auto input = invocation.positional.front().toTensor();
        return TokenViews(module.forward_features(input), input, 16, 1);
      },
      std::move(input_schema), TokenSchema());
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
  return AdaptStages(model, 4);
}
}  // namespace jyppx::liuding::models::detail
