// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/liuding/models/efficientnet.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::models {
namespace {
torch::nn::Conv2d Conv(std::int64_t input, std::int64_t output,
                       std::int64_t kernel, std::int64_t stride = 1,
                       std::int64_t groups = 1, bool bias = false) {
  return torch::nn::Conv2d(torch::nn::Conv2dOptions(input, output, kernel)
                               .stride(stride)
                               .padding((kernel - 1) / 2)
                               .groups(groups)
                               .bias(bias));
}

torch::nn::BatchNorm2d Norm(std::int64_t channels) {
  return torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(channels).eps(1e-5).momentum(0.1));
}

std::int64_t RoundChannels(std::int64_t channels, double multiplier) {
  const auto scaled = static_cast<double>(channels) * multiplier;
  auto rounded = std::max<std::int64_t>(
      8, static_cast<std::int64_t>(scaled + 4.0) / 8 * 8);
  if (static_cast<double>(rounded) < 0.9 * scaled) rounded += 8;
  return rounded;
}

class SqueezeExcite final : public torch::nn::Module {
 public:
  SqueezeExcite(std::int64_t channels, std::int64_t reduced) {
    conv_reduce_ =
        register_module("conv_reduce", Conv(channels, reduced, 1, 1, 1, true));
    conv_expand_ =
        register_module("conv_expand", Conv(reduced, channels, 1, 1, 1, true));
  }
  Tensor forward(const Tensor& input) {
    auto squeeze = input.mean({2, 3}, true);
    squeeze = torch::silu(conv_reduce_(squeeze));
    return input * torch::sigmoid(conv_expand_(squeeze));
  }

 private:
  torch::nn::Conv2d conv_reduce_{nullptr}, conv_expand_{nullptr};
};

class EfficientNetBlock : public torch::nn::Module {
 public:
  virtual Tensor forward(const Tensor& input) = 0;
};

class DepthwiseSeparable final : public EfficientNetBlock {
 public:
  DepthwiseSeparable(std::int64_t input, std::int64_t output)
      : skip_(input == output) {
    conv_dw_ = register_module("conv_dw", Conv(input, input, 3, 1, input));
    bn1_ = register_module("bn1", Norm(input));
    se_ = register_module("se",
                          std::make_shared<SqueezeExcite>(input, input / 4));
    conv_pw_ = register_module("conv_pw", Conv(input, output, 1));
    bn2_ = register_module("bn2", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    auto value = torch::silu(bn1_(conv_dw_(input)));
    value = bn2_(conv_pw_(se_->forward(value)));
    return skip_ ? value + input : value;
  }

 private:
  bool skip_;
  torch::nn::Conv2d conv_dw_{nullptr}, conv_pw_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
  std::shared_ptr<SqueezeExcite> se_;
};

class InvertedResidual final : public EfficientNetBlock {
 public:
  InvertedResidual(std::int64_t input, std::int64_t output, std::int64_t kernel,
                   std::int64_t stride)
      : skip_(stride == 1 && input == output) {
    const auto expanded = RoundChannels(input * 6, 1.0);
    const auto reduced = static_cast<std::int64_t>(
        std::nearbyint(static_cast<double>(expanded) / 24.0));
    conv_pw_ = register_module("conv_pw", Conv(input, expanded, 1));
    bn1_ = register_module("bn1", Norm(expanded));
    conv_dw_ = register_module(
        "conv_dw", Conv(expanded, expanded, kernel, stride, expanded));
    bn2_ = register_module("bn2", Norm(expanded));
    se_ = register_module("se",
                          std::make_shared<SqueezeExcite>(expanded, reduced));
    conv_pwl_ = register_module("conv_pwl", Conv(expanded, output, 1));
    bn3_ = register_module("bn3", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    auto value = torch::silu(bn1_(conv_pw_(input)));
    value = torch::silu(bn2_(conv_dw_(value)));
    value = bn3_(conv_pwl_(se_->forward(value)));
    return skip_ ? value + input : value;
  }

 private:
  bool skip_;
  torch::nn::Conv2d conv_pw_{nullptr}, conv_dw_{nullptr}, conv_pwl_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr}, bn3_{nullptr};
  std::shared_ptr<SqueezeExcite> se_;
};

class EfficientNetStage final : public torch::nn::Module {
 public:
  EfficientNetStage(std::int64_t input, std::int64_t output, std::int64_t count,
                    std::int64_t kernel, std::int64_t stride,
                    bool first_stage) {
    blocks_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
      std::shared_ptr<EfficientNetBlock> block;
      if (first_stage) {
        block = std::make_shared<DepthwiseSeparable>(
            index == 0 ? input : output, output);
      } else {
        block = std::make_shared<InvertedResidual>(index == 0 ? input : output,
                                                   output, kernel,
                                                   index == 0 ? stride : 1);
      }
      blocks_.push_back(
          register_module(std::to_string(index), std::move(block)));
    }
  }
  Tensor forward(Tensor input) {
    for (const auto& block : blocks_) input = block->forward(input);
    return input;
  }

 private:
  std::vector<std::shared_ptr<EfficientNetBlock>> blocks_;
};
}  // namespace

class EfficientNetBlocks final : public torch::nn::Module {
 public:
  explicit EfficientNetBlocks(EfficientNetVariant variant) {
    constexpr std::array<std::int64_t, 7> kBaseCount{1, 2, 2, 3, 3, 4, 1};
    constexpr std::array<std::int64_t, 7> kBaseOutput{16,  24,  40, 80,
                                                      112, 192, 320};
    constexpr std::array<std::int64_t, 7> kKernel{3, 3, 5, 3, 5, 5, 3};
    constexpr std::array<std::int64_t, 7> kStride{1, 2, 2, 2, 1, 2, 1};
    const auto width = variant == EfficientNetVariant::b0 ? 1.0 : 1.1;
    const auto depth = variant == EfficientNetVariant::b0 ? 1.0 : 1.2;
    std::int64_t input = RoundChannels(32, width);
    for (std::size_t index = 0; index < stages_.size(); ++index) {
      const auto output = RoundChannels(kBaseOutput[index], width);
      const auto count = static_cast<std::int64_t>(
          std::ceil(static_cast<double>(kBaseCount[index]) * depth));
      stages_[index] = register_module(std::to_string(index),
                                       std::make_shared<EfficientNetStage>(
                                           input, output, count, kKernel[index],
                                           kStride[index], index == 0));
      input = output;
    }
  }
  std::vector<Tensor> forward_features(Tensor input) {
    std::vector<Tensor> features;
    features.reserve(5);
    for (std::size_t index = 0; index < stages_.size(); ++index) {
      input = stages_[index]->forward(input);
      if (index == 0 || index == 1 || index == 2 || index == 4 || index == 6)
        features.push_back(input);
    }
    return features;
  }

 private:
  std::array<std::shared_ptr<EfficientNetStage>, 7> stages_;
};

EfficientNetImpl::EfficientNetImpl(EfficientNetVariant variant,
                                   EfficientNetOptions options)
    : variant_(variant), options_(options) {
  detail::Backend("efficientnet.construct", [&] {
    if (variant != EfficientNetVariant::b0 &&
        variant != EfficientNetVariant::b2)
      detail::Invalid("efficientnet.variant", "b0 or b2");
    if (options.num_classes <= 0 || options.num_classes > 1000000 ||
        options.input_channels <= 0 || options.input_channels > 1024)
      detail::Invalid("efficientnet.options",
                      "classes 1..1000000 and channels 1..1024");
    const auto width = variant == EfficientNetVariant::b0 ? 1.0 : 1.1;
    const auto stem_channels = RoundChannels(32, width);
    const auto last_channels = RoundChannels(320, width);
    const auto head_channels = RoundChannels(1280, width);
    conv_stem_ = register_module(
        "conv_stem", Conv(options.input_channels, stem_channels, 3, 2));
    bn1_ = register_module("bn1", Norm(stem_channels));
    blocks_ = register_module("blocks",
                              std::make_shared<EfficientNetBlocks>(variant));
    conv_head_ =
        register_module("conv_head", Conv(last_channels, head_channels, 1));
    bn2_ = register_module("bn2", Norm(head_channels));
    classifier_ = register_module(
        "classifier", torch::nn::Linear(head_channels, options.num_classes));
  });
}

std::vector<Tensor> EfficientNetImpl::forward_features(const Tensor& input) {
  return detail::Backend("efficientnet.forward_features", [&] {
    detail::FloatingInput(input, conv_stem_->weight, 4);
    if (input.size(1) != options_.input_channels)
      detail::Invalid("efficientnet.input",
                      "the configured image channel count");
    return blocks_->forward_features(torch::silu(bn1_(conv_stem_(input))));
  });
}

Tensor EfficientNetImpl::forward(const Tensor& input) {
  return detail::Backend("efficientnet.forward", [&] {
    const auto stages = forward_features(input);
    const auto features = torch::silu(bn2_(conv_head_(stages.back())));
    return classifier_(features.mean({2, 3}));
  });
}

const EfficientNetOptions& EfficientNetImpl::options() const noexcept {
  return options_;
}

EfficientNetVariant EfficientNetImpl::variant() const noexcept {
  return variant_;
}

EfficientNet efficientnet_b0(EfficientNetOptions options) {
  return EfficientNet(EfficientNetVariant::b0, options);
}

EfficientNet efficientnet_b2(EfficientNetOptions options) {
  return EfficientNet(EfficientNetVariant::b2, options);
}

nn::ModelAdapter adapt(EfficientNet model) {
  if (!model) detail::Invalid("efficientnet.adapt", "a nonempty model");
  const auto options = model->options();
  TensorSchema input;
  input.rank = 4;
  input.shape = {
      {std::nullopt, options.input_channels, std::nullopt, std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, options.num_classes}};
  return nn::adapt_module(
      model.ptr(),
      [model](const nn::Invocation& invocation) mutable {
        return c10::IValue(
            model->forward(invocation.positional.front().toTensor()));
      },
      {.positional = {ValueSchema::tensor(input, "images",
                                          "normalized_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "logits", "classification_logits"));
}
}  // namespace jyppx::liuding::models
