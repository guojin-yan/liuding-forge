// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/liuding/models/efficientnetv2.hpp>

#include <torch/torch.h>

#include <array>
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
                               .padding(stride == 1 ? (kernel - 1) / 2 : 0)
                               .groups(groups)
                               .bias(bias));
}

torch::nn::BatchNorm2d Norm(std::int64_t channels) {
  return torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(channels).eps(1e-3).momentum(0.1));
}

Tensor SamePad(const Tensor& input, std::int64_t kernel, std::int64_t stride) {
  if (stride == 1) return input;
  const auto pad_h = ((input.size(2) + stride - 1) / stride - 1) * stride +
                     kernel - input.size(2);
  const auto pad_w = ((input.size(3) + stride - 1) / stride - 1) * stride +
                     kernel - input.size(3);
  return torch::constant_pad_nd(
      input, {pad_w / 2, pad_w - pad_w / 2, pad_h / 2, pad_h - pad_h / 2}, 0);
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
    auto pooled = torch::silu(conv_reduce_(input.mean({2, 3}, true)));
    return input * torch::sigmoid(conv_expand_(pooled));
  }

 private:
  torch::nn::Conv2d conv_reduce_{nullptr}, conv_expand_{nullptr};
};

class Block : public torch::nn::Module {
 public:
  virtual Tensor forward(const Tensor& input) = 0;
};

class ConvBnAct final : public Block {
 public:
  explicit ConvBnAct(std::int64_t channels) {
    conv_ = register_module("conv", Conv(channels, channels, 3));
    bn1_ = register_module("bn1", Norm(channels));
  }
  Tensor forward(const Tensor& input) override {
    return input + torch::silu(bn1_(conv_(input)));
  }

 private:
  torch::nn::Conv2d conv_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr};
};

class FusedMbConv final : public Block {
 public:
  FusedMbConv(std::int64_t input, std::int64_t output, std::int64_t stride)
      : skip_(stride == 1 && input == output), stride_(stride) {
    const auto expanded = input * 4;
    conv_exp_ = register_module("conv_exp", Conv(input, expanded, 3, stride));
    bn1_ = register_module("bn1", Norm(expanded));
    conv_pwl_ = register_module("conv_pwl", Conv(expanded, output, 1));
    bn2_ = register_module("bn2", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    auto value = torch::silu(bn1_(conv_exp_(SamePad(input, 3, stride_))));
    value = bn2_(conv_pwl_(value));
    return skip_ ? value + input : value;
  }

 private:
  bool skip_;
  std::int64_t stride_;
  torch::nn::Conv2d conv_exp_{nullptr}, conv_pwl_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};

class MbConv final : public Block {
 public:
  MbConv(std::int64_t input, std::int64_t output, std::int64_t expansion,
         std::int64_t stride)
      : skip_(stride == 1 && input == output), stride_(stride) {
    const auto expanded = input * expansion;
    conv_pw_ = register_module("conv_pw", Conv(input, expanded, 1));
    bn1_ = register_module("bn1", Norm(expanded));
    conv_dw_ = register_module("conv_dw",
                               Conv(expanded, expanded, 3, stride, expanded));
    bn2_ = register_module("bn2", Norm(expanded));
    se_ = register_module("se",
                          std::make_shared<SqueezeExcite>(expanded, input / 4));
    conv_pwl_ = register_module("conv_pwl", Conv(expanded, output, 1));
    bn3_ = register_module("bn3", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    auto value = torch::silu(bn1_(conv_pw_(input)));
    value = torch::silu(bn2_(conv_dw_(SamePad(value, 3, stride_))));
    value = bn3_(conv_pwl_(se_->forward(value)));
    return skip_ ? value + input : value;
  }

 private:
  bool skip_;
  std::int64_t stride_;
  torch::nn::Conv2d conv_pw_{nullptr}, conv_dw_{nullptr}, conv_pwl_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr}, bn3_{nullptr};
  std::shared_ptr<SqueezeExcite> se_;
};

class Stage final : public torch::nn::Module {
 public:
  Stage(std::int64_t input, std::int64_t output, std::int64_t count,
        std::int64_t stride, std::int64_t expansion, bool stem_stage,
        bool fused_stage) {
    blocks_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
      const auto block_input = index == 0 ? input : output;
      const auto block_stride = index == 0 ? stride : 1;
      std::shared_ptr<Block> block;
      if (stem_stage)
        block = std::make_shared<ConvBnAct>(block_input);
      else if (fused_stage)
        block =
            std::make_shared<FusedMbConv>(block_input, output, block_stride);
      else
        block = std::make_shared<MbConv>(block_input, output, expansion,
                                         block_stride);
      blocks_.push_back(register_module(std::to_string(index), block));
    }
  }
  Tensor forward(Tensor input) {
    for (const auto& block : blocks_) input = block->forward(input);
    return input;
  }

 private:
  std::vector<std::shared_ptr<Block>> blocks_;
};
}  // namespace

class EfficientNetV2Blocks final : public torch::nn::Module {
 public:
  EfficientNetV2Blocks() {
    constexpr std::array<std::int64_t, 6> kCounts{2, 4, 4, 6, 9, 15};
    constexpr std::array<std::int64_t, 6> kChannels{24, 48, 64, 128, 160, 256};
    constexpr std::array<std::int64_t, 6> kStride{1, 2, 2, 2, 1, 2};
    constexpr std::array<std::int64_t, 6> kExpansion{1, 4, 4, 4, 6, 6};
    std::int64_t input = 24;
    for (std::size_t index = 0; index < stages_.size(); ++index) {
      stages_[index] = register_module(
          std::to_string(index),
          std::make_shared<Stage>(input, kChannels[index], kCounts[index],
                                  kStride[index], kExpansion[index], index == 0,
                                  index == 1 || index == 2));
      input = kChannels[index];
    }
  }
  std::vector<Tensor> forward_features(Tensor input) {
    std::vector<Tensor> features;
    features.reserve(5);
    for (std::size_t index = 0; index < stages_.size(); ++index) {
      input = stages_[index]->forward(input);
      if (index != 3) features.push_back(input);
    }
    return features;
  }

 private:
  std::array<std::shared_ptr<Stage>, 6> stages_;
};

EfficientNetV2Impl::EfficientNetV2Impl(EfficientNetV2Options options)
    : options_(options) {
  detail::Backend("efficientnetv2.construct", [&] {
    if (options.num_classes <= 0 || options.num_classes > 1000000 ||
        options.input_channels <= 0 || options.input_channels > 1024)
      detail::Invalid("efficientnetv2.options",
                      "classes 1..1000000 and channels 1..1024");
    conv_stem_ =
        register_module("conv_stem", Conv(options.input_channels, 24, 3, 2));
    bn1_ = register_module("bn1", Norm(24));
    blocks_ =
        register_module("blocks", std::make_shared<EfficientNetV2Blocks>());
    conv_head_ = register_module("conv_head", Conv(256, 1280, 1));
    bn2_ = register_module("bn2", Norm(1280));
    classifier_ = register_module("classifier",
                                  torch::nn::Linear(1280, options.num_classes));
  });
}

std::vector<Tensor> EfficientNetV2Impl::forward_features(const Tensor& input) {
  return detail::Backend("efficientnetv2.forward_features", [&] {
    detail::FloatingInput(input, conv_stem_->weight, 4);
    if (input.size(1) != options_.input_channels)
      detail::Invalid("efficientnetv2.input",
                      "the configured image channel count");
    return blocks_->forward_features(
        torch::silu(bn1_(conv_stem_(SamePad(input, 3, 2)))));
  });
}

Tensor EfficientNetV2Impl::forward(const Tensor& input) {
  return detail::Backend("efficientnetv2.forward", [&] {
    const auto stages = forward_features(input);
    const auto features = torch::silu(bn2_(conv_head_(stages.back())));
    return classifier_(features.mean({2, 3}));
  });
}

const EfficientNetV2Options& EfficientNetV2Impl::options() const noexcept {
  return options_;
}

EfficientNetV2 tf_efficientnetv2_s(EfficientNetV2Options options) {
  return EfficientNetV2(options);
}

nn::ModelAdapter adapt(EfficientNetV2 model) {
  if (!model) detail::Invalid("efficientnetv2.adapt", "a nonempty model");
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
