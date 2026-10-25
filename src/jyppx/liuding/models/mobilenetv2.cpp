// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/liuding/models/mobilenetv2.hpp>

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
                       std::int64_t groups = 1) {
  return torch::nn::Conv2d(torch::nn::Conv2dOptions(input, output, kernel)
                               .stride(stride)
                               .padding((kernel - 1) / 2)
                               .groups(groups)
                               .bias(false));
}

torch::nn::BatchNorm2d Norm(std::int64_t channels) {
  return torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(channels).eps(1e-5).momentum(0.1));
}

Tensor Relu6(const Tensor& input) { return torch::clamp(input, 0, 6); }

class MobileNetV2Block : public torch::nn::Module {
 public:
  virtual Tensor forward(const Tensor& input) = 0;
};

class DepthwiseSeparable final : public MobileNetV2Block {
 public:
  DepthwiseSeparable(std::int64_t input, std::int64_t output) {
    conv_dw_ = register_module("conv_dw", Conv(input, input, 3, 1, input));
    bn1_ = register_module("bn1", Norm(input));
    conv_pw_ = register_module("conv_pw", Conv(input, output, 1));
    bn2_ = register_module("bn2", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    return bn2_(conv_pw_(Relu6(bn1_(conv_dw_(input)))));
  }

 private:
  torch::nn::Conv2d conv_dw_{nullptr}, conv_pw_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};

class InvertedResidual final : public MobileNetV2Block {
 public:
  InvertedResidual(std::int64_t input, std::int64_t output, std::int64_t stride)
      : skip_(stride == 1 && input == output) {
    const auto expanded = input * 6;
    conv_pw_ = register_module("conv_pw", Conv(input, expanded, 1));
    bn1_ = register_module("bn1", Norm(expanded));
    conv_dw_ = register_module("conv_dw",
                               Conv(expanded, expanded, 3, stride, expanded));
    bn2_ = register_module("bn2", Norm(expanded));
    conv_pwl_ = register_module("conv_pwl", Conv(expanded, output, 1));
    bn3_ = register_module("bn3", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    auto expanded = Relu6(bn1_(conv_pw_(input)));
    auto depthwise = Relu6(bn2_(conv_dw_(expanded)));
    auto projected = bn3_(conv_pwl_(depthwise));
    return skip_ ? projected + input : projected;
  }

 private:
  bool skip_;
  torch::nn::Conv2d conv_pw_{nullptr}, conv_dw_{nullptr}, conv_pwl_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr}, bn3_{nullptr};
};

class MobileNetV2Stage final : public torch::nn::Module {
 public:
  MobileNetV2Stage(std::int64_t input, std::int64_t output, std::int64_t count,
                   std::int64_t stride, bool first_stage) {
    blocks_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
      std::shared_ptr<MobileNetV2Block> block =
          first_stage ? std::static_pointer_cast<MobileNetV2Block>(
                            std::make_shared<DepthwiseSeparable>(input, output))
                      : std::static_pointer_cast<MobileNetV2Block>(
                            std::make_shared<InvertedResidual>(
                                index == 0 ? input : output, output,
                                index == 0 ? stride : 1));
      blocks_.push_back(
          register_module(std::to_string(index), std::move(block)));
    }
  }
  Tensor forward(Tensor input) {
    for (const auto& block : blocks_) input = block->forward(input);
    return input;
  }

 private:
  std::vector<std::shared_ptr<MobileNetV2Block>> blocks_;
};
}  // namespace

class MobileNetV2Blocks final : public torch::nn::Module {
 public:
  MobileNetV2Blocks() {
    constexpr std::array<std::int64_t, 7> kCount{1, 2, 3, 4, 3, 3, 1};
    constexpr std::array<std::int64_t, 7> kOutput{16, 24, 32, 64, 96, 160, 320};
    constexpr std::array<std::int64_t, 7> kStride{1, 2, 2, 2, 1, 2, 1};
    std::int64_t input = 32;
    for (std::size_t index = 0; index < stages_.size(); ++index) {
      stages_[index] = register_module(std::to_string(index),
                                       std::make_shared<MobileNetV2Stage>(
                                           input, kOutput[index], kCount[index],
                                           kStride[index], index == 0));
      input = kOutput[index];
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
  std::array<std::shared_ptr<MobileNetV2Stage>, 7> stages_;
};

MobileNetV2Impl::MobileNetV2Impl(MobileNetV2Options options)
    : options_(options) {
  detail::Backend("mobilenetv2.construct", [&] {
    if (options.num_classes <= 0 || options.num_classes > 1000000 ||
        options.input_channels <= 0 || options.input_channels > 1024)
      detail::Invalid("mobilenetv2.options",
                      "classes 1..1000000 and channels 1..1024");
    conv_stem_ =
        register_module("conv_stem", Conv(options.input_channels, 32, 3, 2));
    bn1_ = register_module("bn1", Norm(32));
    blocks_ = register_module("blocks", std::make_shared<MobileNetV2Blocks>());
    conv_head_ = register_module("conv_head", Conv(320, 1280, 1));
    bn2_ = register_module("bn2", Norm(1280));
    classifier_ = register_module("classifier",
                                  torch::nn::Linear(1280, options.num_classes));
  });
}

std::vector<Tensor> MobileNetV2Impl::forward_features(const Tensor& input) {
  return detail::Backend("mobilenetv2.forward_features", [&] {
    detail::FloatingInput(input, conv_stem_->weight, 4);
    if (input.size(1) != options_.input_channels)
      detail::Invalid("mobilenetv2.input",
                      "the configured image channel count");
    return blocks_->forward_features(Relu6(bn1_(conv_stem_(input))));
  });
}

Tensor MobileNetV2Impl::forward(const Tensor& input) {
  return detail::Backend("mobilenetv2.forward", [&] {
    const auto stages = forward_features(input);
    const auto features = Relu6(bn2_(conv_head_(stages.back())));
    return classifier_(features.mean({2, 3}));
  });
}

const MobileNetV2Options& MobileNetV2Impl::options() const noexcept {
  return options_;
}

MobileNetV2 mobilenetv2_100(MobileNetV2Options options) {
  return MobileNetV2(options);
}

nn::ModelAdapter adapt(MobileNetV2 model) {
  if (!model) detail::Invalid("mobilenetv2.adapt", "a nonempty model");
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
