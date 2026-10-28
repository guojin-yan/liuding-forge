// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "head_utils.hpp"

#include <jyppx/liuding/models/densenet.hpp>

#include <torch/nn/init.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/pooling.h>
#include <torch/torch.h>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace jyppx::liuding::models {
namespace {
torch::nn::Conv2d Conv(std::int64_t input, std::int64_t output,
                       std::int64_t kernel, std::int64_t stride = 1) {
  return torch::nn::Conv2d(torch::nn::Conv2dOptions(input, output, kernel)
                               .stride(stride)
                               .padding((kernel - 1) / 2)
                               .bias(false));
}

class DenseLayer final : public torch::nn::Module {
 public:
  explicit DenseLayer(std::int64_t input) {
    norm1_ = register_module("norm1", torch::nn::BatchNorm2d(input));
    conv1_ = register_module("conv1", Conv(input, 128, 1));
    norm2_ = register_module("norm2", torch::nn::BatchNorm2d(128));
    conv2_ = register_module("conv2", Conv(128, 32, 3));
  }
  Tensor forward(const Tensor& input) {
    auto bottleneck = conv1_(torch::relu(norm1_(input)));
    return conv2_(torch::relu(norm2_(bottleneck)));
  }

 private:
  torch::nn::BatchNorm2d norm1_{nullptr}, norm2_{nullptr};
  torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
};

class DenseBlock final : public torch::nn::Module {
 public:
  DenseBlock(std::int64_t input, std::int64_t count) {
    layers_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
      auto layer = std::make_shared<DenseLayer>(input + 32 * index);
      layers_.push_back(register_module(
          "denselayer" + std::to_string(index + 1), std::move(layer)));
    }
  }
  Tensor forward(const Tensor& input) {
    std::vector<Tensor> features;
    features.reserve(layers_.size() + 1);
    features.push_back(input);
    for (const auto& layer : layers_) {
      features.push_back(layer->forward(torch::cat(features, 1)));
    }
    return torch::cat(features, 1);
  }

 private:
  std::vector<std::shared_ptr<DenseLayer>> layers_;
};

class Transition final : public torch::nn::Module {
 public:
  Transition(std::int64_t input, std::int64_t output) {
    norm_ = register_module("norm", torch::nn::BatchNorm2d(input));
    conv_ = register_module("conv", Conv(input, output, 1));
  }
  Tensor forward(const Tensor& input) {
    return torch::avg_pool2d(conv_(torch::relu(norm_(input))), {2, 2}, {2, 2});
  }

 private:
  torch::nn::BatchNorm2d norm_{nullptr};
  torch::nn::Conv2d conv_{nullptr};
};

void Initialize(DenseNetImpl& model) {
  torch::NoGradGuard guard;
  for (const auto& module : model.modules(false)) {
    if (const auto conv =
            std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(module)) {
      torch::nn::init::kaiming_normal_(conv->weight, 0, torch::kFanOut,
                                       torch::kReLU);
    } else if (const auto bn =
                   std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(
                       module)) {
      torch::nn::init::ones_(bn->weight);
      torch::nn::init::zeros_(bn->bias);
    } else if (const auto linear =
                   std::dynamic_pointer_cast<torch::nn::LinearImpl>(module)) {
      torch::nn::init::zeros_(linear->bias);
    }
  }
}
}  // namespace

class DenseNetFeatures final : public torch::nn::Module {
 public:
  explicit DenseNetFeatures(std::int64_t input_channels) {
    conv0_ = register_module("conv0", Conv(input_channels, 64, 7, 2));
    norm0_ = register_module("norm0", torch::nn::BatchNorm2d(64));
    relu0_ = register_module("relu0", torch::nn::ReLU());
    pool0_ = register_module(
        "pool0", torch::nn::MaxPool2d(
                     torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
    std::int64_t channels = 64;
    constexpr std::array<std::int64_t, 4> kLayers{6, 12, 24, 16};
    for (std::size_t index = 0; index < kLayers.size(); ++index) {
      blocks_[index] = register_module(
          "denseblock" + std::to_string(index + 1),
          std::make_shared<DenseBlock>(channels, kLayers[index]));
      channels += kLayers[index] * 32;
      if (index + 1 < kLayers.size()) {
        transitions_[index] = register_module(
            "transition" + std::to_string(index + 1),
            std::make_shared<Transition>(channels, channels / 2));
        channels /= 2;
      }
    }
    norm5_ = register_module("norm5", torch::nn::BatchNorm2d(channels));
  }

  const Tensor& stem_weight() const { return conv0_->weight; }
  std::vector<Tensor> forward_stages(const Tensor& input) {
    auto x = pool0_(relu0_(norm0_(conv0_(input))));
    std::vector<Tensor> stages;
    stages.reserve(blocks_.size());
    for (std::size_t index = 0; index < blocks_.size(); ++index) {
      x = blocks_[index]->forward(x);
      stages.push_back(x);
      if (index < transitions_.size()) x = transitions_[index]->forward(x);
    }
    return stages;
  }
  Tensor normalize_output(const Tensor& input) { return norm5_(input); }

 private:
  torch::nn::Conv2d conv0_{nullptr};
  torch::nn::BatchNorm2d norm0_{nullptr}, norm5_{nullptr};
  torch::nn::ReLU relu0_{nullptr};
  torch::nn::MaxPool2d pool0_{nullptr};
  std::array<std::shared_ptr<DenseBlock>, 4> blocks_;
  std::array<std::shared_ptr<Transition>, 3> transitions_;
};

DenseNetImpl::DenseNetImpl(DenseNetOptions options) : options_(options) {
  detail::Backend("densenet.construct", [&] {
    if (options.num_classes < 0 || options.num_classes > 1000000 ||
        options.input_channels <= 0 || options.input_channels > 1024)
      detail::Invalid("densenet.options",
                      "classes 0..1000000 and channels 1..1024");
    features_ = register_module(
        "features", std::make_shared<DenseNetFeatures>(options.input_channels));
    if (options.num_classes > 0) {
      classifier_ = register_module(
          "classifier", torch::nn::Linear(1024, options.num_classes));
    }
    Initialize(*this);
  });
}

std::vector<Tensor> DenseNetImpl::forward_features(const Tensor& input) {
  return detail::Backend("densenet.forward_features", [&] {
    detail::FloatingInput(input, features_->stem_weight(), 4);
    if (input.size(1) != options_.input_channels || input.size(2) < 29 ||
        input.size(3) < 29)
      detail::Invalid("densenet.input",
                      "configured channels and spatial extent >= 29");
    return features_->forward_stages(input);
  });
}

Tensor DenseNetImpl::forward(const Tensor& input) {
  return detail::Backend("densenet.forward", [&] {
    auto stages = forward_features(input);
    auto final = features_->normalize_output(stages.back());
    auto pooled = torch::relu(final).mean({2, 3});
    return classifier_ ? classifier_(pooled) : pooled;
  });
}

void DenseNetImpl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("densenet.reset_classifier", [&] {
    detail::ResetLinear(*this, classifier_, "classifier", 1024, num_classes,
                        features_->stem_weight());
    options_.num_classes = num_classes;
  });
}

const DenseNetOptions& DenseNetImpl::options() const noexcept {
  return options_;
}

DenseNet densenet121(DenseNetOptions options) { return DenseNet(options); }

nn::ModelAdapter adapt(DenseNet model) {
  if (!model) detail::Invalid("densenet.adapt", "a nonempty model");
  const auto options = model->options();
  TensorSchema input;
  input.rank = 4;
  input.shape = {
      {std::nullopt, options.input_channels, std::nullopt, std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {
      {std::nullopt, options.num_classes > 0 ? options.num_classes : 1024}};
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
      ValueSchema::tensor(output, "output",
                          options.num_classes > 0 ? "classification_logits"
                                                  : "image_embedding"));
}
}  // namespace jyppx::liuding::models
