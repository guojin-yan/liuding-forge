// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/liuding/models/vision.hpp>

#include <torch/nn/init.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/pooling.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
namespace jyppx::liuding::models {
namespace {
torch::nn::Conv2d Conv(std::int64_t in, std::int64_t out, std::int64_t kernel,
                       std::int64_t stride = 1, std::int64_t groups = 1,
                       bool bias = false) {
  return torch::nn::Conv2d(torch::nn::Conv2dOptions(in, out, kernel)
                               .stride(stride)
                               .padding((kernel - 1) / 2)
                               .groups(groups)
                               .bias(bias));
}
class BasicBlock final : public torch::nn::Module {
 public:
  BasicBlock(std::int64_t in, std::int64_t out, std::int64_t stride) {
    conv1 = register_module("conv1", Conv(in, out, 3, stride));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(out));
    conv2 = register_module("conv2", Conv(out, out, 3));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(out));
    if (stride != 1 || in != out)
      downsample = register_module(
          "downsample", torch::nn::Sequential(Conv(in, out, 1, stride),
                                              torch::nn::BatchNorm2d(out)));
  }
  Tensor forward(const Tensor& input) {
    auto residual = downsample ? downsample->forward(input) : input;
    auto output = torch::relu(bn1(conv1(input)));
    output = bn2(conv2(output));
    return torch::relu(output + residual);
  }
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
  torch::nn::Sequential downsample{nullptr};
};
class Bottleneck final : public torch::nn::Module {
 public:
  Bottleneck(std::int64_t in, std::int64_t planes, std::int64_t stride,
             std::int64_t groups, std::int64_t base_width) {
    const auto width = planes * base_width / 64 * groups;
    conv1 = register_module("conv1", Conv(in, width, 1));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(width));
    // torchvision v1.5 puts the downsampling stride in the 3x3 convolution.
    conv2 = register_module("conv2", Conv(width, width, 3, stride, groups));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(width));
    conv3 = register_module("conv3", Conv(width, planes * 4, 1));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(planes * 4));
    if (stride != 1 || in != planes * 4)
      downsample = register_module(
          "downsample",
          torch::nn::Sequential(Conv(in, planes * 4, 1, stride),
                                torch::nn::BatchNorm2d(planes * 4)));
  }
  Tensor forward(const Tensor& input) {
    auto residual = downsample ? downsample->forward(input) : input;
    auto output = torch::relu(bn1(conv1(input)));
    output = torch::relu(bn2(conv2(output)));
    output = bn3(conv3(output));
    return torch::relu(output + residual);
  }
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
  torch::nn::Sequential downsample{nullptr};
};
torch::nn::Sequential Stage(bool bottleneck, std::int64_t& input,
                            std::int64_t planes, std::int64_t count,
                            std::int64_t stride, std::int64_t groups = 1,
                            std::int64_t base_width = 64) {
  torch::nn::Sequential result;
  for (std::int64_t i = 0; i < count; ++i) {
    const auto block_stride = i == 0 ? stride : 1;
    if (bottleneck)
      result->push_back(
          std::make_shared<Bottleneck>(input, planes, block_stride, groups,
                                       base_width));
    else
      result->push_back(
          std::make_shared<BasicBlock>(input, planes, block_stride));
    input = planes * (bottleneck ? 4 : 1);
  }
  return result;
}
void InitVision(torch::nn::Module& model, bool mobile) {
  torch::NoGradGuard guard;
  for (const auto& module : model.modules(false)) {
    if (const auto conv =
            std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(module)) {
      torch::nn::init::kaiming_normal_(conv->weight, 0, torch::kFanOut,
                                       torch::kReLU);
      if (conv->bias.defined()) torch::nn::init::zeros_(conv->bias);
    } else if (const auto bn =
                   std::dynamic_pointer_cast<torch::nn::BatchNorm2dImpl>(
                       module)) {
      torch::nn::init::ones_(bn->weight);
      torch::nn::init::zeros_(bn->bias);
    } else if (mobile) {
      if (const auto linear =
              std::dynamic_pointer_cast<torch::nn::LinearImpl>(module)) {
        torch::nn::init::normal_(linear->weight, 0, 0.01);
        torch::nn::init::zeros_(linear->bias);
      }
    }
  }
}
nn::ModelAdapter ImageAdapter(std::shared_ptr<torch::nn::Module> module,
                              nn::ModelInvoker invoker, std::int64_t channels,
                              std::int64_t classes) {
  TensorSchema input;
  input.rank = 4;
  input.shape = {{std::nullopt, channels, std::nullopt, std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, classes}};
  return nn::adapt_module(
      std::move(module), std::move(invoker),
      {.positional = {ValueSchema::tensor(input, "images",
                                          "normalized_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "logits", "classification_logits"));
}
std::int64_t Divisible(double value, std::int64_t divisor = 8) {
  auto result = std::max(
      divisor,
      static_cast<std::int64_t>(value + static_cast<double>(divisor) / 2.0) /
          divisor * divisor);
  if (static_cast<double>(result) < 0.9 * value) result += divisor;
  return result;
}
enum class Activation { none, relu, hard_swish };
class HardSwish final : public torch::nn::Module {
 public:
  Tensor forward(const Tensor& x) { return torch::hardswish(x); }
};
// AnyModule cannot erase SequentialImpl's templated forward. A concrete
// forwarding member preserves the numeric child names without an extra layer.
class MobileConvBlock final : public torch::nn::SequentialImpl {
 public:
  Tensor forward(const Tensor& input) {
    return torch::nn::SequentialImpl::forward(input);
  }
};
std::shared_ptr<MobileConvBlock> MobileConv(std::int64_t in, std::int64_t out,
                                            std::int64_t kernel,
                                            std::int64_t stride,
                                            std::int64_t groups,
                                            Activation activation) {
  auto result = std::make_shared<MobileConvBlock>();
  result->push_back(Conv(in, out, kernel, stride, groups));
  result->push_back(torch::nn::BatchNorm2d(
      torch::nn::BatchNorm2dOptions(out).eps(0.001).momentum(0.01)));
  if (activation == Activation::relu) result->push_back(torch::nn::ReLU());
  if (activation == Activation::hard_swish)
    result->push_back(std::make_shared<HardSwish>());
  return result;
}
class SqueezeExcitation final : public torch::nn::Module {
 public:
  explicit SqueezeExcitation(std::int64_t channels) {
    const auto squeeze = Divisible(static_cast<double>(channels) / 4.0);
    fc1 = register_module("fc1", Conv(channels, squeeze, 1, 1, 1, true));
    fc2 = register_module("fc2", Conv(squeeze, channels, 1, 1, 1, true));
  }
  Tensor forward(const Tensor& input) {
    auto scale = input.mean({2, 3}, true);
    scale = fc2(torch::relu(fc1(scale)));
    return input * torch::hardsigmoid(scale);
  }
  torch::nn::Conv2d fc1{nullptr}, fc2{nullptr};
};
struct MobileBlockConfig {
  std::int64_t input, kernel, expanded, output;
  bool squeeze, hard_swish;
  std::int64_t stride;
};
class InvertedResidual final : public torch::nn::Module {
 public:
  explicit InvertedResidual(const MobileBlockConfig& c)
      : residual_(c.stride == 1 && c.input == c.output) {
    block = register_module("block", torch::nn::Sequential());
    const auto activation =
        c.hard_swish ? Activation::hard_swish : Activation::relu;
    if (c.expanded != c.input)
      block->push_back(MobileConv(c.input, c.expanded, 1, 1, 1, activation));
    block->push_back(MobileConv(c.expanded, c.expanded, c.kernel, c.stride,
                                c.expanded, activation));
    if (c.squeeze)
      block->push_back(std::make_shared<SqueezeExcitation>(c.expanded));
    block->push_back(
        MobileConv(c.expanded, c.output, 1, 1, 1, Activation::none));
  }
  Tensor forward(const Tensor& input) {
    auto output = block->forward(input);
    return residual_ ? output + input : output;
  }
  torch::nn::Sequential block{nullptr};

 private:
  bool residual_;
};
std::vector<MobileBlockConfig> Config(MobileNetV3Variant variant) {
  if (variant == MobileNetV3Variant::small_variant)
    return {{16, 3, 16, 16, true, false, 2},  {16, 3, 72, 24, false, false, 2},
            {24, 3, 88, 24, false, false, 1}, {24, 5, 96, 40, true, true, 2},
            {40, 5, 240, 40, true, true, 1},  {40, 5, 240, 40, true, true, 1},
            {40, 5, 120, 48, true, true, 1},  {48, 5, 144, 48, true, true, 1},
            {48, 5, 288, 96, true, true, 2},  {96, 5, 576, 96, true, true, 1},
            {96, 5, 576, 96, true, true, 1}};
  if (variant == MobileNetV3Variant::large_variant)
    return {
        {16, 3, 16, 16, false, false, 1},  {16, 3, 64, 24, false, false, 2},
        {24, 3, 72, 24, false, false, 1},  {24, 5, 72, 40, true, false, 2},
        {40, 5, 120, 40, true, false, 1},  {40, 5, 120, 40, true, false, 1},
        {40, 3, 240, 80, false, true, 2},  {80, 3, 200, 80, false, true, 1},
        {80, 3, 184, 80, false, true, 1},  {80, 3, 184, 80, false, true, 1},
        {80, 3, 480, 112, true, true, 1},  {112, 3, 672, 112, true, true, 1},
        {112, 5, 672, 160, true, true, 2}, {160, 5, 960, 160, true, true, 1},
        {160, 5, 960, 160, true, true, 1}};
  detail::Invalid("mobilenet.variant", "small or large");
}
void ImageOptions(std::int64_t classes, std::int64_t channels) {
  if (classes <= 0 || classes > 1000000 || channels <= 0 || channels > 1024)
    detail::Invalid("image.options", "classes 1..1000000 and channels 1..1024");
}
}  // namespace
ResNetImpl::ResNetImpl(ResNetVariant variant, ResNetOptions options)
    : options_(options) {
  detail::Backend("resnet.construct", [&] {
    ImageOptions(options.num_classes, options.input_channels);
    std::array<std::int64_t, 4> schedule;
    bool bottleneck = false;
    std::int64_t groups = 1;
    std::int64_t base_width = 64;
    switch (variant) {
      case ResNetVariant::resnet18:
        schedule = {2, 2, 2, 2};
        break;
      case ResNetVariant::resnet34:
        schedule = {3, 4, 6, 3};
        break;
      case ResNetVariant::resnet50:
        schedule = {3, 4, 6, 3};
        bottleneck = true;
        break;
      case ResNetVariant::resnet101:
        schedule = {3, 4, 23, 3};
        bottleneck = true;
        break;
      case ResNetVariant::resnext50_32x4d:
        schedule = {3, 4, 6, 3};
        bottleneck = true;
        groups = 32;
        base_width = 4;
        break;
      default:
        detail::Invalid("resnet.variant",
                        "resnet18, resnet34, resnet50, resnet101 or resnext50_32x4d");
    }
    conv1_ = register_module("conv1", Conv(options.input_channels, 64, 7, 2));
    bn1_ = register_module("bn1", torch::nn::BatchNorm2d(64));
    std::int64_t input = 64;
    layer1_ =
        register_module("layer1", Stage(bottleneck, input, 64, schedule[0], 1,
                                         groups, base_width));
    layer2_ = register_module("layer2",
                              Stage(bottleneck, input, 128, schedule[1], 2,
                                    groups, base_width));
    layer3_ = register_module("layer3",
                              Stage(bottleneck, input, 256, schedule[2], 2,
                                    groups, base_width));
    layer4_ = register_module("layer4",
                              Stage(bottleneck, input, 512, schedule[3], 2,
                                    groups, base_width));
    fc_ = register_module("fc", torch::nn::Linear(input, options.num_classes));
    InitVision(*this, false);
    if (options.zero_init_residual) {
      torch::NoGradGuard guard;
      for (const auto& module : modules(false)) {
        if (const auto block = std::dynamic_pointer_cast<BasicBlock>(module))
          torch::nn::init::zeros_(block->bn2->weight);
        if (const auto block = std::dynamic_pointer_cast<Bottleneck>(module))
          torch::nn::init::zeros_(block->bn3->weight);
      }
    }
  });
}
std::vector<Tensor> ResNetImpl::forward_features(const Tensor& input) {
  return detail::Backend("resnet.forward_features", [&] {
    detail::FloatingInput(input, conv1_->weight, 4);
    if (input.size(1) != options_.input_channels)
      detail::Invalid("resnet.channels", "the configured image channel count");
    auto x = torch::max_pool2d(torch::relu(bn1_(conv1_(input))), {3, 3}, {2, 2},
                               {1, 1});
    auto a = layer1_->forward(x);
    auto b = layer2_->forward(a);
    auto c = layer3_->forward(b);
    auto d = layer4_->forward(c);
    return std::vector<Tensor>{a, b, c, d};
  });
}
Tensor ResNetImpl::forward(const Tensor& input) {
  return detail::Backend("resnet.forward", [&] {
    auto stages = forward_features(input);
    return fc_(stages.back().mean({2, 3}));
  });
}
const ResNetOptions& ResNetImpl::options() const noexcept { return options_; }
ResNet resnet18(ResNetOptions options) {
  return ResNet(ResNetVariant::resnet18, options);
}
ResNet resnet34(ResNetOptions options) {
  return ResNet(ResNetVariant::resnet34, options);
}
ResNet resnet50(ResNetOptions options) {
  return ResNet(ResNetVariant::resnet50, options);
}
ResNet resnet101(ResNetOptions options) {
  return ResNet(ResNetVariant::resnet101, options);
}
ResNet resnext50_32x4d(ResNetOptions options) {
  return ResNet(ResNetVariant::resnext50_32x4d, options);
}
nn::ModelAdapter adapt(ResNet model) {
  if (!model) detail::Invalid("resnet.adapt", "a nonempty model");
  const auto options = model->options();
  auto module = model.ptr();
  return ImageAdapter(
      module,
      [model](const nn::Invocation& input) mutable {
        return c10::IValue(model->forward(input.positional.front().toTensor()));
      },
      options.input_channels, options.num_classes);
}
MobileNetV3Impl::MobileNetV3Impl(MobileNetV3Variant variant,
                                 MobileNetV3Options options)
    : options_(options) {
  detail::Backend("mobilenet.construct", [&] {
    ImageOptions(options.num_classes, options.input_channels);
    if (!std::isfinite(options.width_multiplier) ||
        options.width_multiplier < 0.25 || options.width_multiplier > 2.0 ||
        !std::isfinite(options.dropout) || options.dropout < 0 ||
        options.dropout >= 1)
      detail::Invalid("mobilenet.options",
                      "width in [0.25,2] and dropout in [0,1)");
    auto config = Config(variant);
    for (auto& block : config) {
      block.input = Divisible(static_cast<double>(block.input) *
                              options.width_multiplier);
      block.expanded = Divisible(static_cast<double>(block.expanded) *
                                 options.width_multiplier);
      block.output = Divisible(static_cast<double>(block.output) *
                               options.width_multiplier);
    }
    features_ = register_module("features", torch::nn::Sequential());
    features_->push_back(MobileConv(options.input_channels,
                                    config.front().input, 3, 2, 1,
                                    Activation::hard_swish));
    for (const auto& block : config)
      features_->push_back(std::make_shared<InvertedResidual>(block));
    const auto last_conv = config.back().output * 6;
    features_->push_back(MobileConv(config.back().output, last_conv, 1, 1, 1,
                                    Activation::hard_swish));
    const auto last_channel = Divisible(
        (variant == MobileNetV3Variant::small_variant ? 1024.0 : 1280.0) *
        options.width_multiplier);
    classifier_ = register_module("classifier", torch::nn::Sequential());
    classifier_->push_back(torch::nn::Linear(last_conv, last_channel));
    classifier_->push_back(std::make_shared<HardSwish>());
    classifier_->push_back(torch::nn::Dropout(options.dropout));
    classifier_->push_back(
        torch::nn::Linear(last_channel, options.num_classes));
    InitVision(*this, true);
  });
}
Tensor MobileNetV3Impl::forward_features(const Tensor& input) {
  return detail::Backend("mobilenet.forward_features", [&] {
    const auto& stem = features_->at<MobileConvBlock>(0);
    const auto& conv = stem.at<torch::nn::Conv2dImpl>(0);
    detail::FloatingInput(input, conv.weight, 4);
    if (input.size(1) != options_.input_channels)
      detail::Invalid("mobilenet.channels",
                      "the configured image channel count");
    return features_->forward(input);
  });
}
Tensor MobileNetV3Impl::forward(const Tensor& input) {
  return detail::Backend("mobilenet.forward", [&] {
    return classifier_->forward(forward_features(input).mean({2, 3}));
  });
}
const MobileNetV3Options& MobileNetV3Impl::options() const noexcept {
  return options_;
}
MobileNetV3 mobilenet_v3_small(MobileNetV3Options options) {
  return MobileNetV3(MobileNetV3Variant::small_variant, options);
}
MobileNetV3 mobilenet_v3_large(MobileNetV3Options options) {
  return MobileNetV3(MobileNetV3Variant::large_variant, options);
}
nn::ModelAdapter adapt(MobileNetV3 model) {
  if (!model) detail::Invalid("mobilenet.adapt", "a nonempty model");
  const auto options = model->options();
  auto module = model.ptr();
  return ImageAdapter(
      module,
      [model](const nn::Invocation& input) mutable {
        return c10::IValue(model->forward(input.positional.front().toTensor()));
      },
      options.input_channels, options.num_classes);
}
}  // namespace jyppx::liuding::models
