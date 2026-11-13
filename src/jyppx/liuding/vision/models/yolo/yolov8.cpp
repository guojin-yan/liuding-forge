// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/vision/models/yolo/yolov8.hpp>

#include <torch/nn/functional/upsampling.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace jyppx::liuding::models::yolo {
namespace {

struct Scale {
  double depth;
  double width;
  std::int64_t cap;
  const char* name;
};

Scale ResolveScale(YoloV8Capacity capacity) {
  switch (capacity) {
    case YoloV8Capacity::nano_capacity:
      return {0.33, 0.25, 1024, "n"};
    case YoloV8Capacity::small_capacity:
      return {0.33, 0.50, 1024, "s"};
    case YoloV8Capacity::medium_capacity:
      return {0.67, 0.75, 768, "m"};
    case YoloV8Capacity::large_capacity:
      return {1.00, 1.00, 512, "l"};
    case YoloV8Capacity::extra_large_capacity:
      return {1.00, 1.25, 512, "x"};
  }
  throw InvalidArgument("Unknown YOLOv8 capacity");
}

class ConvBlockImpl final : public torch::nn::Module {
 public:
  ConvBlockImpl(std::int64_t input, std::int64_t output,
                std::int64_t kernel = 3, std::int64_t stride = 1) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                                         input, output, kernel)
                                                         .stride(stride)
                                                         .padding(kernel / 2)
                                                         .bias(false)));
    bn = register_module(
        "bn",
        torch::nn::BatchNorm2d(
            torch::nn::BatchNorm2dOptions(output).eps(1e-3).momentum(0.03)));
  }
  Tensor forward(Tensor value) { return torch::silu(bn(conv(value))); }

 private:
  torch::nn::Conv2d conv{nullptr};
  torch::nn::BatchNorm2d bn{nullptr};
};
TORCH_MODULE(ConvBlock);

class BottleneckImpl final : public torch::nn::Module {
 public:
  BottleneckImpl(std::int64_t channels, bool shortcut) : shortcut_(shortcut) {
    first_ = register_module("cv1", ConvBlock(channels, channels));
    second_ = register_module("cv2", ConvBlock(channels, channels));
  }
  Tensor forward(Tensor value) {
    auto output = second_(first_(value));
    return shortcut_ ? output + value : output;
  }

 private:
  ConvBlock first_{nullptr}, second_{nullptr};
  bool shortcut_;
};
TORCH_MODULE(Bottleneck);

class C2fImpl final : public torch::nn::Module {
 public:
  C2fImpl(std::int64_t input, std::int64_t output, int repeats, bool shortcut) {
    hidden_ = output / 2;
    first_ = register_module("cv1", ConvBlock(input, hidden_ * 2, 1));
    last_ =
        register_module("cv2", ConvBlock((repeats + 2) * hidden_, output, 1));
    blocks_ = register_module("m", torch::nn::ModuleList());
    for (int index = 0; index < repeats; ++index) {
      blocks_->push_back(Bottleneck(hidden_, shortcut));
    }
  }
  Tensor forward(Tensor value) {
    auto chunks = first_(value).split(hidden_, 1);
    std::vector<Tensor> features{chunks[0], chunks[1]};
    features.reserve(blocks_->size() + 2);
    for (auto& block : *blocks_) {
      features.push_back(block->as<BottleneckImpl>()->forward(features.back()));
    }
    return last_(torch::cat(features, 1));
  }

 private:
  std::int64_t hidden_;
  ConvBlock first_{nullptr}, last_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
};
TORCH_MODULE(C2f);

class SppfImpl final : public torch::nn::Module {
 public:
  explicit SppfImpl(std::int64_t channels) {
    first_ = register_module("cv1", ConvBlock(channels, channels / 2, 1));
    last_ = register_module("cv2", ConvBlock(channels * 2, channels, 1));
    pool_ = register_module(
        "m", torch::nn::MaxPool2d(
                 torch::nn::MaxPool2dOptions(5).stride(1).padding(2)));
  }
  Tensor forward(Tensor value) {
    auto first = first_(value);
    auto second = pool_(first);
    auto third = pool_(second);
    return last_(torch::cat({first, second, third, pool_(third)}, 1));
  }

 private:
  ConvBlock first_{nullptr}, last_{nullptr};
  torch::nn::MaxPool2d pool_{nullptr};
};
TORCH_MODULE(Sppf);

class DetectImpl final : public torch::nn::Module {
 public:
  DetectImpl(std::array<std::int64_t, 3> channels,
             const YoloV8Options& options) {
    regression_ = register_module("cv2", torch::nn::ModuleList());
    classification_ = register_module("cv3", torch::nn::ModuleList());
    const auto box_width =
        (std::max)({std::int64_t{16}, channels[0] / 4, options.reg_max * 4});
    const auto class_width = (std::max)(
        channels[0], (std::min)(options.num_classes, std::int64_t{100}));
    const torch::NoGradGuard no_grad;
    for (std::size_t index = 0; index < channels.size(); ++index) {
      auto box_output = torch::nn::Conv2d(
          torch::nn::Conv2dOptions(box_width, options.reg_max * 4, 1));
      auto class_output = torch::nn::Conv2d(
          torch::nn::Conv2dOptions(class_width, options.num_classes, 1));
      box_output->bias.fill_(1.0);
      const double stride = static_cast<double>(8 << index);
      class_output->bias.fill_(
          std::log(5.0 / options.num_classes / std::pow(640.0 / stride, 2.0)));
      regression_->push_back(
          torch::nn::Sequential(ConvBlock(channels[index], box_width),
                                ConvBlock(box_width, box_width), box_output));
      classification_->push_back(torch::nn::Sequential(
          ConvBlock(channels[index], class_width),
          ConvBlock(class_width, class_width), class_output));
    }
  }
  std::array<Tensor, 3> forward(const std::array<Tensor, 3>& features) {
    std::array<Tensor, 3> output;
    for (std::size_t index = 0; index < features.size(); ++index) {
      output[index] = torch::cat(
          {regression_->at<torch::nn::SequentialImpl>(index).forward(
               features[index]),
           classification_->at<torch::nn::SequentialImpl>(index).forward(
               features[index])},
          1);
    }
    return output;
  }

 private:
  torch::nn::ModuleList regression_{nullptr}, classification_{nullptr};
};
TORCH_MODULE(Detect);

Tensor Upsample(const Tensor& value) {
  return torch::nn::functional::interpolate(
      value, torch::nn::functional::InterpolateFuncOptions()
                 .scale_factor(std::vector<double>{2.0, 2.0})
                 .mode(torch::kNearest));
}

}  // namespace

struct YoloV8Impl::Impl {
  YoloV8Options options;
  torch::nn::ModuleList layers{nullptr};
  ConvBlock cv0{nullptr}, cv1{nullptr}, cv3{nullptr}, cv5{nullptr},
      cv7{nullptr};
  ConvBlock cv16{nullptr}, cv19{nullptr};
  C2f c2{nullptr}, c4{nullptr}, c6{nullptr}, c8{nullptr};
  C2f c12{nullptr}, c15{nullptr}, c18{nullptr}, c21{nullptr};
  Sppf pool{nullptr};
  Detect head{nullptr};

  Impl(YoloV8Impl& owner, const YoloV8Options& selected) : options(selected) {
    const auto scale = ResolveScale(options.capacity);
    if (options.num_classes < 1 || options.num_classes > 10000 ||
        options.reg_max < 2 || options.reg_max > 32 ||
        options.input_channels < 1 || options.input_channels > 16) {
      throw InvalidArgument(
          "Invalid YOLOv8 classes, input channels or regression bins");
    }
    const auto channel = [scale](std::int64_t base) {
      return static_cast<std::int64_t>(
                 std::ceil((std::min)(base, scale.cap) * scale.width / 8.0)) *
             8;
    };
    const auto repeats = [scale](int count) {
      return (std::max)(1, static_cast<int>(std::round(count * scale.depth)));
    };
    const auto a = channel(64), b = channel(128), c = channel(256);
    const auto d = channel(512), e = channel(1024);
    layers = owner.register_module("model", torch::nn::ModuleList());
    layers->push_back(cv0 = ConvBlock(options.input_channels, a, 3, 2));
    layers->push_back(cv1 = ConvBlock(a, b, 3, 2));
    layers->push_back(c2 = C2f(b, b, repeats(3), true));
    layers->push_back(cv3 = ConvBlock(b, c, 3, 2));
    layers->push_back(c4 = C2f(c, c, repeats(6), true));
    layers->push_back(cv5 = ConvBlock(c, d, 3, 2));
    layers->push_back(c6 = C2f(d, d, repeats(6), true));
    layers->push_back(cv7 = ConvBlock(d, e, 3, 2));
    layers->push_back(c8 = C2f(e, e, repeats(3), true));
    layers->push_back(pool = Sppf(e));
    layers->push_back(torch::nn::Identity());  // Parameter-free upsample.
    layers->push_back(torch::nn::Identity());  // Parameter-free concat.
    layers->push_back(c12 = C2f(e + d, d, repeats(3), false));
    layers->push_back(torch::nn::Identity());
    layers->push_back(torch::nn::Identity());
    layers->push_back(c15 = C2f(d + c, c, repeats(3), false));
    layers->push_back(cv16 = ConvBlock(c, c, 3, 2));
    layers->push_back(torch::nn::Identity());
    layers->push_back(c18 = C2f(c + d, d, repeats(3), false));
    layers->push_back(cv19 = ConvBlock(d, d, 3, 2));
    layers->push_back(torch::nn::Identity());
    layers->push_back(c21 = C2f(d + e, e, repeats(3), false));
    layers->push_back(
        head = Detect(std::array<std::int64_t, 3>{c, d, e}, options));
  }
};

YoloV8Impl::YoloV8Impl(const YoloV8Options& options)
    : impl_(std::make_unique<Impl>(*this, options)) {}
YoloV8Impl::~YoloV8Impl() = default;

const YoloV8Options& YoloV8Impl::options() const noexcept {
  return impl_->options;
}

std::array<Tensor, 3> YoloV8Impl::forward_levels(const Tensor& images) {
  if (!images.defined() || images.layout() != torch::kStrided ||
      images.dim() != 4 || !images.is_floating_point() || images.size(0) <= 0 ||
      images.size(1) != impl_->options.input_channels || images.size(2) <= 0 ||
      images.size(3) <= 0 || images.size(2) % 32 != 0 ||
      images.size(3) % 32 != 0) {
    throw InvalidArgument(
        "YOLOv8 requires floating NCHW input with H/W divisible by 32");
  }
  auto& state = *impl_;
  auto value = state.c2(state.cv1(state.cv0(images)));
  auto p3 = state.c4(state.cv3(value));
  auto p4 = state.c6(state.cv5(p3));
  auto p5 = state.pool(state.c8(state.cv7(p4)));
  auto top4 = state.c12(torch::cat({Upsample(p5), p4}, 1));
  auto out3 = state.c15(torch::cat({Upsample(top4), p3}, 1));
  auto out4 = state.c18(torch::cat({state.cv16(out3), top4}, 1));
  auto out5 = state.c21(torch::cat({state.cv19(out4), p5}, 1));
  return state.head->forward({out3, out4, out5});
}

Tensor YoloV8Impl::forward(const Tensor& images) {
  auto levels = forward_levels(images);
  return torch::cat(
      {levels[0].flatten(2), levels[1].flatten(2), levels[2].flatten(2)}, 2);
}

nn::ModelAdapter create_yolov8(const YoloV8Options& options) {
  return nn::adapt_tensor_module(std::make_shared<YoloV8Impl>(options),
                                 TensorSchema{.rank = 4},
                                 TensorSchema{.rank = 3});
}

std::string yolov8_model_id(YoloV8Capacity capacity) {
  return std::string("yolo.ultralytics.v8.") + ResolveScale(capacity).name;
}

}  // namespace jyppx::liuding::models::yolo
