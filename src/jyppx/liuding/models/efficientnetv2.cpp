// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "head_utils.hpp"

#include <jyppx/liuding/models/efficientnetv2.hpp>

#include <torch/torch.h>
#if defined(USE_ROCM)
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include <ATen/autocast_mode.h>
#include <ATen/ops/_batch_norm_impl_index.h>
#include <ATen/ops/_batch_norm_impl_index_backward.h>
#include <ATen/ops/silu_backward.h>
#endif

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

#if defined(USE_ROCM)
class RecomputedBnAct final
    : public torch::autograd::Function<RecomputedBnAct> {
 public:
  static Tensor forward(torch::autograd::AutogradContext* context, Tensor input,
                        Tensor weight, Tensor bias,
                        torch::nn::BatchNorm2d norm) {
    // Original module forward alone updates live running statistics/counter.
    auto normalized = norm->forward(input);
    context->save_for_backward({input, weight, bias});
    context->saved_data["eps"] = norm->options.eps();
    context->saved_data["momentum"] = *norm->options.momentum();
    context->saved_data["cudnn"] = at::globalContext().userEnabledCuDNN();
    return torch::silu(normalized);
  }
  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* context,
      torch::autograd::variable_list gradients) {
    TORCH_CHECK(
        !torch::GradMode::is_enabled(),
        "EfficientNetV2 BN activation recomputation is first-order only");
    TORCH_CHECK(
        !at::autocast::is_autocast_enabled(torch::kCUDA),
        "EfficientNetV2 BN activation recomputation rejects autocast backward");
    auto saved = context->get_saved_variables();
    // Batch-mode replay has no running estimates. Only the original forward
    // updates the module buffers; replay allocates no disposable estimates.
    auto replay = at::_batch_norm_impl_index(
        saved[0], saved[1], saved[2], std::nullopt, std::nullopt, true,
        context->saved_data["momentum"].toDouble(),
        context->saved_data["eps"].toDouble(),
        context->saved_data["cudnn"].toBool());
    auto gradient = at::silu_backward(gradients[0], std::get<0>(replay));
    const std::array<bool, 3> required{context->needs_input_grad(0),
                                       context->needs_input_grad(1),
                                       context->needs_input_grad(2)};
    auto result = at::_batch_norm_impl_index_backward(
        std::get<4>(replay), saved[0], gradient, saved[1], std::nullopt,
        std::nullopt, std::get<1>(replay), std::get<2>(replay), true,
        context->saved_data["eps"].toDouble(), required, std::get<3>(replay));
    return {std::get<0>(result), std::get<1>(result), std::get<2>(result),
            Tensor{}};
  }
};
#endif
Tensor BnAct(const Tensor& input, torch::nn::BatchNorm2d& norm, bool enabled) {
#if defined(USE_ROCM)
  if (enabled && input.is_cuda() && input.scalar_type() == torch::kFloat32 &&
      input.is_contiguous() && !input.is_conj() && !input.is_neg() &&
      norm->weight.defined() && norm->bias.defined() &&
      norm->weight.scalar_type() == torch::kFloat32 &&
      norm->bias.scalar_type() == torch::kFloat32 &&
      (input.requires_grad() || norm->weight.requires_grad() ||
       norm->bias.requires_grad()) &&
      norm->is_training() && norm->options.affine() &&
      norm->options.track_running_stats() &&
      norm->options.momentum().has_value() && torch::GradMode::is_enabled() &&
      !torch::jit::tracer::getTracingState() &&
      !at::autocast::is_autocast_enabled(torch::kCUDA) &&
      !at::globalContext().deterministicAlgorithms())
    return RecomputedBnAct::apply(input, norm->weight, norm->bias, norm);
#else
  static_cast<void>(enabled);
#endif
  return torch::silu(norm->forward(input));
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
  ConvBnAct(std::int64_t channels, bool recomputation)
      : recomputation_(recomputation) {
    conv_ = register_module("conv", Conv(channels, channels, 3));
    bn1_ = register_module("bn1", Norm(channels));
  }
  Tensor forward(const Tensor& input) override {
    return input + BnAct(conv_(input), bn1_, recomputation_);
  }

 private:
  bool recomputation_;
  torch::nn::Conv2d conv_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr};
};

class FusedMbConv final : public Block {
 public:
  FusedMbConv(std::int64_t input, std::int64_t output, std::int64_t stride,
              bool recomputation)
      : recomputation_(recomputation),
        skip_(stride == 1 && input == output),
        stride_(stride) {
    const auto expanded = input * 4;
    conv_exp_ = register_module("conv_exp", Conv(input, expanded, 3, stride));
    bn1_ = register_module("bn1", Norm(expanded));
    conv_pwl_ = register_module("conv_pwl", Conv(expanded, output, 1));
    bn2_ = register_module("bn2", Norm(output));
  }
  Tensor forward(const Tensor& input) override {
    auto value =
        BnAct(conv_exp_(SamePad(input, 3, stride_)), bn1_, recomputation_);
    value = bn2_(conv_pwl_(value));
    return skip_ ? value + input : value;
  }

 private:
  bool recomputation_;
  bool skip_;
  std::int64_t stride_;
  torch::nn::Conv2d conv_exp_{nullptr}, conv_pwl_{nullptr};
  torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};

class MbConv final : public Block {
 public:
  MbConv(std::int64_t input, std::int64_t output, std::int64_t expansion,
         std::int64_t stride, bool recomputation)
      : recomputation_(recomputation),
        skip_(stride == 1 && input == output),
        stride_(stride) {
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
    auto value = BnAct(conv_pw_(input), bn1_, recomputation_);
    value = BnAct(conv_dw_(SamePad(value, 3, stride_)), bn2_, recomputation_);
    value = bn3_(conv_pwl_(se_->forward(value)));
    return skip_ ? value + input : value;
  }

 private:
  bool recomputation_;
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
        bool fused_stage, bool recomputation) {
    blocks_.reserve(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
      const auto block_input = index == 0 ? input : output;
      const auto block_stride = index == 0 ? stride : 1;
      std::shared_ptr<Block> block;
      if (stem_stage)
        block = std::make_shared<ConvBnAct>(block_input, recomputation);
      else if (fused_stage)
        block = std::make_shared<FusedMbConv>(block_input, output, block_stride,
                                              recomputation);
      else
        block = std::make_shared<MbConv>(block_input, output, expansion,
                                         block_stride, recomputation);
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
  explicit EfficientNetV2Blocks(bool recomputation) {
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
                                  index == 1 || index == 2, recomputation));
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
    if (options.num_classes < 0 || options.num_classes > 1000000 ||
        options.input_channels <= 0 || options.input_channels > 1024)
      detail::Invalid("efficientnetv2.options",
                      "classes 0..1000000 and channels 1..1024");
    conv_stem_ =
        register_module("conv_stem", Conv(options.input_channels, 24, 3, 2));
    bn1_ = register_module("bn1", Norm(24));
    blocks_ = register_module("blocks", std::make_shared<EfficientNetV2Blocks>(
                                            options.activation_recomputation));
    conv_head_ = register_module("conv_head", Conv(256, 1280, 1));
    bn2_ = register_module("bn2", Norm(1280));
    if (options.num_classes > 0) {
      classifier_ = register_module(
          "classifier", torch::nn::Linear(1280, options.num_classes));
    }
  });
}

std::vector<Tensor> EfficientNetV2Impl::forward_features(const Tensor& input) {
  return detail::Backend("efficientnetv2.forward_features", [&] {
    detail::FloatingInput(input, conv_stem_->weight, 4);
    if (input.size(1) != options_.input_channels)
      detail::Invalid("efficientnetv2.input",
                      "the configured image channel count");
    return blocks_->forward_features(BnAct(conv_stem_(SamePad(input, 3, 2)),
                                           bn1_,
                                           options_.activation_recomputation));
  });
}

Tensor EfficientNetV2Impl::forward(const Tensor& input) {
  return detail::Backend("efficientnetv2.forward", [&] {
    const auto stages = forward_features(input);
    const auto features = BnAct(conv_head_(stages.back()), bn2_,
                                options_.activation_recomputation);
    auto pooled = features.mean({2, 3});
    return classifier_ ? classifier_(pooled) : pooled;
  });
}

void EfficientNetV2Impl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("efficientnetv2.reset_classifier", [&] {
    detail::ResetLinear(*this, classifier_, "classifier", 1280, num_classes,
                        conv_stem_->weight);
    options_.num_classes = num_classes;
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
  output.shape = {
      {std::nullopt, options.num_classes > 0 ? options.num_classes : 1280}};
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
