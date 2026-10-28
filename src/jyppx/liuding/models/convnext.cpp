// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "detail.hpp"

#include <jyppx/liuding/models/convnext.hpp>

#include <torch/nn/init.h>
#include <torch/torch.h>

#include <array>
#include <cmath>

namespace jyppx::liuding::models {
namespace {

class LayerNorm2d final : public torch::nn::Module {
 public:
  explicit LayerNorm2d(std::int64_t channels)
      : weight_(register_parameter("weight", torch::ones({channels}))),
        bias_(register_parameter("bias", torch::zeros({channels}))) {}

  Tensor forward(const Tensor& input) {
    auto channels_last = input.permute({0, 2, 3, 1});
    const auto mean = channels_last.mean(-1, true);
    const auto centered = channels_last - mean;
    const auto variance = centered.square().mean(-1, true);
    return ((centered / torch::sqrt(variance + 1e-6)) * weight_ + bias_)
        .permute({0, 3, 1, 2});
  }

 private:
  Tensor weight_, bias_;
};

class Mlp final : public torch::nn::Module {
 public:
  explicit Mlp(std::int64_t channels) {
    fc1_ = register_module("fc1", torch::nn::Linear(channels, channels * 4));
    fc2_ = register_module("fc2", torch::nn::Linear(channels * 4, channels));
  }

  Tensor forward(const Tensor& input) {
    return fc2_->forward(torch::gelu(fc1_->forward(input)));
  }

 private:
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};

class Block final : public torch::nn::Module {
 public:
  Block(std::int64_t channels, double layer_scale_init) {
    conv_dw_ = register_module(
        "conv_dw",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 7)
                              .stride(1)
                              .padding(3)
                              .groups(channels)
                              .bias(true)));
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({channels}).eps(1e-6)));
    mlp_ = register_module("mlp", std::make_shared<Mlp>(channels));
    gamma_ =
        register_parameter("gamma", torch::full({channels}, layer_scale_init));
  }

  Tensor forward(const Tensor& input) {
    auto output = conv_dw_->forward(input).permute({0, 2, 3, 1});
    output = mlp_->forward(norm_->forward(output));
    output = output.permute({0, 3, 1, 2}) * gamma_.view({1, -1, 1, 1});
    return input + output;
  }

 private:
  torch::nn::Conv2d conv_dw_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::shared_ptr<Mlp> mlp_;
  Tensor gamma_;
};

class Stage final : public torch::nn::Module {
 public:
  Stage(std::int64_t input_channels, std::int64_t channels, std::int64_t depth,
        double layer_scale_init) {
    downsample_ = register_module("downsample", torch::nn::Sequential());
    if (input_channels != channels) {
      downsample_->push_back(std::make_shared<LayerNorm2d>(input_channels));
      downsample_->push_back(torch::nn::Conv2d(
          torch::nn::Conv2dOptions(input_channels, channels, 2)
              .stride(2)
              .bias(true)));
    }
    blocks_ = register_module("blocks", torch::nn::Sequential());
    for (std::int64_t i = 0; i < depth; ++i)
      blocks_->push_back(std::make_shared<Block>(channels, layer_scale_init));
  }

  Tensor forward(const Tensor& input) {
    auto output = downsample_->size() ? downsample_->forward(input) : input;
    return blocks_->forward(output);
  }

 private:
  torch::nn::Sequential downsample_{nullptr}, blocks_{nullptr};
};

std::array<std::int64_t, 4> Depths(ConvNeXtVariant variant) {
  switch (variant) {
    case ConvNeXtVariant::tiny:
      return {3, 3, 9, 3};
    case ConvNeXtVariant::small_capacity:
      return {3, 3, 27, 3};
  }
  detail::Invalid("convnext.variant", "tiny or small");
}

void Validate(const ConvNeXtOptions& options) {
  if (options.num_classes < 0 || options.num_classes > 1'000'000 ||
      options.input_channels <= 0 || options.input_channels > 1024 ||
      !std::isfinite(options.layer_scale_init) ||
      options.layer_scale_init <= 0 || options.layer_scale_init > 1)
    detail::Invalid("convnext.options",
                    "nonnegative classes, positive channels and LayerScale");
}

void Initialize(torch::nn::Module& module, double layer_scale_init) {
  torch::NoGradGuard guard;
  for (const auto& parameter : module.named_parameters(true)) {
    const auto name = parameter.key();
    if (name.ends_with("gamma"))
      parameter.value().fill_(layer_scale_init);
    else if (name.ends_with("weight") && parameter.value().dim() >= 2)
      torch::nn::init::normal_(parameter.value(), 0.0, 0.02);
    else if (name.ends_with("bias"))
      torch::nn::init::zeros_(parameter.value());
    else if (name.ends_with("weight"))
      torch::nn::init::ones_(parameter.value());
  }
}

}  // namespace

class ConvNeXtHead final : public torch::nn::Module {
 public:
  explicit ConvNeXtHead(std::int64_t num_classes) {
    norm_ = register_module("norm", std::make_shared<LayerNorm2d>(768));
    if (num_classes > 0)
      fc_ = register_module("fc", torch::nn::Linear(768, num_classes));
  }

  Tensor forward(const Tensor& input) {
    auto output = norm_->forward(input.mean({2, 3}, true)).flatten(1);
    return fc_ ? fc_->forward(output) : output;
  }

  void reset(std::int64_t num_classes, const Tensor& reference) {
    if (num_classes == 0) {
      if (fc_) {
        unregister_module("fc");
        fc_ = nullptr;
      }
      return;
    }
    auto replacement = torch::nn::Linear(768, num_classes);
    {
      torch::NoGradGuard guard;
      torch::nn::init::normal_(replacement->weight, 0.0, 0.02);
      torch::nn::init::zeros_(replacement->bias);
    }
    replacement->to(reference.device(), reference.scalar_type());
    replacement->train(is_training());
    if (fc_)
      fc_ = replace_module("fc", replacement);
    else
      fc_ = register_module("fc", replacement);
  }

 private:
  std::shared_ptr<LayerNorm2d> norm_;
  torch::nn::Linear fc_{nullptr};
};

ConvNeXtImpl::ConvNeXtImpl(ConvNeXtVariant variant, ConvNeXtOptions options)
    : options_(options), variant_(variant) {
  detail::Backend("convnext.construct", [&] {
    Validate(options_);
    const auto depths = Depths(variant_);
    constexpr std::array<std::int64_t, 4> channels{96, 192, 384, 768};
    stem_ = register_module("stem", torch::nn::Sequential());
    stem_->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(options_.input_channels, channels.front(), 4)
            .stride(4)
            .bias(true)));
    stem_->push_back(std::make_shared<LayerNorm2d>(channels.front()));
    stages_ = register_module("stages", torch::nn::ModuleList());
    for (std::size_t index = 0; index < channels.size(); ++index)
      stages_->push_back(std::make_shared<Stage>(
          index ? channels[index - 1] : channels.front(), channels[index],
          depths[index], options_.layer_scale_init));
    head_ = register_module(
        "head", std::make_shared<ConvNeXtHead>(options_.num_classes));
    Initialize(*this, options_.layer_scale_init);
  });
}

std::vector<Tensor> ConvNeXtImpl::forward_features(const Tensor& input) {
  return detail::Backend("convnext.forward_features", [&] {
    const auto stem_conv = stem_->ptr(0)->as<torch::nn::Conv2d>();
    detail::FloatingInput(input, stem_conv->weight, 4);
    if (input.size(1) != options_.input_channels || input.size(2) < 32 ||
        input.size(3) < 32)
      detail::Invalid("convnext.input",
                      "NCHW floating images with dimensions >= 32");
    auto output = stem_->forward(input);
    std::vector<Tensor> stages;
    stages.reserve(4);
    for (std::size_t index = 0; index < 4; ++index) {
      output = stages_[index]->as<Stage>()->forward(output);
      stages.push_back(output);
    }
    return stages;
  });
}

Tensor ConvNeXtImpl::forward(const Tensor& input) {
  return detail::Backend("convnext.forward", [&] {
    return head_->forward(forward_features(input).back());
  });
}

void ConvNeXtImpl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("convnext.reset_classifier", [&] {
    if (num_classes < 0 || num_classes > 1'000'000)
      detail::Invalid("convnext.head", "nonnegative classes <= 1e6");
    head_->reset(num_classes, stem_->ptr(0)->as<torch::nn::Conv2d>()->weight);
    options_.num_classes = num_classes;
  });
}

const ConvNeXtOptions& ConvNeXtImpl::options() const noexcept {
  return options_;
}

ConvNeXtVariant ConvNeXtImpl::variant() const noexcept { return variant_; }

ConvNeXt convnext_tiny(ConvNeXtOptions options) {
  return ConvNeXt(
      std::make_shared<ConvNeXtImpl>(ConvNeXtVariant::tiny, options));
}

ConvNeXt convnext_small(ConvNeXtOptions options) {
  return ConvNeXt(
      std::make_shared<ConvNeXtImpl>(ConvNeXtVariant::small_capacity, options));
}

nn::ModelAdapter adapt(ConvNeXt model) {
  if (!model) detail::Invalid("convnext.adapt", "a nonempty model");
  TensorSchema input;
  input.rank = 4;
  input.shape = {{std::nullopt, model->options().input_channels, std::nullopt,
                  std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, model->options().num_classes > 0
                                     ? model->options().num_classes
                                     : 768}};
  return nn::adapt_holder(
      model,
      [](ConvNeXtImpl& module, const nn::Invocation& invocation) {
        return module.forward(invocation.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor(input, "images", "convnext_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "output", "convnext_logits_or_embedding"));
}

}  // namespace jyppx::liuding::models
