// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "detail.hpp"

#include <jyppx/liuding/models/dinov3_convnext.hpp>

#include <torch/nn/init.h>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>

namespace jyppx::liuding::models {
namespace {

struct Config {
  std::vector<std::int64_t> depths;
  std::vector<std::int64_t> channels;
};

Config VariantConfig(DinoV3ConvNeXtVariant variant) {
  switch (variant) {
    case DinoV3ConvNeXtVariant::tiny:
      return {{3, 3, 9, 3}, {96, 192, 384, 768}};
    case DinoV3ConvNeXtVariant::small_capacity:
      return {{3, 3, 27, 3}, {96, 192, 384, 768}};
    case DinoV3ConvNeXtVariant::base:
      return {{3, 3, 27, 3}, {128, 256, 512, 1024}};
    case DinoV3ConvNeXtVariant::large:
      return {{3, 3, 27, 3}, {192, 384, 768, 1536}};
  }
  detail::Invalid("dinov3.convnext.variant", "tiny, small, base or large");
}

void Validate(const DinoV3ConvNeXtOptions& options) {
  if (options.num_classes < 0 || options.num_classes > 1'000'000 ||
      options.input_channels <= 0 || options.input_channels > 1024 ||
      !std::isfinite(options.layer_scale_init) ||
      options.layer_scale_init <= 0 || options.layer_scale_init > 1) {
    detail::Invalid("dinov3.convnext.options",
                    "nonnegative classes, channels in 1..1024 and a finite "
                    "positive LayerScale value");
  }
}

void Initialize(torch::nn::Module& module) {
  torch::NoGradGuard guard;
  for (const auto& parameter : module.named_parameters(/*recurse=*/true)) {
    const auto name = parameter.key();
    if (name.ends_with("weight") && parameter.value().dim() >= 2) {
      torch::nn::init::normal_(parameter.value(), 0.0, 0.02);
    } else if (name.ends_with("bias")) {
      torch::nn::init::zeros_(parameter.value());
    } else if (name.ends_with("weight")) {
      torch::nn::init::ones_(parameter.value());
    }
  }
}

class ChannelsFirstLayerNorm final : public torch::nn::Module {
 public:
  explicit ChannelsFirstLayerNorm(std::int64_t channels) {
    weight_ = register_parameter("weight", torch::ones({channels}));
    bias_ = register_parameter("bias", torch::zeros({channels}));
  }

  Tensor forward(const Tensor& input) {
    const auto mean = input.mean(1, true);
    const auto centered = input - mean;
    const auto variance = centered.pow(2).mean(1, true);
    const auto normalized = centered / torch::sqrt(variance + 1e-6);
    return weight_.view({1, -1, 1, 1}) * normalized +
           bias_.view({1, -1, 1, 1});
  }

 private:
  Tensor weight_, bias_;
};

class ConvNeXtBlock final : public torch::nn::Module {
 public:
  ConvNeXtBlock(std::int64_t channels, double layer_scale_init) {
    depthwise_ = register_module(
        "dwconv", torch::nn::Conv2d(
                      torch::nn::Conv2dOptions(channels, channels, 7)
                          .stride(1)
                          .padding(3)
                          .groups(channels)
                          .bias(true)));
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({channels}).eps(1e-6)));
    pointwise1_ = register_module(
        "pwconv1", torch::nn::Linear(channels, channels * 4));
    pointwise2_ = register_module(
        "pwconv2", torch::nn::Linear(channels * 4, channels));
    layer_scale_ = register_parameter(
        "gamma", torch::full({channels}, layer_scale_init), true);
  }

  Tensor forward(const Tensor& input) {
    auto output = depthwise_->forward(input);
    output = norm_->forward(output.permute({0, 2, 3, 1}));
    output = pointwise1_->forward(output);
    output = torch::gelu(output);
    output = pointwise2_->forward(output);
    output = output * layer_scale_.view({1, 1, 1, -1});
    return input + output.permute({0, 3, 1, 2});
  }

 private:
  torch::nn::Conv2d depthwise_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear pointwise1_{nullptr}, pointwise2_{nullptr};
  Tensor layer_scale_;
};

}  // namespace

DinoV3ConvNeXtImpl::DinoV3ConvNeXtImpl(DinoV3ConvNeXtVariant variant,
                                       DinoV3ConvNeXtOptions options)
    : options_(options), variant_(variant) {
  detail::Backend("dinov3.convnext.construct", [&] {
    Validate(options_);
    const auto config = VariantConfig(variant_);
    channels_ = config.channels;
    downsample_layers_ =
        register_module("downsample_layers", torch::nn::ModuleList());
    auto stem = torch::nn::Sequential();
    stem_ = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(options_.input_channels,
                                 channels_.front(), 4)
            .stride(4)
            .padding(0)
            .bias(true));
    stem->push_back(stem_);
    stem->push_back(
        std::make_shared<ChannelsFirstLayerNorm>(channels_.front()));
    downsample_layers_->push_back(stem);
    stages_ = register_module("stages", torch::nn::ModuleList());
    for (std::size_t index = 0; index < channels_.size(); ++index) {
      if (index > 0) {
        auto downsample = torch::nn::Sequential();
        downsample->push_back(
            std::make_shared<ChannelsFirstLayerNorm>(channels_[index - 1]));
        downsample->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels_[index - 1], channels_[index],
                                     2)
                .stride(2)
                .padding(0)
                .bias(true)));
        downsample_layers_->push_back(downsample);
      }
      auto stage = torch::nn::Sequential();
      for (std::int64_t block = 0; block < config.depths[index]; ++block)
        stage->push_back(std::make_shared<ConvNeXtBlock>(
            channels_[index], options_.layer_scale_init));
      stages_->push_back(stage);
    }
    final_norm_ = register_module(
        "norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({channels_.back()}).eps(1e-6)));
    if (options_.num_classes > 0)
      head_ = register_module(
          "head", torch::nn::Linear(channels_.back(), options_.num_classes));
    Initialize(*this);
    for (std::size_t index = 0; index < stages_->size(); ++index) {
      const auto stage = stages_[index]->as<torch::nn::Sequential>();
      for (const auto& child : stage->children()) {
        if (const auto block = std::dynamic_pointer_cast<ConvNeXtBlock>(child)) {
          torch::NoGradGuard guard;
          block->named_parameters()["gamma"].fill_(options_.layer_scale_init);
        }
      }
    }
  });
}

std::vector<Tensor> DinoV3ConvNeXtImpl::forward_features(const Tensor& input) {
  return detail::Backend("dinov3.convnext.forward_features", [&] {
    detail::FloatingInput(input, stem_->weight, 4);
    if (input.size(1) != options_.input_channels || input.size(2) < 32 ||
        input.size(3) < 32)
      detail::Invalid("dinov3.convnext.input",
                      "NCHW floating images with dimensions >= 32");
    const auto height = (input.size(2) / 32) * 32;
    const auto width = (input.size(3) / 32) * 32;
    auto output = input.narrow(2, 0, height).narrow(3, 0, width);
    std::vector<Tensor> features;
    features.reserve(stages_->size());
    for (std::size_t index = 0; index < stages_->size(); ++index) {
      output = downsample_layers_[index]
                   ->as<torch::nn::Sequential>()
                   ->forward(output);
      output = stages_[index]->as<torch::nn::Sequential>()->forward(output);
      features.push_back(output);
    }
    return features;
  });
}

Tensor DinoV3ConvNeXtImpl::forward(const Tensor& input) {
  return detail::Backend("dinov3.convnext.forward", [&] {
    auto features = forward_features(input);
    auto pooled = features.back().mean({2, 3});
    pooled = final_norm_->forward(pooled);
    return head_ ? head_->forward(pooled) : pooled;
  });
}

const DinoV3ConvNeXtOptions& DinoV3ConvNeXtImpl::options() const noexcept {
  return options_;
}

DinoV3ConvNeXtVariant DinoV3ConvNeXtImpl::variant() const noexcept {
  return variant_;
}

void DinoV3ConvNeXtImpl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("dinov3.convnext.reset_classifier", [&] {
    if (num_classes < 0 || num_classes > 1'000'000)
      detail::Invalid("dinov3.convnext.head",
                      "nonnegative classes no greater than 1e6");
    if (num_classes == 0) {
      if (head_) {
        unregister_module("head");
        head_ = nullptr;
      }
      options_.num_classes = 0;
      return;
    }
    auto replacement = torch::nn::Linear(channels_.back(), num_classes);
    {
      torch::NoGradGuard guard;
      torch::nn::init::normal_(replacement->weight, 0.0, 0.02);
      torch::nn::init::zeros_(replacement->bias);
    }
    replacement->to(stem_->weight.device(), stem_->weight.scalar_type());
    if (head_)
      head_ = replace_module("head", replacement);
    else
      head_ = register_module("head", replacement);
    options_.num_classes = num_classes;
  });
}

DinoV3ConvNeXt dinov3_convnext_tiny(DinoV3ConvNeXtOptions options) {
  return DinoV3ConvNeXt(std::make_shared<DinoV3ConvNeXtImpl>(
      DinoV3ConvNeXtVariant::tiny, options));
}

DinoV3ConvNeXt dinov3_convnext_small(DinoV3ConvNeXtOptions options) {
  return DinoV3ConvNeXt(std::make_shared<DinoV3ConvNeXtImpl>(
      DinoV3ConvNeXtVariant::small_capacity, options));
}

DinoV3ConvNeXt dinov3_convnext_base(DinoV3ConvNeXtOptions options) {
  return DinoV3ConvNeXt(std::make_shared<DinoV3ConvNeXtImpl>(
      DinoV3ConvNeXtVariant::base, options));
}

DinoV3ConvNeXt dinov3_convnext_large(DinoV3ConvNeXtOptions options) {
  return DinoV3ConvNeXt(std::make_shared<DinoV3ConvNeXtImpl>(
      DinoV3ConvNeXtVariant::large, options));
}

nn::ModelAdapter adapt(DinoV3ConvNeXt model) {
  if (!model) detail::Invalid("dinov3.convnext.adapt", "a nonempty model");
  TensorSchema input;
  input.rank = 4;
  input.shape = {{std::nullopt, model->options().input_channels, std::nullopt,
                  std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, model->options().num_classes > 0
                                     ? model->options().num_classes
                                     : VariantConfig(model->variant()).channels.back()}};
  return nn::adapt_holder(
      model,
      [](DinoV3ConvNeXtImpl& module, const nn::Invocation& invocation) {
        return module.forward(invocation.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor(
           input, "images", "dinov3_convnext_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "output", "dinov3_convnext_logits_or_cls"));
}

}  // namespace jyppx::liuding::models
