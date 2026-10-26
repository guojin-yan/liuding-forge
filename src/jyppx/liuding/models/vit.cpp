// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "detail.hpp"

#include <jyppx/liuding/models/vit.hpp>

#include <torch/nn/functional/upsampling.h>
#include <torch/nn/init.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace jyppx::liuding::models {
namespace {

std::int64_t Dimension(ViTVariant variant) {
  switch (variant) {
    case ViTVariant::tiny:
      return 192;
    case ViTVariant::small:
      return 384;
    case ViTVariant::base:
      return 768;
  }
  detail::Invalid("vit.variant", "tiny, small or base");
}

void ValidateOptions(const ViTOptions& options) {
  if (options.num_classes < 0 || options.num_classes > 1'000'000 ||
      options.input_channels <= 0 || options.input_channels > 1024)
    detail::Invalid("vit.options",
                    "nonnegative classes and input channels in 1..1024");
}

class PatchEmbed final : public torch::nn::Module {
 public:
  PatchEmbed(std::int64_t channels, std::int64_t dimension) {
    projection_ = register_module(
        "proj",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, dimension, 16)
                              .stride(16)
                              .bias(true)));
  }

  Tensor forward(const Tensor& input) { return projection_->forward(input); }
  const Tensor& weight() const noexcept { return projection_->weight; }

 private:
  torch::nn::Conv2d projection_{nullptr};
};

PatchEmbed& Patch(const std::shared_ptr<torch::nn::Module>& module) {
  const auto patch = std::dynamic_pointer_cast<PatchEmbed>(module);
  if (!patch) detail::Invalid("vit.patch_embed", "the registered patch module");
  return *patch;
}

class Attention final : public torch::nn::Module {
 public:
  Attention(std::int64_t dimension, std::int64_t heads)
      : dimension_(dimension), heads_(heads) {
    qkv_ = register_module("qkv", torch::nn::Linear(dimension, dimension * 3));
    projection_ =
        register_module("proj", torch::nn::Linear(dimension, dimension));
  }

  Tensor forward(const Tensor& input) {
    const auto batch = input.size(0);
    const auto tokens = input.size(1);
    const auto head_dimension = dimension_ / heads_;
    auto qkv = qkv_->forward(input)
                   .reshape({batch, tokens, 3, heads_, head_dimension})
                   .permute({2, 0, 3, 1, 4});
    auto query = qkv.select(0, 0);
    auto key = qkv.select(0, 1);
    auto value = qkv.select(0, 2);
    auto attention = torch::softmax(
        torch::matmul(
            query * (1.0 / std::sqrt(static_cast<double>(head_dimension))),
            key.transpose(-2, -1)),
        -1);
    auto attended = torch::matmul(attention, value)
                        .transpose(1, 2)
                        .reshape({batch, tokens, dimension_});
    return projection_->forward(attended);
  }

 private:
  std::int64_t dimension_;
  std::int64_t heads_;
  torch::nn::Linear qkv_{nullptr}, projection_{nullptr};
};

class Mlp final : public torch::nn::Module {
 public:
  explicit Mlp(std::int64_t dimension) {
    fc1_ = register_module("fc1", torch::nn::Linear(dimension, dimension * 4));
    fc2_ = register_module("fc2", torch::nn::Linear(dimension * 4, dimension));
  }

  Tensor forward(const Tensor& input) {
    return fc2_->forward(torch::gelu(fc1_->forward(input)));
  }

 private:
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};

class Block final : public torch::nn::Module {
 public:
  Block(std::int64_t dimension, std::int64_t heads) {
    norm1_ = register_module(
        "norm1", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({dimension}).eps(1e-6)));
    attention_ =
        register_module("attn", std::make_shared<Attention>(dimension, heads));
    norm2_ = register_module(
        "norm2", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({dimension}).eps(1e-6)));
    mlp_ = register_module("mlp", std::make_shared<Mlp>(dimension));
  }

  Tensor forward(const Tensor& input) {
    auto output = input + attention_->forward(norm1_->forward(input));
    return output + mlp_->forward(norm2_->forward(output));
  }

 private:
  torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
  std::shared_ptr<Attention> attention_;
  std::shared_ptr<Mlp> mlp_;
};

void TruncatedNormal_(const Tensor& tensor, double standard_deviation) {
  auto values = torch::empty_like(tensor).view({-1});
  const auto bound = 2.0 * standard_deviation;
  torch::nn::init::normal_(values, 0.0, standard_deviation);
  auto rejected = values.abs().gt(bound);
  while (rejected.any().item<bool>()) {
    const auto count = rejected.sum().item<std::int64_t>();
    auto replacement = torch::empty({count}, values.options());
    torch::nn::init::normal_(replacement, 0.0, standard_deviation);
    auto replacement_rejected = replacement.abs().gt(bound);
    while (replacement_rejected.any().item<bool>()) {
      const auto refill_count = replacement_rejected.sum().item<std::int64_t>();
      auto refill = torch::empty({refill_count}, values.options());
      torch::nn::init::normal_(refill, 0.0, standard_deviation);
      replacement.masked_scatter_(replacement_rejected, refill);
      replacement_rejected = replacement.abs().gt(bound);
    }
    values.masked_scatter_(rejected, replacement);
    rejected = values.abs().gt(bound);
  }
  tensor.detach().copy_(values.view(tensor.sizes()));
}

void InitializeChildren(torch::nn::Module& module) {
  for (const auto& child : module.children()) {
    if (const auto linear =
            std::dynamic_pointer_cast<torch::nn::LinearImpl>(child)) {
      TruncatedNormal_(linear->weight, 0.02);
      if (linear->bias.defined()) torch::nn::init::zeros_(linear->bias);
    } else if (const auto norm =
                   std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(child)) {
      torch::nn::init::ones_(norm->weight);
      torch::nn::init::zeros_(norm->bias);
    }
    InitializeChildren(*child);
  }
}

Tensor PositionForGrid(const Tensor& positions, std::int64_t height,
                       std::int64_t width, std::int64_t dimension) {
  constexpr std::int64_t reference_grid = 14;
  if (height == reference_grid && width == reference_grid) return positions;
  auto cls = positions.slice(1, 0, 1);
  auto patches = positions.slice(1, 1)
                     .reshape({1, reference_grid, reference_grid, dimension})
                     .permute({0, 3, 1, 2});
  patches = torch::nn::functional::interpolate(
                patches.to(torch::kFloat32),
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<std::int64_t>{height, width})
                    .mode(torch::kBicubic)
                    .align_corners(false)
                    .antialias(true))
                .to(positions.scalar_type());
  patches =
      patches.permute({0, 2, 3, 1}).reshape({1, height * width, dimension});
  return torch::cat({cls, patches}, 1);
}

}  // namespace

ViTImpl::ViTImpl(ViTVariant variant, ViTOptions options)
    : options_(options), variant_(variant) {
  detail::Backend("vit.construct", [&] {
    ValidateOptions(options_);
    dimension_ = Dimension(variant_);
    patch_embed_ = register_module(
        "patch_embed",
        std::make_shared<PatchEmbed>(options_.input_channels, dimension_));
    cls_token_ =
        register_parameter("cls_token", torch::empty({1, 1, dimension_}), true);
    pos_embed_ = register_parameter(
        "pos_embed", torch::empty({1, 1 + 14 * 14, dimension_}), true);
    blocks_ = register_module("blocks", torch::nn::Sequential());
    for (std::int64_t i = 0; i < 12; ++i)
      blocks_->push_back(std::make_shared<Block>(dimension_, dimension_ / 64));
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({dimension_}).eps(1e-6)));
    if (options_.num_classes > 0)
      head_ = register_module(
          "head", torch::nn::Linear(dimension_, options_.num_classes));
    torch::NoGradGuard guard;
    InitializeChildren(*this);
    torch::nn::init::normal_(cls_token_, 0.0, 1e-6);
    TruncatedNormal_(pos_embed_, 0.02);
  });
}

Tensor ViTImpl::forward_features(const Tensor& input) {
  return detail::Backend("vit.forward_features", [&] {
    detail::FloatingInput(input, Patch(patch_embed_).weight(), 4);
    if (input.size(1) != options_.input_channels ||
        (options_.dynamic_image_size
             ? (input.size(2) < 16 || input.size(3) < 16 ||
                input.size(2) % 16 != 0 || input.size(3) % 16 != 0)
             : (input.size(2) != 224 || input.size(3) != 224)))
      detail::Invalid("vit.input",
                      options_.dynamic_image_size
                          ? "NCHW images with positive H/W divisible by 16"
                          : "NCHW images of exactly 224x224");
    auto patches = Patch(patch_embed_).forward(input);
    const auto height = patches.size(2);
    const auto width = patches.size(3);
    auto tokens = patches.flatten(2).transpose(1, 2);
    tokens =
        torch::cat({cls_token_.expand({input.size(0), -1, -1}), tokens}, 1);
    tokens = tokens + PositionForGrid(pos_embed_, height, width, dimension_);
    return norm_->forward(blocks_->forward(tokens));
  });
}

Tensor ViTImpl::forward_patch_grid(const Tensor& input) {
  return detail::Backend("vit.forward_patch_grid", [&] {
    auto tokens = forward_features(input).slice(1, 1);
    return tokens.transpose(1, 2).reshape(
        {input.size(0), dimension_, input.size(2) / 16, input.size(3) / 16});
  });
}

Tensor ViTImpl::forward(const Tensor& input) {
  return detail::Backend("vit.forward", [&] {
    auto cls = forward_features(input).select(1, 0);
    return head_ ? head_->forward(cls) : cls;
  });
}

const ViTOptions& ViTImpl::options() const noexcept { return options_; }
ViTVariant ViTImpl::variant() const noexcept { return variant_; }

void ViTImpl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("vit.reset_classifier", [&] {
    if (num_classes < 0 || num_classes > 1'000'000)
      detail::Invalid("vit.head", "nonnegative classes no greater than 1e6");
    if (num_classes == 0) {
      if (head_) {
        unregister_module("head");
        head_ = nullptr;
      }
      options_.num_classes = 0;
      return;
    }
    auto replacement = torch::nn::Linear(dimension_, num_classes);
    {
      torch::NoGradGuard guard;
      TruncatedNormal_(replacement->weight, 0.02);
      torch::nn::init::zeros_(replacement->bias);
    }
    replacement->to(Patch(patch_embed_).weight().device(),
                    Patch(patch_embed_).weight().scalar_type());
    if (head_)
      head_ = replace_module("head", replacement);
    else
      head_ = register_module("head", replacement);
    options_.num_classes = num_classes;
  });
}

ViT vit_tiny_patch16_224(ViTOptions options) {
  return ViT(ViTVariant::tiny, options);
}
ViT vit_small_patch16_224(ViTOptions options) {
  return ViT(ViTVariant::small, options);
}
ViT vit_base_patch16_224(ViTOptions options) {
  return ViT(ViTVariant::base, options);
}

nn::ModelAdapter adapt(ViT model) {
  if (!model) detail::Invalid("vit.adapt", "a nonempty model");
  TensorSchema input;
  input.rank = 4;
  input.shape = {{std::nullopt, model->options().input_channels, std::nullopt,
                  std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, model->options().num_classes > 0
                                     ? model->options().num_classes
                                     : Dimension(model->variant())}};
  return nn::adapt_holder(
      model,
      [](ViTImpl& module, const nn::Invocation& invocation) {
        return module.forward(invocation.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor(input, "images", "vit_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "output", "vit_logits_or_cls"));
}

}  // namespace jyppx::liuding::models
