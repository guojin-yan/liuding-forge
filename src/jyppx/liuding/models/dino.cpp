// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "detail.hpp"

#include <jyppx/liuding/models/dino.hpp>

#include <torch/nn/functional/upsampling.h>
#include <torch/nn/init.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace jyppx::liuding::models {
namespace {

struct VariantConfig {
  std::int64_t dimension;
  std::int64_t heads;
  std::int64_t blocks;
  std::int64_t hidden;
  bool swiglu;
};

VariantConfig Config(DinoV2Variant variant) {
  switch (variant) {
    case DinoV2Variant::small:
      return {.dimension = 384,
              .heads = 6,
              .blocks = 12,
              .hidden = 1536,
              .swiglu = false};
    case DinoV2Variant::base:
      return {.dimension = 768,
              .heads = 12,
              .blocks = 12,
              .hidden = 3072,
              .swiglu = false};
    case DinoV2Variant::large:
      return {.dimension = 1024,
              .heads = 16,
              .blocks = 24,
              .hidden = 4096,
              .swiglu = false};
    case DinoV2Variant::giant:
      return {.dimension = 1536,
              .heads = 24,
              .blocks = 40,
              .hidden = 4096,
              .swiglu = true};
  }
  detail::Invalid("dinov2.variant", "small, base, large or giant");
}

void ValidateOptions(const DinoV2Options& options) {
  if (options.num_classes < 0 || options.num_classes > 1'000'000 ||
      options.input_channels <= 0 || options.input_channels > 1024 ||
      !std::isfinite(options.layer_scale_init) ||
      options.layer_scale_init <= 0 || options.layer_scale_init > 1) {
    detail::Invalid("dinov2.options",
                    "nonnegative classes, channels in 1..1024 and a finite "
                    "positive LayerScale value");
  }
}

class DinoV2Attention final : public torch::nn::Module {
 public:
  DinoV2Attention(std::int64_t dimension, std::int64_t heads)
      : dimension_(dimension), heads_(heads) {
    qkv_ = register_module("qkv", torch::nn::Linear(dimension, dimension * 3));
    projection_ =
        register_module("proj", torch::nn::Linear(dimension, dimension));
  }

  Tensor forward(const Tensor& input) {
    const auto batch = input.size(0);
    const auto tokens = input.size(1);
    const auto head_dimension = dimension_ / heads_;
    auto qkv = qkv_->forward(input).chunk(3, -1);
    auto query =
        qkv[0].view({batch, tokens, heads_, head_dimension}).transpose(1, 2);
    auto key =
        qkv[1].view({batch, tokens, heads_, head_dimension}).transpose(1, 2);
    auto value =
        qkv[2].view({batch, tokens, heads_, head_dimension}).transpose(1, 2);
    auto attention =
        torch::softmax(torch::matmul(query, key.transpose(-2, -1)) /
                           std::sqrt(static_cast<double>(head_dimension)),
                       -1);
    auto attended = torch::matmul(attention, value)
                        .transpose(1, 2)
                        .contiguous()
                        .view({batch, tokens, dimension_});
    return projection_->forward(attended);
  }

 private:
  std::int64_t dimension_;
  std::int64_t heads_;
  torch::nn::Linear qkv_{nullptr}, projection_{nullptr};
};

class DinoV2Mlp final : public torch::nn::Module {
 public:
  DinoV2Mlp(std::int64_t dimension, std::int64_t hidden, bool swiglu)
      : hidden_(hidden), swiglu_(swiglu) {
    fc1_ = register_module(
        "fc1", torch::nn::Linear(dimension, swiglu ? hidden * 2 : hidden));
    fc2_ = register_module("fc2", torch::nn::Linear(hidden, dimension));
  }

  Tensor forward(const Tensor& input) {
    auto feedforward = fc1_->forward(input);
    if (swiglu_) {
      auto gates = feedforward.chunk(2, -1);
      feedforward = torch::silu(gates[0]) * gates[1];
    } else {
      feedforward = torch::gelu(feedforward);
    }
    return fc2_->forward(feedforward);
  }

  void initialize_swiglu() {
    if (!swiglu_) return;
    auto weight = fc1_->weight.detach();
    torch::nn::init::normal_(weight.slice(0, hidden_), 0.0, 1e-6);
    auto bias = fc1_->bias.detach();
    bias.slice(0, hidden_).fill_(1.0);
  }

 private:
  std::int64_t hidden_;
  bool swiglu_;
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};

class DinoV2LayerScale final : public torch::nn::Module {
 public:
  DinoV2LayerScale(std::int64_t dimension, double initial_value)
      : dimension_(dimension) {
    gamma_ = register_parameter("gamma",
                                torch::full({dimension}, initial_value), true);
  }

  Tensor forward(const Tensor& input) {
    return input * gamma_.view({1, 1, dimension_});
  }

 private:
  std::int64_t dimension_;
  Tensor gamma_;
};

class DinoV2Block final : public torch::nn::Module {
 public:
  DinoV2Block(std::int64_t dimension, std::int64_t heads, std::int64_t hidden,
              bool swiglu, double layer_scale) {
    if (dimension <= 0 || heads <= 0 || dimension % heads != 0 || hidden <= 0 ||
        layer_scale <= 0) {
      detail::Invalid("dinov2.block", "positive divisible attention widths");
    }
    norm1_ = register_module(
        "norm1", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({dimension}).eps(1e-6)));
    attention_ = register_module(
        "attn", std::make_shared<DinoV2Attention>(dimension, heads));
    layer_scale1_ = register_module(
        "ls1", std::make_shared<DinoV2LayerScale>(dimension, layer_scale));
    norm2_ = register_module(
        "norm2", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({dimension}).eps(1e-6)));
    mlp_ = register_module(
        "mlp", std::make_shared<DinoV2Mlp>(dimension, hidden, swiglu));
    layer_scale2_ = register_module(
        "ls2", std::make_shared<DinoV2LayerScale>(dimension, layer_scale));
  }

  Tensor forward(const Tensor& input) {
    auto output = input + layer_scale1_->forward(
                              attention_->forward(norm1_->forward(input)));
    return output +
           layer_scale2_->forward(mlp_->forward(norm2_->forward(output)));
  }

 private:
  torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
  std::shared_ptr<DinoV2Attention> attention_;
  std::shared_ptr<DinoV2Mlp> mlp_;
  std::shared_ptr<DinoV2LayerScale> layer_scale1_, layer_scale2_;
};

class DinoV2PatchEmbed final : public torch::nn::Module {
 public:
  DinoV2PatchEmbed(std::int64_t input_channels, std::int64_t dimension) {
    projection_ = register_module(
        "proj", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(input_channels, dimension, 14)
                        .stride(14)
                        .padding(0)
                        .bias(true)));
  }

  Tensor forward(const Tensor& input) { return projection_->forward(input); }
  const Tensor& weight() const noexcept { return projection_->weight; }

 private:
  torch::nn::Conv2d projection_{nullptr};
};

DinoV2PatchEmbed& PatchEmbed(const std::shared_ptr<torch::nn::Module>& module) {
  const auto patch = std::dynamic_pointer_cast<DinoV2PatchEmbed>(module);
  if (!patch)
    detail::Invalid("dinov2.patch_embed", "the registered patch module");
  return *patch;
}

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
    if (const auto mlp = std::dynamic_pointer_cast<DinoV2Mlp>(child))
      mlp->initialize_swiglu();
  }
}

void Initialize(torch::nn::Module& module) {
  torch::NoGradGuard guard;
  InitializeChildren(module);
}

Tensor PositionForGrid(const Tensor& position_embedding, std::int64_t height,
                       std::int64_t width, std::int64_t dimension) {
  constexpr std::int64_t reference_grid = 37;
  auto cls = position_embedding.slice(1, 0, 1);
  auto patches = position_embedding.slice(1, 1).reshape(
      {1, reference_grid, reference_grid, dimension});
  patches = patches.permute({0, 3, 1, 2});
  if (height != reference_grid || width != reference_grid) {
    patches = torch::nn::functional::interpolate(
                  patches.to(torch::kFloat32),
                  torch::nn::functional::InterpolateFuncOptions()
                      .size(std::vector<std::int64_t>{height, width})
                      .mode(torch::kBicubic)
                      .align_corners(false)
                      .antialias(true))
                  .to(position_embedding.scalar_type());
  }
  patches =
      patches.permute({0, 2, 3, 1}).reshape({1, height * width, dimension});
  return torch::cat({cls, patches}, 1);
}

}  // namespace

DinoV2Impl::DinoV2Impl(DinoV2Variant variant, DinoV2Options options)
    : options_(options), variant_(variant) {
  detail::Backend("dinov2.construct", [&] {
    ValidateOptions(options_);
    const auto config = Config(variant_);
    embedding_dimension_ = config.dimension;
    attention_heads_ = config.heads;
    block_count_ = config.blocks;
    hidden_dimension_ = config.hidden;
    swiglu_ = config.swiglu;
    patch_embed_ = register_module(
        "patch_embed", std::make_shared<DinoV2PatchEmbed>(
                           options_.input_channels, embedding_dimension_));
    cls_token_ = register_parameter(
        "cls_token", torch::zeros({1, 1, embedding_dimension_}), true);
    position_embedding_ = register_parameter(
        "pos_embed", torch::zeros({1, 1 + 37 * 37, embedding_dimension_}),
        true);
    blocks_ = register_module("blocks", torch::nn::Sequential());
    for (std::int64_t i = 0; i < block_count_; ++i) {
      blocks_->push_back(std::make_shared<DinoV2Block>(
          embedding_dimension_, attention_heads_, hidden_dimension_, swiglu_,
          options_.layer_scale_init));
    }
    final_norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({embedding_dimension_}).eps(1e-6)));
    if (options_.num_classes > 0)
      head_ = register_module("head", torch::nn::Linear(embedding_dimension_,
                                                        options_.num_classes));
    Initialize(*this);
    torch::nn::init::normal_(cls_token_, 0.0, 1e-6);
    TruncatedNormal_(position_embedding_, 0.02);
  });
}

Tensor DinoV2Impl::forward_features(const Tensor& input) {
  return detail::Backend("dinov2.forward_features", [&] {
    detail::FloatingInput(input, PatchEmbed(patch_embed_).weight(), 4);
    if (input.size(1) != options_.input_channels || input.size(2) <= 0 ||
        input.size(3) <= 0 || input.size(2) % 14 != 0 ||
        input.size(3) % 14 != 0)
      detail::Invalid("dinov2.input",
                      "NCHW floating images with H/W divisible by 14");
    auto patches = PatchEmbed(patch_embed_).forward(input);
    const auto height = patches.size(2);
    const auto width = patches.size(3);
    auto tokens = patches.flatten(2).transpose(1, 2);
    auto cls = cls_token_.expand({input.size(0), -1, -1});
    tokens = torch::cat({cls, tokens}, 1);
    tokens = tokens + PositionForGrid(position_embedding_, height, width,
                                      embedding_dimension_);
    return final_norm_->forward(blocks_->forward(tokens));
  });
}

Tensor DinoV2Impl::forward(const Tensor& input) {
  return detail::Backend("dinov2.forward", [&] {
    auto tokens = forward_features(input);
    auto cls = tokens.select(1, 0);
    return head_ ? head_->forward(cls) : cls;
  });
}

const DinoV2Options& DinoV2Impl::options() const noexcept { return options_; }

DinoV2Variant DinoV2Impl::variant() const noexcept { return variant_; }

void DinoV2Impl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("dinov2.reset_classifier", [&] {
    if (num_classes < 0 || num_classes > 1'000'000)
      detail::Invalid("dinov2.head", "nonnegative classes no greater than 1e6");
    if (num_classes == 0) {
      if (head_) {
        unregister_module("head");
        head_ = nullptr;
      }
      options_.num_classes = 0;
      return;
    }

    // Construct and initialize the replacement before touching the currently
    // registered head. Allocation/device failures therefore preserve the old
    // classifier and its option value.
    auto replacement = torch::nn::Linear(embedding_dimension_, num_classes);
    {
      torch::NoGradGuard guard;
      TruncatedNormal_(replacement->weight, 0.02);
      torch::nn::init::zeros_(replacement->bias);
    }
    replacement->to(PatchEmbed(patch_embed_).weight().device(),
                    PatchEmbed(patch_embed_).weight().scalar_type());
    if (head_)
      head_ = replace_module("head", replacement);
    else
      head_ = register_module("head", replacement);
    options_.num_classes = num_classes;
  });
}

DinoV2 dinov2_vits14(DinoV2Options options) {
  return DinoV2(DinoV2Variant::small, options);
}
DinoV2 dinov2_vitb14(DinoV2Options options) {
  return DinoV2(DinoV2Variant::base, options);
}
DinoV2 dinov2_vitl14(DinoV2Options options) {
  return DinoV2(DinoV2Variant::large, options);
}
DinoV2 dinov2_vit_giant14(DinoV2Options options) {
  return DinoV2(DinoV2Variant::giant, options);
}

nn::ModelAdapter adapt(DinoV2 model) {
  if (!model) detail::Invalid("dinov2.adapt", "a nonempty model");
  TensorSchema input;
  input.rank = 4;
  input.shape = {{std::nullopt, model->options().input_channels, std::nullopt,
                  std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, model->options().num_classes > 0
                                     ? model->options().num_classes
                                     : Config(model->variant()).dimension}};
  return nn::adapt_holder(
      model,
      [](DinoV2Impl& module, const nn::Invocation& invocation) {
        return module.forward(invocation.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor(input, "images", "dino_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "output", "dino_logits_or_cls"));
}

namespace {

struct DinoV3Config {
  std::int64_t dimension;
  std::int64_t heads;
  std::int64_t blocks;
  std::int64_t hidden;
  bool swiglu;
};

DinoV3Config Config(DinoV3Variant variant) {
  switch (variant) {
    case DinoV3Variant::small:
      return {.dimension = 384,
              .heads = 6,
              .blocks = 12,
              .hidden = 1536,
              .swiglu = false};
    case DinoV3Variant::small_plus:
      return {.dimension = 384,
              .heads = 6,
              .blocks = 12,
              .hidden = 1536,
              .swiglu = true};
    case DinoV3Variant::base:
      return {.dimension = 768,
              .heads = 12,
              .blocks = 12,
              .hidden = 3072,
              .swiglu = false};
    case DinoV3Variant::large:
      return {.dimension = 1024,
              .heads = 16,
              .blocks = 24,
              .hidden = 4096,
              .swiglu = false};
    case DinoV3Variant::large_plus:
      return {.dimension = 1024,
              .heads = 16,
              .blocks = 24,
              .hidden = 4096,
              .swiglu = true};
    case DinoV3Variant::huge_plus:
      return {.dimension = 1280,
              .heads = 20,
              .blocks = 32,
              .hidden = 7680,
              .swiglu = true};
    case DinoV3Variant::seven_b:
      return {.dimension = 4096,
              .heads = 32,
              .blocks = 40,
              .hidden = 12288,
              .swiglu = true};
  }
  detail::Invalid("dinov3.variant",
                  "small, small_plus, base, large, large_plus, huge_plus or "
                  "seven_b");
}

void ValidateOptions(const DinoV3Options& options) {
  if (options.num_classes < 0 || options.num_classes > 1'000'000 ||
      options.input_channels <= 0 || options.input_channels > 1024 ||
      options.extra_tokens != 4 || !std::isfinite(options.layer_scale_init) ||
      options.layer_scale_init <= 0 || options.layer_scale_init > 1) {
    detail::Invalid("dinov3.options",
                    "nonnegative classes, channels in 1..1024, exactly four "
                    "extra tokens and a finite positive LayerScale value");
  }
}

class DinoV3Attention final : public torch::nn::Module {
 public:
  DinoV3Attention(std::int64_t dimension, std::int64_t heads)
      : dimension_(dimension), heads_(heads) {
    qkv_ = register_module("qkv", torch::nn::Linear(dimension, dimension * 3));
    projection_ =
        register_module("proj", torch::nn::Linear(dimension, dimension));
  }

  Tensor forward(const Tensor& input) {
    const auto batch = input.size(0);
    const auto tokens = input.size(1);
    const auto head_dimension = dimension_ / heads_;
    auto qkv = qkv_->forward(input).chunk(3, -1);
    auto query =
        qkv[0].view({batch, tokens, heads_, head_dimension}).transpose(1, 2);
    auto key =
        qkv[1].view({batch, tokens, heads_, head_dimension}).transpose(1, 2);
    auto value =
        qkv[2].view({batch, tokens, heads_, head_dimension}).transpose(1, 2);

    // DINOv3 uses rotary position embeddings. The implementation deliberately
    // keeps the token contract explicit: the same deterministic row-major
    // positions are applied to CLS, extra and patch tokens, while the first
    // four positions remain stable when the patch grid changes.
    const auto half = head_dimension / 2;
    auto float_options = input.options().dtype(torch::kFloat32);
    auto position = torch::arange(tokens, float_options).view({1, 1, tokens, 1});
    auto index = torch::arange(half, float_options).view({1, 1, 1, half});
    auto frequency = torch::exp(-std::log(100.0) * index / half);
    auto angles = position * frequency;
    auto cosine = torch::cos(angles).to(query.scalar_type());
    auto sine = torch::sin(angles).to(query.scalar_type());
    auto rotate = [&](const Tensor& tensor) {
      auto even = tensor.slice(-1, 0, half);
      auto odd = tensor.slice(-1, half, head_dimension);
      return torch::cat({even * cosine - odd * sine,
                         even * sine + odd * cosine}, -1);
    };
    query = rotate(query);
    key = rotate(key);
    auto attention =
        torch::softmax(torch::matmul(query, key.transpose(-2, -1)) /
                           std::sqrt(static_cast<double>(head_dimension)),
                       -1);
    auto attended = torch::matmul(attention, value)
                        .transpose(1, 2)
                        .contiguous()
                        .view({batch, tokens, dimension_});
    return projection_->forward(attended);
  }

 private:
  std::int64_t dimension_;
  std::int64_t heads_;
  torch::nn::Linear qkv_{nullptr}, projection_{nullptr};
};

class DinoV3Block final : public torch::nn::Module {
 public:
  DinoV3Block(std::int64_t dimension, std::int64_t heads,
              std::int64_t hidden, bool swiglu, double layer_scale) {
    if (dimension <= 0 || heads <= 0 || dimension % heads != 0 || hidden <= 0 ||
        layer_scale <= 0) {
      detail::Invalid("dinov3.block", "positive divisible attention widths");
    }
    norm1_ = register_module(
        "norm1", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({dimension}).eps(1e-6)));
    attention_ = register_module(
        "attn", std::make_shared<DinoV3Attention>(dimension, heads));
    layer_scale1_ = register_module(
        "ls1", std::make_shared<DinoV2LayerScale>(dimension, layer_scale));
    norm2_ = register_module(
        "norm2", torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({dimension}).eps(1e-6)));
    mlp_ = register_module(
        "mlp", std::make_shared<DinoV2Mlp>(dimension, hidden, swiglu));
    layer_scale2_ = register_module(
        "ls2", std::make_shared<DinoV2LayerScale>(dimension, layer_scale));
  }

  Tensor forward(const Tensor& input) {
    auto output = input + layer_scale1_->forward(
                              attention_->forward(norm1_->forward(input)));
    return output +
           layer_scale2_->forward(mlp_->forward(norm2_->forward(output)));
  }

 private:
  torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
  std::shared_ptr<DinoV3Attention> attention_;
  std::shared_ptr<DinoV2Mlp> mlp_;
  std::shared_ptr<DinoV2LayerScale> layer_scale1_, layer_scale2_;
};

class DinoV3PatchEmbed final : public torch::nn::Module {
 public:
  DinoV3PatchEmbed(std::int64_t input_channels, std::int64_t dimension) {
    projection_ = register_module(
        "proj", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(input_channels, dimension, 16)
                        .stride(16)
                        .padding(0)
                        .bias(true)));
  }

  Tensor forward(const Tensor& input) { return projection_->forward(input); }
  const Tensor& weight() const noexcept { return projection_->weight; }

 private:
  torch::nn::Conv2d projection_{nullptr};
};

DinoV3PatchEmbed& PatchEmbedV3(
    const std::shared_ptr<torch::nn::Module>& module) {
  const auto patch = std::dynamic_pointer_cast<DinoV3PatchEmbed>(module);
  if (!patch)
    detail::Invalid("dinov3.patch_embed", "the registered patch module");
  return *patch;
}

}  // namespace

DinoV3Impl::DinoV3Impl(DinoV3Variant variant, DinoV3Options options)
    : options_(options), variant_(variant) {
  detail::Backend("dinov3.construct", [&] {
    ValidateOptions(options_);
    const auto config = Config(variant_);
    embedding_dimension_ = config.dimension;
    attention_heads_ = config.heads;
    block_count_ = config.blocks;
    hidden_dimension_ = config.hidden;
    swiglu_ = config.swiglu;
    patch_embed_ = register_module(
        "patch_embed", std::make_shared<DinoV3PatchEmbed>(
                           options_.input_channels, embedding_dimension_));
    cls_token_ = register_parameter(
        "cls_token", torch::zeros({1, 1, embedding_dimension_}), true);
    extra_tokens_ = register_parameter(
        "extra_tokens",
        torch::zeros({1, options_.extra_tokens, embedding_dimension_}), true);
    blocks_ = register_module("blocks", torch::nn::Sequential());
    for (std::int64_t i = 0; i < block_count_; ++i) {
      blocks_->push_back(std::make_shared<DinoV3Block>(
          embedding_dimension_, attention_heads_, hidden_dimension_, swiglu_,
          options_.layer_scale_init));
    }
    final_norm_ = register_module(
        "norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({embedding_dimension_}).eps(1e-6)));
    if (options_.num_classes > 0)
      head_ = register_module("head", torch::nn::Linear(embedding_dimension_,
                                                        options_.num_classes));
    Initialize(*this);
    torch::nn::init::normal_(cls_token_, 0.0, 1e-6);
    torch::nn::init::normal_(extra_tokens_, 0.0, 1e-6);
  });
}

Tensor DinoV3Impl::forward_features(const Tensor& input) {
  return detail::Backend("dinov3.forward_features", [&] {
    detail::FloatingInput(input, PatchEmbedV3(patch_embed_).weight(), 4);
    if (input.size(1) != options_.input_channels || input.size(2) < 16 ||
        input.size(3) < 16)
      detail::Invalid("dinov3.input",
                      "NCHW floating images with positive dimensions >= 16");
    const auto height = (input.size(2) / 16) * 16;
    const auto width = (input.size(3) / 16) * 16;
    auto cropped = input.narrow(2, 0, height).narrow(3, 0, width);
    auto patches = PatchEmbedV3(patch_embed_).forward(cropped);
    auto tokens = patches.flatten(2).transpose(1, 2);
    auto cls = cls_token_.expand({input.size(0), -1, -1});
    auto extra = extra_tokens_.expand({input.size(0), -1, -1});
    tokens = torch::cat({cls, extra, tokens}, 1);
    return final_norm_->forward(blocks_->forward(tokens));
  });
}

Tensor DinoV3Impl::forward(const Tensor& input) {
  return detail::Backend("dinov3.forward", [&] {
    auto tokens = forward_features(input);
    auto cls = tokens.select(1, 0);
    return head_ ? head_->forward(cls) : cls;
  });
}

const DinoV3Options& DinoV3Impl::options() const noexcept { return options_; }

DinoV3Variant DinoV3Impl::variant() const noexcept { return variant_; }

void DinoV3Impl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("dinov3.reset_classifier", [&] {
    if (num_classes < 0 || num_classes > 1'000'000)
      detail::Invalid("dinov3.head", "nonnegative classes no greater than 1e6");
    if (num_classes == 0) {
      if (head_) {
        unregister_module("head");
        head_ = nullptr;
      }
      options_.num_classes = 0;
      return;
    }
    auto replacement = torch::nn::Linear(embedding_dimension_, num_classes);
    {
      torch::NoGradGuard guard;
      TruncatedNormal_(replacement->weight, 0.02);
      torch::nn::init::zeros_(replacement->bias);
    }
    replacement->to(PatchEmbedV3(patch_embed_).weight().device(),
                    PatchEmbedV3(patch_embed_).weight().scalar_type());
    if (head_)
      head_ = replace_module("head", replacement);
    else
      head_ = register_module("head", replacement);
    options_.num_classes = num_classes;
  });
}

DinoV3 dinov3_vits16(DinoV3Options options) {
  return DinoV3(DinoV3Variant::small, options);
}
DinoV3 dinov3_vits16plus(DinoV3Options options) {
  return DinoV3(DinoV3Variant::small_plus, options);
}
DinoV3 dinov3_vitb16(DinoV3Options options) {
  return DinoV3(DinoV3Variant::base, options);
}
DinoV3 dinov3_vitl16(DinoV3Options options) {
  return DinoV3(DinoV3Variant::large, options);
}
DinoV3 dinov3_vitl16plus(DinoV3Options options) {
  return DinoV3(DinoV3Variant::large_plus, options);
}
DinoV3 dinov3_vith16plus(DinoV3Options options) {
  return DinoV3(DinoV3Variant::huge_plus, options);
}
DinoV3 dinov3_vit7b16(DinoV3Options options) {
  return DinoV3(DinoV3Variant::seven_b, options);
}

nn::ModelAdapter adapt(DinoV3 model) {
  if (!model) detail::Invalid("dinov3.adapt", "a nonempty model");
  TensorSchema input;
  input.rank = 4;
  input.shape = {{std::nullopt, model->options().input_channels, std::nullopt,
                  std::nullopt}};
  TensorSchema output;
  output.rank = 2;
  output.shape = {{std::nullopt, model->options().num_classes > 0
                                     ? model->options().num_classes
                                     : Config(model->variant()).dimension}};
  return nn::adapt_holder(
      model,
      [](DinoV3Impl& module, const nn::Invocation& invocation) {
        return module.forward(invocation.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor(input, "images", "dino_v3_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "output", "dino_v3_logits_or_cls"));
}

}  // namespace jyppx::liuding::models
