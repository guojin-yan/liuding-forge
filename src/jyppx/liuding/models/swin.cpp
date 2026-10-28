// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#include "detail.hpp"

#include <jyppx/liuding/models/swin.hpp>

#include <torch/nn/init.h>
#include <torch/torch.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace jyppx::liuding::models {
namespace {

constexpr std::int64_t kWindow = 7;
constexpr std::int64_t kWindowArea = kWindow * kWindow;

void ValidateOptions(const SwinOptions& options) {
  if (options.num_classes < 0 || options.num_classes > 1'000'000 ||
      options.input_channels <= 0 || options.input_channels > 1024)
    detail::Invalid("swin.options",
                    "nonnegative classes and input channels in 1..1024");
}

Tensor DropPath(const Tensor& input, double rate, bool training) {
  if (!training || rate == 0.0) return input;
  const auto keep = 1.0 - rate;
  const auto mask =
      (torch::rand({input.size(0), 1, 1, 1}, input.options()) + keep).floor();
  return input * mask / keep;
}

Tensor PartitionWindows(const Tensor& input) {
  const auto batch = input.size(0);
  const auto height = input.size(1);
  const auto width = input.size(2);
  const auto channels = input.size(3);
  return input.contiguous()
      .view({batch, height / kWindow, kWindow, width / kWindow, kWindow,
             channels})
      .permute({0, 1, 3, 2, 4, 5})
      .contiguous()
      .view({-1, kWindowArea, channels});
}

Tensor ReverseWindows(const Tensor& windows, std::int64_t batch,
                      std::int64_t height, std::int64_t width) {
  const auto channels = windows.size(2);
  return windows.contiguous()
      .view({batch, height / kWindow, width / kWindow, kWindow, kWindow,
             channels})
      .permute({0, 1, 3, 2, 4, 5})
      .contiguous()
      .view({batch, height, width, channels});
}

Tensor ShiftMask(std::int64_t height, std::int64_t width,
                 const Tensor& reference) {
  auto regions = torch::zeros({1, height, width, 1}, reference.options());
  const std::array<std::int64_t, 4> rows{0, height - kWindow, height - 3,
                                         height};
  const std::array<std::int64_t, 4> columns{0, width - kWindow, width - 3,
                                            width};
  std::int64_t region = 0;
  for (std::int64_t row = 0; row < 3; ++row)
    for (std::int64_t column = 0; column < 3; ++column)
      regions.slice(1, rows[row], rows[row + 1])
          .slice(2, columns[column], columns[column + 1])
          .fill_(region++);
  auto labels = PartitionWindows(regions).squeeze(-1);
  auto differences = labels.unsqueeze(1) - labels.unsqueeze(2);
  return differences.masked_fill(differences.ne(0), -100.0)
      .masked_fill_(differences.eq(0), 0.0);
}

class WindowAttention final : public torch::nn::Module {
 public:
  WindowAttention(std::int64_t channels, std::int64_t heads)
      : channels_(channels), heads_(heads) {
    relative_position_bias_table_ = register_parameter(
        "relative_position_bias_table",
        torch::empty({(2 * kWindow - 1) * (2 * kWindow - 1), heads}));
    qkv_ = register_module("qkv", torch::nn::Linear(channels, 3 * channels));
    projection_ =
        register_module("proj", torch::nn::Linear(channels, channels));
    std::vector<std::int64_t> indexes;
    indexes.reserve(kWindowArea * kWindowArea);
    for (std::int64_t query = 0; query < kWindowArea; ++query)
      for (std::int64_t key = 0; key < kWindowArea; ++key)
        indexes.push_back(((query / kWindow - key / kWindow) + kWindow - 1) *
                              (2 * kWindow - 1) +
                          ((query % kWindow - key % kWindow) + kWindow - 1));
    position_index_ =
        torch::from_blob(indexes.data(), {kWindowArea * kWindowArea},
                         torch::TensorOptions().dtype(torch::kInt64))
            .clone();
    torch::NoGradGuard guard;
    torch::nn::init::normal_(relative_position_bias_table_, 0.0, 0.02);
  }

  Tensor forward(const Tensor& input, const Tensor& mask,
                 std::int64_t windows_per_image) {
    const auto window_batches = input.size(0);
    const auto head_width = channels_ / heads_;
    auto qkv =
        qkv_->forward(input)
            .reshape({window_batches, kWindowArea, 3, heads_, head_width})
            .permute({2, 0, 3, 1, 4});
    auto query = qkv.select(0, 0);
    auto key = qkv.select(0, 1);
    auto value = qkv.select(0, 2);
    auto attention = torch::matmul(
        query * (1.0 / std::sqrt(static_cast<double>(head_width))),
        key.transpose(-2, -1));
    auto bias = relative_position_bias_table_
                    .index_select(0, position_index_.to(input.device()))
                    .view({kWindowArea, kWindowArea, heads_})
                    .permute({2, 0, 1})
                    .unsqueeze(0);
    attention = attention + bias;
    if (mask.defined())
      attention = (attention.view({-1, windows_per_image, heads_, kWindowArea,
                                   kWindowArea}) +
                   mask.unsqueeze(0).unsqueeze(2))
                      .view({-1, heads_, kWindowArea, kWindowArea});
    attention = torch::softmax(attention, -1);
    auto output = torch::matmul(attention, value)
                      .transpose(1, 2)
                      .reshape({window_batches, kWindowArea, channels_});
    return projection_->forward(output);
  }

 private:
  std::int64_t channels_;
  std::int64_t heads_;
  Tensor relative_position_bias_table_;
  Tensor position_index_;
  torch::nn::Linear qkv_{nullptr}, projection_{nullptr};
};

class SwinMlp final : public torch::nn::Module {
 public:
  explicit SwinMlp(std::int64_t channels) {
    fc1_ = register_module("fc1", torch::nn::Linear(channels, 4 * channels));
    fc2_ = register_module("fc2", torch::nn::Linear(4 * channels, channels));
  }

  Tensor forward(const Tensor& input) {
    return fc2_->forward(torch::gelu(fc1_->forward(input)));
  }

 private:
  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};

class SwinBlock final : public torch::nn::Module {
 public:
  SwinBlock(std::int64_t channels, std::int64_t heads, bool shifted,
            double drop_path)
      : shifted_(shifted), drop_path_(drop_path) {
    norm1_ = register_module(
        "norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
    attention_ = register_module(
        "attn", std::make_shared<WindowAttention>(channels, heads));
    norm2_ = register_module(
        "norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({channels})));
    mlp_ = register_module("mlp", std::make_shared<SwinMlp>(channels));
  }

  Tensor forward(const Tensor& input) {
    const auto batch = input.size(0);
    const auto height = input.size(1);
    const auto width = input.size(2);
    auto normalized = norm1_->forward(input);
    if (shifted_) normalized = torch::roll(normalized, {-3, -3}, {1, 2});
    const auto padded_height = height + (kWindow - height % kWindow) % kWindow;
    const auto padded_width = width + (kWindow - width % kWindow) % kWindow;
    normalized = torch::constant_pad_nd(
        normalized, {0, 0, 0, padded_width - width, 0, padded_height - height});
    const auto windows_per_image = padded_height * padded_width / kWindowArea;
    const auto mask = shifted_
                          ? ShiftMask(padded_height, padded_width, normalized)
                          : Tensor{};
    auto windows = attention_->forward(PartitionWindows(normalized), mask,
                                       windows_per_image);
    auto attended = ReverseWindows(windows, batch, padded_height, padded_width)
                        .slice(1, 0, height)
                        .slice(2, 0, width)
                        .contiguous();
    if (shifted_) attended = torch::roll(attended, {3, 3}, {1, 2});
    auto output = input + DropPath(attended, drop_path_, is_training());
    return output + DropPath(mlp_->forward(norm2_->forward(output)), drop_path_,
                             is_training());
  }

 private:
  bool shifted_;
  double drop_path_;
  torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
  std::shared_ptr<WindowAttention> attention_;
  std::shared_ptr<SwinMlp> mlp_;
};

class PatchMerging final : public torch::nn::Module {
 public:
  PatchMerging(std::int64_t input_channels, std::int64_t output_channels) {
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({4 * input_channels})));
    reduction_ = register_module(
        "reduction", torch::nn::Linear(torch::nn::LinearOptions(
                                           4 * input_channels, output_channels)
                                           .bias(false)));
  }

  Tensor forward(const Tensor& input) {
    const auto batch = input.size(0);
    const auto height = input.size(1);
    const auto width = input.size(2);
    const auto channels = input.size(3);
    auto padded =
        torch::constant_pad_nd(input, {0, 0, 0, width % 2, 0, height % 2});
    const auto padded_height = padded.size(1);
    const auto padded_width = padded.size(2);
    auto merged = padded
                      .reshape({batch, padded_height / 2, 2, padded_width / 2,
                                2, channels})
                      .permute({0, 1, 3, 4, 2, 5})
                      .flatten(3);
    return reduction_->forward(norm_->forward(merged));
  }

 private:
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear reduction_{nullptr};
};

class SwinStage final : public torch::nn::Module {
 public:
  SwinStage(std::int64_t index, std::int64_t input_channels,
            std::int64_t output_channels, std::int64_t depth,
            std::int64_t heads, std::int64_t& block_index) {
    if (index > 0)
      downsample_ = register_module(
          "downsample",
          std::make_shared<PatchMerging>(input_channels, output_channels));
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (std::int64_t block = 0; block < depth; ++block) {
      const auto shifted = block % 2 != 0 && index != 3;
      blocks_->push_back(std::make_shared<SwinBlock>(
          output_channels, heads, shifted,
          0.1 * static_cast<double>(block_index++) / 11.0));
    }
  }

  Tensor forward(Tensor input) {
    if (downsample_) input = downsample_->forward(input);
    for (std::size_t index = 0; index < blocks_->size(); ++index)
      input = blocks_[index]->as<SwinBlock>()->forward(input);
    return input;
  }

 private:
  std::shared_ptr<PatchMerging> downsample_;
  torch::nn::ModuleList blocks_{nullptr};
};

}  // namespace

class SwinPatchEmbed final : public torch::nn::Module {
 public:
  explicit SwinPatchEmbed(std::int64_t input_channels) {
    projection_ = register_module(
        "proj", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(input_channels, 96, 4).stride(4)));
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({96})));
  }

  Tensor forward(const Tensor& input) {
    return norm_->forward(projection_->forward(input).permute({0, 2, 3, 1}));
  }
  const Tensor& weight() const noexcept { return projection_->weight; }

 private:
  torch::nn::Conv2d projection_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
};

class SwinHead final : public torch::nn::Module {
 public:
  explicit SwinHead(std::int64_t num_classes) {
    if (num_classes > 0)
      classifier_ = register_module("fc", torch::nn::Linear(768, num_classes));
  }

  Tensor forward(const Tensor& input) {
    auto pooled = input.mean({1, 2});
    return classifier_ ? classifier_->forward(pooled) : pooled;
  }

  void reset(std::int64_t num_classes, const Tensor& reference) {
    if (num_classes == 0) {
      if (classifier_) {
        unregister_module("fc");
        classifier_ = nullptr;
      }
      return;
    }
    auto replacement = torch::nn::Linear(768, num_classes);
    replacement->to(reference.device(), reference.scalar_type());
    replacement->train(is_training());
    if (classifier_)
      classifier_ = replace_module("fc", replacement);
    else
      classifier_ = register_module("fc", replacement);
  }

 private:
  torch::nn::Linear classifier_{nullptr};
};

SwinTinyImpl::SwinTinyImpl(SwinOptions options) : options_(options) {
  detail::Backend("swin.construct", [&] {
    ValidateOptions(options_);
    patch_embed_ = register_module(
        "patch_embed",
        std::make_shared<SwinPatchEmbed>(options_.input_channels));
    layers_ = register_module("layers", torch::nn::ModuleList());
    constexpr std::array<std::int64_t, 4> channels{96, 192, 384, 768};
    constexpr std::array<std::int64_t, 4> depths{2, 2, 6, 2};
    constexpr std::array<std::int64_t, 4> heads{3, 6, 12, 24};
    std::int64_t block_index = 0;
    for (std::int64_t index = 0; index < 4; ++index)
      layers_->push_back(std::make_shared<SwinStage>(
          index, index == 0 ? channels[0] : channels[index - 1],
          channels[index], depths[index], heads[index], block_index));
    norm_ = register_module(
        "norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({768})));
    head_ = register_module("head",
                            std::make_shared<SwinHead>(options_.num_classes));
  });
}

std::vector<Tensor> SwinTinyImpl::forward_features(const Tensor& input) {
  return detail::Backend("swin.forward_features", [&] {
    detail::FloatingInput(input, patch_embed_->weight(), 4);
    if (input.size(1) != options_.input_channels ||
        (options_.dynamic_image_size
             ? (input.size(2) < 4 || input.size(3) < 4 ||
                input.size(2) % 4 != 0 || input.size(3) % 4 != 0)
             : (input.size(2) != 224 || input.size(3) != 224)))
      detail::Invalid("swin.input",
                      options_.dynamic_image_size
                          ? "NCHW images with positive H/W divisible by 4"
                          : "NCHW images of exactly 224x224");
    auto output = patch_embed_->forward(input);
    std::vector<Tensor> stages;
    stages.reserve(4);
    for (std::size_t index = 0; index < layers_->size(); ++index) {
      output = layers_[index]->as<SwinStage>()->forward(output);
      stages.push_back(output.permute({0, 3, 1, 2}).contiguous());
    }
    return stages;
  });
}

Tensor SwinTinyImpl::forward(const Tensor& input) {
  return detail::Backend("swin.forward", [&] {
    auto stages = forward_features(input);
    auto final = stages.back().permute({0, 2, 3, 1});
    return head_->forward(norm_->forward(final));
  });
}

void SwinTinyImpl::reset_classifier(std::int64_t num_classes) {
  detail::Backend("swin.reset_classifier", [&] {
    if (num_classes < 0 || num_classes > 1'000'000)
      detail::Invalid("swin.head", "nonnegative classes no greater than 1e6");
    head_->reset(num_classes, norm_->weight);
    options_.num_classes = num_classes;
  });
}

const SwinOptions& SwinTinyImpl::options() const noexcept { return options_; }

SwinTiny swin_tiny_patch4_window7_224(SwinOptions options) {
  return SwinTiny(options);
}

nn::ModelAdapter adapt(SwinTiny model) {
  if (!model) detail::Invalid("swin.adapt", "a nonempty model");
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
      [](SwinTinyImpl& module, const nn::Invocation& invocation) {
        return module.forward(invocation.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor(input, "images", "swin_images")},
       .keyword = {},
       .allow_extra_keywords = false},
      ValueSchema::tensor(output, "output", "swin_logits_or_embedding"));
}

}  // namespace jyppx::liuding::models
