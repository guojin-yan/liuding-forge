// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/models/transformer.hpp>

#include <torch/nn/modules/normalization.h>
#include <torch/nn/modules/transformerlayer.h>

#include <cmath>
#include <utility>
namespace jyppx::models {
namespace {
void Mask(const Tensor& mask, const Tensor& input, std::int64_t first,
          std::int64_t second, const char* operation) {
  if (!mask.defined()) return;
  if (mask.layout() != torch::kStrided || mask.scalar_type() != torch::kBool ||
      mask.device() != input.device() || mask.dim() != 2 ||
      mask.size(0) != first || mask.size(1) != second)
    detail::Invalid(operation, "bool mask of exact documented shape/device");
}
}  // namespace
TransformerEncoderImpl::TransformerEncoderImpl(
    TransformerEncoderOptions options)
    : options_(options) {
  detail::Backend("transformer.construct", [&] {
    if (options.model_dimension <= 0 || options.model_dimension > 65536 ||
        options.attention_heads <= 0 || options.attention_heads > 65536 ||
        options.model_dimension % options.attention_heads != 0 ||
        options.feedforward_dimension <= 0 ||
        options.feedforward_dimension > 1048576 || options.layers <= 0 ||
        options.layers > 1024 || !std::isfinite(options.dropout) ||
        options.dropout < 0 || options.dropout >= 1)
      detail::Invalid(
          "transformer.options",
          "bounded positive widths/layers, divisible heads, dropout in [0,1)");
    auto layer_options = torch::nn::TransformerEncoderLayerOptions(
                             options.model_dimension, options.attention_heads)
                             .dim_feedforward(options.feedforward_dimension)
                             .dropout(options.dropout);
    if (options.gelu) layer_options.activation(torch::kGELU);
    auto encoder_options = torch::nn::TransformerEncoderOptions(
        torch::nn::TransformerEncoderLayer(layer_options), options.layers);
    if (options.final_norm)
      encoder_options.norm(torch::nn::AnyModule(torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({options.model_dimension}).eps(1e-5))));
    encoder_ = register_module("encoder",
                               torch::nn::TransformerEncoder(encoder_options));
  });
}
Tensor TransformerEncoderImpl::forward(const Tensor& input,
                                       const Tensor& attention_mask,
                                       const Tensor& padding_mask) {
  return detail::Backend("transformer.forward", [&] {
    detail::FloatingInput(input, encoder_->parameters().front(), 3);
    if (input.size(2) != options_.model_dimension)
      detail::Invalid("transformer.features", "configured model_dimension");
    Mask(attention_mask, input, input.size(1), input.size(1), "attention_mask");
    Mask(padding_mask, input, input.size(0), input.size(1), "padding_mask");
    // The LibTorch C++ stack is sequence-first. This explicit transpose keeps
    // public batch/sequence semantics independent of C++ frontend defaults.
    return encoder_
        ->forward(input.transpose(0, 1), attention_mask, padding_mask)
        .transpose(0, 1);
  });
}
const TransformerEncoderOptions& TransformerEncoderImpl::options()
    const noexcept {
  return options_;
}
TransformerEncoder transformer_encoder(TransformerEncoderOptions options) {
  return TransformerEncoder(options);
}
nn::ModelAdapter adapt(TransformerEncoder model) {
  if (!model) detail::Invalid("transformer.adapt", "a nonempty model");
  TensorSchema features;
  features.rank = 3;
  features.shape = std::vector<std::optional<std::int64_t>>{
      std::nullopt, std::nullopt, model->options().model_dimension};
  TensorSchema mask;
  mask.dtype = torch::kBool;
  mask.rank = 2;
  auto attention =
      ValueSchema::tensor(mask, "attention_mask", "blocked_attention");
  attention.required = false;
  auto padding = ValueSchema::tensor(mask, "padding_mask", "ignored_keys");
  padding.required = false;
  return nn::adapt_holder(
      model,
      [](TransformerEncoderImpl& module, const nn::Invocation& input) {
        const auto attention_arg = input.keyword.find("attention_mask");
        const auto padding_arg = input.keyword.find("padding_mask");
        return module.forward(input.positional.front().toTensor(),
                              attention_arg == input.keyword.end()
                                  ? Tensor{}
                                  : attention_arg->second.toTensor(),
                              padding_arg == input.keyword.end()
                                  ? Tensor{}
                                  : padding_arg->second.toTensor());
      },
      {.positional = {ValueSchema::tensor(features, "features",
                                          "embedded_sequence")},
       .keyword = {{"attention_mask", attention}, {"padding_mask", padding}},
       .allow_extra_keywords = false},
      ValueSchema::tensor(features, "encoded", "encoded_sequence"));
}
}  // namespace jyppx::models
