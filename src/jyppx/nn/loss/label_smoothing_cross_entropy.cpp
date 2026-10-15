// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>

#include <torch/nn/functional/loss.h>

#include <cmath>
#include <utility>
#include <variant>

namespace jyppx::nn {
namespace {

using ReductionVariant =
    std::variant<torch::enumtype::kNone, torch::enumtype::kMean,
                 torch::enumtype::kSum>;

ReductionVariant parse_reduction(const std::string& reduction) {
  if (reduction == "none") {
    return torch::kNone;
  }
  if (reduction == "mean") {
    return torch::kMean;
  }
  if (reduction == "sum") {
    return torch::kSum;
  }
  throw InvalidArgument(
      "Unsupported label-smoothing reduction",
      ErrorDetails{.component = "nn.loss.label_smoothing_cross_entropy",
                   .operation = "reduction",
                   .expected = "none, mean, or sum",
                   .actual = reduction,
                   .hint = "Choose one of the documented reduction names."});
}

void validate_configuration(double smoothing, const std::string& reduction) {
  JYPPX_CHECK_ARG(
      std::isfinite(smoothing) && smoothing >= 0.0 && smoothing <= 1.0,
      "nn.loss.label_smoothing_cross_entropy", "smoothing",
      "a finite value in [0, 1]", smoothing,
      "Choose the fraction of target probability assigned uniformly.");
  static_cast<void>(parse_reduction(reduction));
}

}  // namespace

namespace functional {

Tensor label_smoothing_cross_entropy(const Tensor& input, const Tensor& target,
                                     double smoothing,
                                     const std::string& reduction) {
  validate_configuration(smoothing, reduction);
  JYPPX_CHECK_ARG(input.defined(), "nn.loss.label_smoothing_cross_entropy",
                  "input", "a defined logits tensor", "undefined",
                  "Pass the model logits returned by a forward operation.");
  JYPPX_CHECK_ARG(target.defined(), "nn.loss.label_smoothing_cross_entropy",
                  "target", "a defined class-index tensor", "undefined",
                  "Pass class indices matching the logits batch and shape.");

  try {
    return torch::nn::functional::cross_entropy(
        input, target,
        torch::nn::functional::CrossEntropyFuncOptions()
            .reduction(parse_reduction(reduction))
            .label_smoothing(smoothing));
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause,
        detail::BackendErrorContext{
            .component = "nn.loss.label_smoothing_cross_entropy",
            .operation = "torch::nn::functional::cross_entropy",
            .failure_kind = detail::BackendFailureKind::unknown,
            .expected = "compatible logits and class-index target tensors",
            .hint = "Check tensor shape, dtype, device, and target indices."});
  }
}

}  // namespace functional

LabelSmoothingCrossEntropy::LabelSmoothingCrossEntropy(double smoothing,
                                                       std::string reduction)
    : smoothing_(smoothing), reduction_(std::move(reduction)) {
  validate_configuration(smoothing_, reduction_);
}

Tensor LabelSmoothingCrossEntropy::forward(const Tensor& input,
                                           const Tensor& target) const {
  return functional::label_smoothing_cross_entropy(input, target, smoothing_,
                                                   reduction_);
}

}  // namespace jyppx::nn
