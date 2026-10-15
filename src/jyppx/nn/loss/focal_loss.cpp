// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/loss/focal_loss.hpp>

#include <torch/nn/functional.h>

#include <cmath>
#include <utility>

namespace jyppx::nn {
namespace {

constexpr char kComponent[] = "nn.loss.focal";

void validate_reduction(const std::string& reduction) {
  JYPPX_CHECK_ARG(
      reduction == "none" || reduction == "mean" || reduction == "sum",
      kComponent, "reduction", "none, mean, or sum", reduction,
      "Choose one of the documented reduction names.");
}

void validate_configuration(double alpha, double gamma,
                            const std::string& reduction) {
  JYPPX_CHECK_ARG(std::isfinite(alpha) && alpha >= 0.0 && alpha <= 1.0,
                  kComponent, "alpha", "a finite value in [0, 1]", alpha,
                  "Use a class-balancing weight between zero and one.");
  JYPPX_CHECK_ARG(std::isfinite(gamma) && gamma >= 0.0, kComponent, "gamma",
                  "a finite non-negative value", gamma,
                  "Use a finite focusing exponent.");
  validate_reduction(reduction);
}

void validate_defined(const Tensor& input, const Tensor& target) {
  JYPPX_CHECK_ARG(input.defined(), kComponent, "input", "a defined tensor",
                  "undefined", "Pass floating-point logits.");
  JYPPX_CHECK_ARG(target.defined(), kComponent, "target", "a defined tensor",
                  "undefined", "Pass binary labels or class indices.");
  JYPPX_CHECK_ARG(
      at::isFloatingType(input.scalar_type()), kComponent, "input_dtype",
      "a floating-point dtype", input.scalar_type(),
      "Use float32, float64, or a backend-supported floating dtype.");
  JYPPX_CHECK_ARG(input.device() == target.device(), kComponent, "device",
                  "input and target on the same device", target.device().str(),
                  "Move logits and targets to one device.");
}

void validate_binary_target(const Tensor& input, const Tensor& target) {
  JYPPX_CHECK_ARG(target.sizes() == input.sizes(), kComponent, "target.shape",
                  "the same shape as binary input", target.sizes(),
                  "Use one binary label per logit.");
  JYPPX_CHECK_ARG(at::isIntegralType(target.scalar_type(), true) ||
                      at::isFloatingType(target.scalar_type()),
                  kComponent, "target_dtype", "integral or floating labels",
                  target.scalar_type(), "Encode binary labels as zero or one.");
  const auto options = torch::TensorOptions()
                           .dtype(target.scalar_type())
                           .device(target.device());
  const auto allowed = torch::tensor({0, 1}, options);
  JYPPX_CHECK_ARG(torch::isin(target, allowed).all().item<bool>(), kComponent,
                  "target.values", "only zero or one", "out-of-range value",
                  "Encode binary labels as zero or one.");
}

void validate_multiclass_target(const Tensor& input, const Tensor& target) {
  JYPPX_CHECK_ARG(input.dim() >= 2, kComponent, "input.shape",
                  "rank >= 2 for multiclass logits", input.sizes(),
                  "Use rank-2 or spatial class logits.");
  JYPPX_CHECK_ARG(target.dim() == input.dim() - 1, kComponent, "target.shape",
                  "one fewer dimension than input", target.sizes(),
                  "Remove the class dimension from the target.");
  JYPPX_CHECK_ARG(target.size(0) == input.size(0), kComponent, "target.shape",
                  "matching batch dimension", target.sizes(),
                  "Match target batch size to input.");
  for (int64_t dim = 1; dim < target.dim(); ++dim) {
    JYPPX_CHECK_ARG(target.size(dim) == input.size(dim + 1), kComponent,
                    "target.shape", "matching spatial dimensions",
                    target.sizes(), "Match every target spatial dimension.");
  }
  JYPPX_CHECK_ARG(at::isIntegralType(target.scalar_type(), true), kComponent,
                  "target_dtype", "an integral class-index dtype",
                  target.scalar_type(), "Use integer class indices.");
  const int64_t classes = input.size(1);
  JYPPX_CHECK_ARG(classes > 0, kComponent, "input.shape", "at least one class",
                  input.sizes(), "Provide a non-empty class dimension.");
  const auto allowed = torch::arange(classes, torch::TensorOptions()
                                                  .dtype(target.scalar_type())
                                                  .device(target.device()));
  JYPPX_CHECK_ARG(torch::isin(target, allowed).all().item<bool>(), kComponent,
                  "target.values", "class indices in [0, C)",
                  "out-of-range class index", "Use valid class indices.");
}

Tensor reduce_loss(Tensor loss, const std::string& reduction) {
  if (reduction == "none") {
    return loss;
  }
  if (reduction == "mean") {
    return loss.mean();
  }
  return loss.sum();
}

}  // namespace

namespace functional {

Tensor focal_loss(const Tensor& input, const Tensor& target, double alpha,
                  double gamma, const std::string& reduction) {
  validate_configuration(alpha, gamma, reduction);
  validate_defined(input, target);

  try {
    Tensor loss;
    if (input.sizes() == target.sizes()) {
      validate_binary_target(input, target);
      const auto truth = target.to(input.scalar_type());
      const auto logpt = torch::nn::functional::logsigmoid(input);
      const auto pt = torch::exp(logpt);
      const auto neg_logpt = torch::nn::functional::logsigmoid(-input);
      loss =
          -(alpha * truth * torch::pow(1.0 - pt, gamma) * logpt +
            (1.0 - alpha) * (1.0 - truth) * torch::pow(pt, gamma) * neg_logpt);
    } else {
      validate_multiclass_target(input, target);
      const auto logpt = -torch::nn::functional::cross_entropy(
          input, target,
          torch::nn::functional::CrossEntropyFuncOptions().reduction(
              torch::kNone));
      const auto pt = torch::exp(logpt);
      loss = -alpha * torch::pow(1.0 - pt, gamma) * logpt;
    }
    return reduce_loss(std::move(loss), reduction);
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause,
        detail::BackendErrorContext{
            .component = kComponent,
            .operation = "focal tensor validation and reduction",
            .failure_kind = detail::BackendFailureKind::unknown,
            .expected = "compatible floating logits and focal targets",
            .hint = "Check the backend dtype, device, and target values."});
  }
}

}  // namespace functional

FocalLoss::FocalLoss(double alpha, double gamma, std::string reduction)
    : alpha_(alpha), gamma_(gamma), reduction_(std::move(reduction)) {
  validate_configuration(alpha_, gamma_, reduction_);
}

Tensor FocalLoss::forward(const Tensor& input, const Tensor& target) const {
  return functional::focal_loss(input, target, alpha_, gamma_, reduction_);
}

}  // namespace jyppx::nn
