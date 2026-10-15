// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/loss/contrastive_loss.hpp>

#include <ATen/Dispatch.h>
#include <cmath>
#include <utility>

namespace jyppx::nn {
namespace {

constexpr char kComponent[] = "nn.loss.contrastive";

void validate_reduction(const std::string& reduction) {
  JYPPX_CHECK_ARG(
      reduction == "none" || reduction == "mean" || reduction == "sum",
      kComponent, "reduction", "none, mean, or sum", reduction,
      "Choose one of the documented reduction names.");
}

void validate_tensor_contract(const Tensor& x1, const Tensor& x2,
                              const Tensor& target) {
  JYPPX_CHECK_ARG(x1.defined(), kComponent, "x1", "a defined feature tensor",
                  "undefined", "Pass the first feature batch.");
  JYPPX_CHECK_ARG(x2.defined(), kComponent, "x2", "a defined feature tensor",
                  "undefined", "Pass the second feature batch.");
  JYPPX_CHECK_ARG(target.defined(), kComponent, "target",
                  "a defined binary label tensor", "undefined",
                  "Pass one label for every feature pair.");
  JYPPX_CHECK_ARG(x1.dim() == 2, kComponent, "x1.shape", "rank-2 (N, D)",
                  x1.sizes(), "Flatten each sample to one feature row.");
  JYPPX_CHECK_ARG(x2.dim() == 2, kComponent, "x2.shape", "rank-2 (N, D)",
                  x2.sizes(), "Flatten each sample to one feature row.");
  JYPPX_CHECK_ARG(x1.sizes() == x2.sizes(), kComponent, "feature_shapes",
                  "matching (N, D) shapes", x1.sizes(),
                  "Use paired feature batches with identical geometry.");
  JYPPX_CHECK_ARG(target.dim() == 1, kComponent, "target.shape", "rank-1 (N)",
                  target.sizes(), "Provide one scalar label per pair.");
  JYPPX_CHECK_ARG(target.size(0) == x1.size(0), kComponent, "target.shape",
                  "N labels matching the feature batch", target.sizes(),
                  "Match target length to the first feature dimension.");
  JYPPX_CHECK_ARG(x1.device() == x2.device(), kComponent, "feature_device",
                  "x1 and x2 on the same device",
                  x1.device().str() + " vs " + x2.device().str(),
                  "Move both feature tensors to one device.");
  JYPPX_CHECK_ARG(x1.device() == target.device(), kComponent, "target_device",
                  "target on the feature device", target.device().str(),
                  "Move labels to the feature tensor device.");
  JYPPX_CHECK_ARG(x1.scalar_type() == x2.scalar_type(), kComponent,
                  "feature_dtype", "matching feature dtypes", x1.scalar_type(),
                  "Cast both feature batches to the same floating dtype.");
  JYPPX_CHECK_ARG(
      at::isFloatingType(x1.scalar_type()), kComponent, "feature_dtype",
      "a floating-point feature dtype", x1.scalar_type(),
      "Use float32, float64, or a backend-supported floating dtype.");
  JYPPX_CHECK_ARG(at::isIntegralType(target.scalar_type(), true) ||
                      at::isFloatingType(target.scalar_type()),
                  kComponent, "target_dtype", "integral or floating labels",
                  target.scalar_type(), "Encode pair labels as zero or one.");
}

void validate_binary_target(const Tensor& target) {
  const auto options = torch::TensorOptions()
                           .dtype(target.scalar_type())
                           .device(target.device());
  const auto allowed = torch::tensor({0, 1}, options);
  JYPPX_CHECK_ARG(torch::isin(target, allowed).all().item<bool>(), kComponent,
                  "target.values", "only zero or one", "out-of-range value",
                  "Use one for matching pairs and zero otherwise.");
}

void validate_configuration(double margin, const std::string& reduction) {
  JYPPX_CHECK_ARG(std::isfinite(margin) && margin > 0.0, kComponent, "margin",
                  "a finite positive value", margin,
                  "Choose the maximum distance penalized for a negative pair.");
  validate_reduction(reduction);
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

Tensor contrastive_loss(const Tensor& x1, const Tensor& x2,
                        const Tensor& target, double margin,
                        const std::string& reduction) {
  validate_configuration(margin, reduction);
  validate_tensor_contract(x1, x2, target);
  validate_binary_target(target);

  try {
    const auto distance = (x1 - x2).norm(2.0, /*dim=*/1);
    const auto y = target.to(x1.scalar_type());
    const auto positive_term = y * distance.pow(2);
    const auto negative_term =
        (1.0 - y) * torch::clamp(margin - distance, /*min=*/0.0).pow(2);
    return reduce_loss(positive_term + negative_term, reduction);
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause,
        detail::BackendErrorContext{
            .component = kComponent,
            .operation = "contrastive tensor distance and reduction",
            .failure_kind = detail::BackendFailureKind::unknown,
            .expected = "compatible feature and binary target tensors",
            .hint = "Check the backend dtype, device, and tensor values."});
  }
}

}  // namespace functional

ContrastiveLoss::ContrastiveLoss(double margin, std::string reduction)
    : margin_(margin), reduction_(std::move(reduction)) {
  validate_configuration(margin_, reduction_);
}

Tensor ContrastiveLoss::forward(const Tensor& x1, const Tensor& x2,
                                const Tensor& target) const {
  return functional::contrastive_loss(x1, x2, target, margin_, reduction_);
}

}  // namespace jyppx::nn
