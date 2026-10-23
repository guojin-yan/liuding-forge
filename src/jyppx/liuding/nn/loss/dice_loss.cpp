// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/loss/dice_loss.hpp>

#include <torch/nn/functional.h>

#include <cmath>
#include <utility>

namespace jyppx::liuding::nn {
namespace {

constexpr char kComponent[] = "nn.loss.dice";

struct PreparedDiceInput {
  Tensor probabilities;
  Tensor truth;
  Tensor valid_samples;
};

const char* mode_name(DiceMode mode) {
  switch (mode) {
    case DiceMode::binary:
      return "binary";
    case DiceMode::multiclass:
      return "multiclass";
    case DiceMode::multilabel:
      return "multilabel";
  }
  return "unknown";
}

const char* average_name(DiceAverage average) {
  switch (average) {
    case DiceAverage::micro:
      return "micro";
    case DiceAverage::macro:
      return "macro";
  }
  return "unknown";
}

void validate_configuration(const DiceLossOptions& options) {
  JYPPX_CHECK_ARG(options.mode == DiceMode::binary ||
                      options.mode == DiceMode::multiclass ||
                      options.mode == DiceMode::multilabel,
                  kComponent, "mode", "binary, multiclass, or multilabel",
                  mode_name(options.mode), "Choose a documented Dice mode.");
  JYPPX_CHECK_ARG(options.average == DiceAverage::micro ||
                      options.average == DiceAverage::macro,
                  kComponent, "average", "micro or macro",
                  average_name(options.average),
                  "Choose a documented class aggregation.");
  JYPPX_CHECK_ARG(std::isfinite(options.smooth) && options.smooth >= 0.0,
                  kComponent, "smooth", "a finite non-negative value",
                  options.smooth, "Use finite non-negative smoothing.");
  JYPPX_CHECK_ARG(std::isfinite(options.epsilon) && options.epsilon > 0.0,
                  kComponent, "epsilon", "a finite positive value",
                  options.epsilon, "Use a positive denominator clamp.");
  JYPPX_CHECK_ARG(options.reduction == "none" || options.reduction == "mean" ||
                      options.reduction == "sum",
                  kComponent, "reduction", "none, mean, or sum",
                  options.reduction,
                  "Choose one of the documented batch reductions.");
  JYPPX_CHECK_ARG(
      !options.ignore_index.has_value() || options.mode == DiceMode::multiclass,
      kComponent, "ignore_index", "set only for multiclass mode",
      options.ignore_index.has_value() ? "set" : "unset",
      "Remove ignore_index for binary or multilabel targets.");
  JYPPX_CHECK_ARG(
      options.include_background || options.mode != DiceMode::binary,
      kComponent, "include_background", "true for binary mode",
      options.include_background,
      "Binary mode has no explicit background channel to remove.");
}

void validate_common_tensor_contract(const Tensor& input,
                                     const Tensor& target) {
  JYPPX_CHECK_ARG(input.defined(), kComponent, "input", "a defined tensor",
                  "undefined", "Pass floating-point logits.");
  JYPPX_CHECK_ARG(target.defined(), kComponent, "target", "a defined tensor",
                  "undefined", "Pass a target matching the configured mode.");
  JYPPX_CHECK_ARG(at::isFloatingType(input.scalar_type()), kComponent,
                  "input_dtype", "a floating-point dtype", input.scalar_type(),
                  "Use a backend-supported floating logit dtype.");
  JYPPX_CHECK_ARG(input.device() == target.device(), kComponent, "device",
                  "input and target on the same device", target.device().str(),
                  "Move logits and targets to one device.");
}

void validate_binary_values(const Tensor& target) {
  JYPPX_CHECK_ARG(
      at::isIntegralType(target.scalar_type(), true) ||
          at::isFloatingType(target.scalar_type()),
      kComponent, "target_dtype", "an integral, boolean, or floating dtype",
      target.scalar_type(), "Encode binary targets as zero or one.");
  const auto binary = target.eq(0).logical_or(target.eq(1));
  JYPPX_CHECK_ARG(binary.all().item<bool>(), kComponent, "target.values",
                  "only zero or one", "out-of-range or non-finite value",
                  "Encode every binary target as exactly zero or one.");
}

PreparedDiceInput prepare_binary(const Tensor& input, const Tensor& target) {
  JYPPX_CHECK_ARG(input.dim() >= 1, kComponent, "input.shape", "rank >= 1",
                  input.sizes(), "Include a non-empty batch dimension.");
  JYPPX_CHECK_ARG(input.sizes() == target.sizes(), kComponent, "target.shape",
                  "the same shape as binary input", target.sizes(),
                  "Provide one binary target for every logit.");
  JYPPX_CHECK_ARG(input.numel() > 0 && input.size(0) > 0, kComponent,
                  "input.shape", "no empty dimensions", input.sizes(),
                  "Provide at least one sample and one value per sample.");
  validate_binary_values(target);

  const auto batch = input.size(0);
  return PreparedDiceInput{
      .probabilities = torch::sigmoid(input).reshape({batch, 1, -1}),
      .truth = target.to(input.scalar_type()).reshape({batch, 1, -1}),
      .valid_samples = torch::ones(
          {batch},
          torch::TensorOptions().dtype(torch::kBool).device(input.device()))};
}

PreparedDiceInput prepare_multilabel(const Tensor& input,
                                     const Tensor& target) {
  JYPPX_CHECK_ARG(input.dim() >= 2, kComponent, "input.shape", "rank >= 2",
                  input.sizes(),
                  "Include explicit non-empty batch and class dimensions.");
  JYPPX_CHECK_ARG(input.sizes() == target.sizes(), kComponent, "target.shape",
                  "the same shape as multilabel input", target.sizes(),
                  "Provide one binary target for every class logit.");
  JYPPX_CHECK_ARG(input.numel() > 0 && input.size(0) > 0 && input.size(1) > 0,
                  kComponent, "input.shape", "no empty dimensions",
                  input.sizes(),
                  "Provide at least one sample, class, and spatial value.");
  validate_binary_values(target);

  const auto batch = input.size(0);
  const auto classes = input.size(1);
  return PreparedDiceInput{
      .probabilities = torch::sigmoid(input).reshape({batch, classes, -1}),
      .truth = target.to(input.scalar_type()).reshape({batch, classes, -1}),
      .valid_samples = torch::ones(
          {batch},
          torch::TensorOptions().dtype(torch::kBool).device(input.device()))};
}

PreparedDiceInput prepare_multiclass(const Tensor& input, const Tensor& target,
                                     const DiceLossOptions& options) {
  JYPPX_CHECK_ARG(input.dim() >= 2, kComponent, "input.shape", "rank >= 2",
                  input.sizes(),
                  "Include explicit non-empty batch and class dimensions.");
  JYPPX_CHECK_ARG(target.dim() == input.dim() - 1, kComponent, "target.shape",
                  "one fewer dimension than input", target.sizes(),
                  "Remove the class dimension from the target.");
  JYPPX_CHECK_ARG(input.numel() > 0 && input.size(0) > 0 && input.size(1) > 0,
                  kComponent, "input.shape", "no empty dimensions",
                  input.sizes(),
                  "Provide at least one sample, class, and spatial value.");
  JYPPX_CHECK_ARG(target.size(0) == input.size(0), kComponent, "target.shape",
                  "matching batch dimension", target.sizes(),
                  "Match the target batch size to input.");
  for (int64_t dim = 1; dim < target.dim(); ++dim) {
    JYPPX_CHECK_ARG(target.size(dim) == input.size(dim + 1), kComponent,
                    "target.shape", "matching spatial dimensions",
                    target.sizes(), "Match every target spatial dimension.");
  }
  JYPPX_CHECK_ARG(at::isIntegralType(target.scalar_type(), false), kComponent,
                  "target_dtype", "a non-boolean integral class-index dtype",
                  target.scalar_type(), "Use integer class indices.");

  const auto classes = input.size(1);
  auto valid_locations = torch::ones_like(target, torch::kBool);
  if (options.ignore_index.has_value()) {
    valid_locations = target.ne(*options.ignore_index);
  }
  const auto in_range = target.ge(0).logical_and(target.lt(classes));
  const auto accepted = in_range.logical_or(valid_locations.logical_not());
  JYPPX_CHECK_ARG(accepted.all().item<bool>(), kComponent, "target.values",
                  "class indices in [0, C) or ignore_index",
                  "out-of-range class index",
                  "Use valid class indices or the configured ignore value.");

  const auto safe_target =
      torch::where(valid_locations, target, torch::zeros_like(target));
  const auto batch = input.size(0);
  const auto flat_target = safe_target.reshape({batch, -1});
  auto truth =
      torch::nn::functional::one_hot(flat_target.to(torch::kLong), classes)
          .permute({0, 2, 1})
          .to(input.scalar_type());
  const auto mask = valid_locations.reshape({batch, 1, -1});
  const auto numeric_mask = mask.to(input.scalar_type());
  truth = truth * numeric_mask;
  auto probabilities =
      torch::softmax(input, 1).reshape({batch, classes, -1}) * numeric_mask;
  return PreparedDiceInput{
      .probabilities = std::move(probabilities),
      .truth = std::move(truth),
      .valid_samples = valid_locations.reshape({batch, -1}).any(1)};
}

Tensor reduce_batch(Tensor loss, const std::string& reduction) {
  if (reduction == "none") {
    return loss;
  }
  if (reduction == "mean") {
    return loss.mean();
  }
  return loss.sum();
}

Tensor calculate_dice(PreparedDiceInput prepared,
                      const DiceLossOptions& options) {
  if (!options.include_background) {
    JYPPX_CHECK_ARG(prepared.probabilities.size(1) > 1, kComponent,
                    "include_background", "at least two explicit channels",
                    prepared.probabilities.size(1),
                    "Include background or provide foreground channels.");
    prepared.probabilities = prepared.probabilities.slice(1, 1);
    prepared.truth = prepared.truth.slice(1, 1);
  }

  auto intersection = (prepared.probabilities * prepared.truth).sum(2);
  auto prediction_sum = prepared.probabilities.sum(2);
  auto target_sum = prepared.truth.sum(2);
  auto class_mask = torch::ones_like(target_sum, torch::kBool);
  if (!options.include_empty_classes) {
    class_mask = target_sum.gt(0);
  }
  const auto selected = class_mask.to(prepared.probabilities.scalar_type());
  const auto selected_count = selected.sum(1);

  Tensor sample_loss;
  if (options.average == DiceAverage::macro) {
    const auto denominator = (prediction_sum + target_sum + options.smooth)
                                 .clamp_min(options.epsilon);
    const auto per_class =
        1.0 - (2.0 * intersection + options.smooth) / denominator;
    sample_loss = (per_class * selected).sum(1) / selected_count.clamp_min(1.0);
  } else {
    intersection = (intersection * selected).sum(1);
    prediction_sum = (prediction_sum * selected).sum(1);
    target_sum = (target_sum * selected).sum(1);
    const auto denominator = (prediction_sum + target_sum + options.smooth)
                                 .clamp_min(options.epsilon);
    sample_loss = 1.0 - (2.0 * intersection + options.smooth) / denominator;
  }

  const auto contributes =
      prepared.valid_samples.logical_and(selected_count.gt(0));
  return torch::where(contributes, sample_loss, torch::zeros_like(sample_loss));
}

}  // namespace

namespace functional {

Tensor dice_loss(const Tensor& input, const Tensor& target,
                 const DiceLossOptions& options) {
  validate_configuration(options);
  validate_common_tensor_contract(input, target);

  try {
    PreparedDiceInput prepared;
    switch (options.mode) {
      case DiceMode::binary:
        prepared = prepare_binary(input, target);
        break;
      case DiceMode::multiclass:
        prepared = prepare_multiclass(input, target, options);
        break;
      case DiceMode::multilabel:
        prepared = prepare_multilabel(input, target);
        break;
    }
    return reduce_batch(calculate_dice(std::move(prepared), options),
                        options.reduction);
  } catch (const c10::Error& cause) {
    detail::rethrow_backend_error(
        cause,
        detail::BackendErrorContext{
            .component = kComponent,
            .operation = "Dice probability conversion and aggregation",
            .failure_kind = detail::BackendFailureKind::unknown,
            .expected = "compatible non-empty Dice logits and targets",
            .hint =
                "Check backend support, tensor layout, dtype, and device."});
  }
}

}  // namespace functional

DiceLoss::DiceLoss(DiceLossOptions options) : options_(std::move(options)) {
  validate_configuration(options_);
}

Tensor DiceLoss::forward(const Tensor& input, const Tensor& target) const {
  return functional::dice_loss(input, target, options_);
}

}  // namespace jyppx::liuding::nn
