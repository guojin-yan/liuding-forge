// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/nn/loss/contrastive_loss.hpp>
#include <jyppx/liuding/nn/loss/dice_loss.hpp>
#include <jyppx/liuding/nn/loss/focal_loss.hpp>
#include <jyppx/liuding/nn/loss/label_smoothing_cross_entropy.hpp>

#include <torch/cuda.h>
#include <torch/nn/functional/loss.h>

#include <cstdint>
#include <functional>
#include <gtest/gtest.h>

namespace {

namespace functional = jyppx::liuding::nn::functional;

torch::Tensor binary_focal_reference(const torch::Tensor& logits,
                                     const torch::Tensor& target, double alpha,
                                     double gamma) {
  const auto cross_entropy =
      torch::nn::functional::binary_cross_entropy_with_logits(
          logits, target,
          torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
              .reduction(torch::kNone));
  const auto probability = torch::exp(-cross_entropy);
  const auto alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha);
  return (alpha_factor * torch::pow(1.0 - probability, gamma) * cross_entropy)
      .mean();
}

torch::Tensor multiclass_focal_reference(const torch::Tensor& logits,
                                         const torch::Tensor& target,
                                         double alpha, double gamma) {
  const auto cross_entropy = torch::nn::functional::cross_entropy(
      logits, target,
      torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
  return (alpha * torch::pow(1.0 - torch::exp(-cross_entropy), gamma) *
          cross_entropy)
      .mean();
}

torch::Tensor contrastive_reference(const torch::Tensor& x1,
                                    const torch::Tensor& x2,
                                    const torch::Tensor& target,
                                    double margin) {
  const auto distance = (x1 - x2).square().sum(1).sqrt();
  const auto labels = target.to(x1.scalar_type());
  return (labels * distance.square() +
          (1.0 - labels) * torch::clamp(margin - distance, 0.0).square())
      .mean();
}

torch::Tensor binary_dice_reference(const torch::Tensor& logits,
                                    const torch::Tensor& target, double smooth,
                                    double epsilon) {
  const auto batch = logits.size(0);
  const auto probabilities = torch::sigmoid(logits).reshape({batch, -1});
  const auto truth = target.to(logits.scalar_type()).reshape({batch, -1});
  const auto intersection = (probabilities * truth).sum(1);
  const auto denominator =
      (probabilities.sum(1) + truth.sum(1) + smooth).clamp_min(epsilon);
  return (1.0 - (2.0 * intersection + smooth) / denominator).mean();
}

double central_difference(
    const torch::Tensor& input, std::int64_t flat_index,
    const std::function<torch::Tensor(const torch::Tensor&)>& loss) {
  constexpr double kStep = 1e-6;
  torch::NoGradGuard guard;
  auto plus = input.detach().clone();
  auto minus = input.detach().clone();
  plus.reshape({-1})[flat_index].add_(kStep);
  minus.reshape({-1})[flat_index].sub_(kStep);
  return (loss(plus).item<double>() - loss(minus).item<double>()) /
         (2.0 * kStep);
}

TEST(LossReferenceMatrixTest, LabelSmoothingMatchesLibTorchGradient) {
  torch::manual_seed(20261021);
  auto actual_logits = torch::randn(
      {5, 4},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto reference_logits =
      actual_logits.detach().clone().set_requires_grad(true);
  const auto target = torch::tensor({0, 3, 1, 2, 1}, torch::kLong);

  const auto actual = functional::label_smoothing_cross_entropy(
      actual_logits, target, 0.3, "mean");
  const auto reference = torch::nn::functional::cross_entropy(
      reference_logits, target,
      torch::nn::functional::CrossEntropyFuncOptions()
          .label_smoothing(0.3)
          .reduction(torch::kMean));
  actual.backward();
  reference.backward();

  EXPECT_TRUE(torch::allclose(actual, reference, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                              1e-12, 1e-12));
}

TEST(LossReferenceMatrixTest, BinaryFocalMatchesTorchReferenceAndGradient) {
  auto actual_logits = torch::tensor(
      {{-1.2, 0.4, 2.0}, {0.7, -0.8, 1.1}},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto reference_logits =
      actual_logits.detach().clone().set_requires_grad(true);
  const auto target =
      torch::tensor({{0.0, 1.0, 1.0}, {1.0, 0.0, 0.0}}, torch::kFloat64);

  const auto actual =
      functional::focal_loss(actual_logits, target, 0.35, 1.75, "mean");
  const auto reference =
      binary_focal_reference(reference_logits, target, 0.35, 1.75);
  actual.backward();
  reference.backward();

  EXPECT_TRUE(torch::allclose(actual, reference, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                              1e-11, 1e-11));
  const auto finite_difference =
      central_difference(actual_logits, 1, [&](const torch::Tensor& value) {
        return functional::focal_loss(value, target, 0.35, 1.75, "mean");
      });
  EXPECT_NEAR(actual_logits.grad().reshape({-1})[1].item<double>(),
              finite_difference, 1e-7);
}

TEST(LossReferenceMatrixTest, MulticlassFocalMatchesTorchReferenceAndGradient) {
  auto actual_logits = torch::tensor(
      {{1.2, -0.3, 0.8}, {-0.4, 1.7, 0.2}, {0.1, -0.5, 2.1}},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto reference_logits =
      actual_logits.detach().clone().set_requires_grad(true);
  const auto target = torch::tensor({2, 1, 0}, torch::kLong);

  const auto actual =
      functional::focal_loss(actual_logits, target, 0.6, 2.25, "mean");
  const auto reference =
      multiclass_focal_reference(reference_logits, target, 0.6, 2.25);
  actual.backward();
  reference.backward();

  EXPECT_TRUE(torch::allclose(actual, reference, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                              1e-11, 1e-11));
}

TEST(LossReferenceMatrixTest, ContrastiveMatchesTorchReferenceAndGradient) {
  auto actual_x1 = torch::tensor(
      {{0.2, -0.7, 1.1}, {1.4, 0.3, -0.2}, {-0.6, 1.2, 0.5}},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto reference_x1 = actual_x1.detach().clone().set_requires_grad(true);
  const auto x2 = torch::tensor(
      {{0.9, -0.1, 0.4}, {0.2, 1.0, 0.6}, {-0.3, 0.4, 1.4}}, torch::kFloat64);
  const auto target = torch::tensor({1, 0, 1}, torch::kLong);

  const auto actual =
      functional::contrastive_loss(actual_x1, x2, target, 1.8, "mean");
  const auto reference = contrastive_reference(reference_x1, x2, target, 1.8);
  actual.backward();
  reference.backward();

  EXPECT_TRUE(torch::allclose(actual, reference, 1e-12, 1e-12));
  EXPECT_TRUE(
      torch::allclose(actual_x1.grad(), reference_x1.grad(), 1e-11, 1e-11));
  const auto finite_difference =
      central_difference(actual_x1, 3, [&](const torch::Tensor& value) {
        return functional::contrastive_loss(value, x2, target, 1.8, "mean");
      });
  EXPECT_NEAR(actual_x1.grad().reshape({-1})[3].item<double>(),
              finite_difference, 1e-7);
}

TEST(LossReferenceMatrixTest, BinaryDiceMatchesTorchReferenceAndGradient) {
  auto actual_logits = torch::tensor(
      {{-1.0, 0.3, 1.4, -0.2}, {0.8, -0.5, 0.1, 1.6}},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto reference_logits =
      actual_logits.detach().clone().set_requires_grad(true);
  const auto target = torch::tensor(
      {{0.0, 1.0, 1.0, 0.0}, {1.0, 0.0, 1.0, 1.0}}, torch::kFloat64);
  jyppx::liuding::nn::DiceLossOptions options;
  options.mode = jyppx::liuding::nn::DiceMode::binary;
  options.smooth = 0.75;
  options.epsilon = 1e-9;

  const auto actual = functional::dice_loss(actual_logits, target, options);
  const auto reference =
      binary_dice_reference(reference_logits, target, 0.75, 1e-9);
  actual.backward();
  reference.backward();

  EXPECT_TRUE(torch::allclose(actual, reference, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                              1e-11, 1e-11));
  const auto finite_difference =
      central_difference(actual_logits, 6, [&](const torch::Tensor& value) {
        return functional::dice_loss(value, target, options);
      });
  EXPECT_NEAR(actual_logits.grad().reshape({-1})[6].item<double>(),
              finite_difference, 1e-7);
}

TEST(LossReferenceMatrixCudaTest, HalfForwardBackwardMatchesReferences) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

  {
    auto actual_logits = torch::randn({12, 5}, options.requires_grad(true));
    auto reference_logits =
        actual_logits.detach().clone().set_requires_grad(true);
    const auto target = torch::randint(
        0, 5, {12},
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    const auto actual = functional::label_smoothing_cross_entropy(
        actual_logits, target, 0.2, "mean");
    const auto reference = torch::nn::functional::cross_entropy(
        reference_logits, target,
        torch::nn::functional::CrossEntropyFuncOptions()
            .label_smoothing(0.2)
            .reduction(torch::kMean));
    actual.backward();
    reference.backward();
    EXPECT_TRUE(torch::allclose(actual, reference, 2e-3, 2e-3));
    EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                                3e-3, 3e-3));
  }

  {
    auto actual_logits = torch::randn({8, 6}, options.requires_grad(true));
    auto reference_logits =
        actual_logits.detach().clone().set_requires_grad(true);
    const auto target = torch::randint(0, 2, {8, 6}, options);
    const auto actual =
        functional::focal_loss(actual_logits, target, 0.4, 1.5, "mean");
    const auto reference =
        binary_focal_reference(reference_logits, target, 0.4, 1.5);
    actual.backward();
    reference.backward();
    EXPECT_TRUE(torch::allclose(actual, reference, 3e-3, 3e-3));
    EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                                4e-3, 4e-3));
  }

  {
    auto actual_x1 = torch::randn({10, 7}, options.requires_grad(true));
    auto reference_x1 = actual_x1.detach().clone().set_requires_grad(true);
    const auto x2 = torch::randn({10, 7}, options);
    const auto target = torch::randint(
        0, 2, {10},
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
    const auto actual =
        functional::contrastive_loss(actual_x1, x2, target, 5.0, "mean");
    const auto reference = contrastive_reference(reference_x1, x2, target, 5.0);
    actual.backward();
    reference.backward();
    EXPECT_TRUE(torch::allclose(actual, reference, 4e-3, 4e-3));
    EXPECT_TRUE(
        torch::allclose(actual_x1.grad(), reference_x1.grad(), 5e-3, 5e-3));
  }

  {
    auto actual_logits = torch::randn({4, 3, 8}, options.requires_grad(true));
    auto reference_logits =
        actual_logits.detach().clone().set_requires_grad(true);
    const auto target = torch::randint(0, 2, {4, 3, 8}, options);
    jyppx::liuding::nn::DiceLossOptions dice_options;
    dice_options.mode = jyppx::liuding::nn::DiceMode::binary;
    dice_options.smooth = 0.5;
    dice_options.epsilon = 1e-4;
    const auto actual =
        functional::dice_loss(actual_logits, target, dice_options);
    const auto reference =
        binary_dice_reference(reference_logits, target, 0.5, 1e-4);
    actual.backward();
    reference.backward();
    EXPECT_TRUE(torch::allclose(actual, reference, 3e-3, 3e-3));
    EXPECT_TRUE(torch::allclose(actual_logits.grad(), reference_logits.grad(),
                                4e-3, 4e-3));
  }
}

}  // namespace
