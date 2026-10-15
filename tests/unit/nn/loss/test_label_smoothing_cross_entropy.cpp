// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/error.hpp>
#include <jyppx/nn/loss/label_smoothing_cross_entropy.hpp>

#include <torch/cuda.h>
#include <torch/nn/functional/loss.h>

#include <cmath>
#include <exception>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <variant>

namespace {

namespace functional = jyppx::nn::functional;

using ReductionVariant =
    std::variant<torch::enumtype::kNone, torch::enumtype::kMean,
                 torch::enumtype::kSum>;

torch::Tensor reference_loss(const torch::Tensor& input,
                             const torch::Tensor& target, double smoothing,
                             ReductionVariant reduction) {
  return torch::nn::functional::cross_entropy(
      input, target,
      torch::nn::functional::CrossEntropyFuncOptions()
          .reduction(reduction)
          .label_smoothing(smoothing));
}

TEST(LabelSmoothingCrossEntropyTest, MatchesLibTorchAcrossReductions) {
  torch::manual_seed(20261015);
  const auto logits =
      torch::randn({10, 5}, torch::TensorOptions().dtype(torch::kFloat64));
  const auto target = torch::randint(0, 5, {10}, torch::kLong);

  const auto none =
      functional::label_smoothing_cross_entropy(logits, target, 0.2, "none");
  const auto sum =
      functional::label_smoothing_cross_entropy(logits, target, 0.2, "sum");
  const auto mean =
      functional::label_smoothing_cross_entropy(logits, target, 0.2, "mean");

  EXPECT_TRUE(torch::allclose(
      none, reference_loss(logits, target, 0.2, torch::kNone), 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(
      sum, reference_loss(logits, target, 0.2, torch::kSum), 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(
      mean, reference_loss(logits, target, 0.2, torch::kMean), 1e-12, 1e-12));
}

TEST(LabelSmoothingCrossEntropyTest, SupportsUnbatchedAndSpatialInputs) {
  const auto unbatched_logits =
      torch::tensor({2.0, -1.0, 0.5}, torch::kFloat64);
  const auto unbatched_target = torch::tensor(2, torch::kLong);
  EXPECT_TRUE(torch::allclose(
      functional::label_smoothing_cross_entropy(unbatched_logits,
                                                unbatched_target, 0.3),
      reference_loss(unbatched_logits, unbatched_target, 0.3, torch::kMean),
      1e-12, 1e-12));

  const auto spatial_logits =
      torch::randn({2, 4, 3, 5}, torch::TensorOptions().dtype(torch::kFloat64));
  const auto spatial_target = torch::randint(0, 4, {2, 3, 5}, torch::kLong);
  EXPECT_TRUE(torch::allclose(
      functional::label_smoothing_cross_entropy(spatial_logits, spatial_target,
                                                0.15, "none"),
      reference_loss(spatial_logits, spatial_target, 0.15, torch::kNone), 1e-12,
      1e-12));
}

TEST(LabelSmoothingCrossEntropyTest, PreservesNonContiguousInputSemantics) {
  const auto contiguous =
      torch::randn({4, 3, 2}, torch::TensorOptions().dtype(torch::kFloat64));
  const auto non_contiguous = contiguous.transpose(0, 2);
  ASSERT_FALSE(non_contiguous.is_contiguous());
  const auto target = torch::randint(0, 3, {2, 4}, torch::kLong);

  const auto actual = functional::label_smoothing_cross_entropy(
      non_contiguous, target, 0.25, "none");
  const auto expected =
      reference_loss(non_contiguous.contiguous(), target, 0.25, torch::kNone);
  EXPECT_TRUE(torch::allclose(actual, expected, 1e-12, 1e-12));
}

TEST(LabelSmoothingCrossEntropyTest, BackwardMatchesLibTorch) {
  torch::manual_seed(17);
  auto actual_logits = torch::randn(
      {7, 4},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  auto expected_logits = actual_logits.detach().clone().set_requires_grad(true);
  const auto target = torch::randint(0, 4, {7}, torch::kLong);

  const auto actual =
      functional::label_smoothing_cross_entropy(actual_logits, target, 0.4);
  const auto expected =
      reference_loss(expected_logits, target, 0.4, torch::kMean);
  actual.backward();
  expected.backward();

  EXPECT_TRUE(torch::allclose(actual, expected, 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(actual_logits.grad(), expected_logits.grad(),
                              1e-12, 1e-12));
}

TEST(LabelSmoothingCrossEntropyTest, DefinesBoundarySmoothingAndEmptyNone) {
  const auto logits = torch::tensor({{20.0, -20.0, 0.0}}, torch::kFloat64);
  const auto target = torch::tensor({0}, torch::kLong);
  EXPECT_TRUE(torch::allclose(
      functional::label_smoothing_cross_entropy(logits, target, 0.0),
      reference_loss(logits, target, 0.0, torch::kMean), 1e-12, 1e-12));
  EXPECT_TRUE(torch::allclose(
      functional::label_smoothing_cross_entropy(logits, target, 1.0),
      reference_loss(logits, target, 1.0, torch::kMean), 1e-12, 1e-12));

  const auto empty_logits = torch::empty({0, 3}, torch::kFloat32);
  const auto empty_target = torch::empty({0}, torch::kLong);
  const auto empty = functional::label_smoothing_cross_entropy(
      empty_logits, empty_target, 0.1, "none");
  EXPECT_EQ(empty.numel(), 0);
  EXPECT_EQ(empty.sizes(), empty_target.sizes());
}

TEST(LabelSmoothingCrossEntropyTest, RejectsInvalidConfiguration) {
  const auto logits = torch::randn({2, 3});
  const auto target = torch::tensor({0, 2}, torch::kLong);

  EXPECT_THROW(functional::label_smoothing_cross_entropy(logits, target, -0.01),
               jyppx::InvalidArgument);
  EXPECT_THROW(functional::label_smoothing_cross_entropy(logits, target, 1.01),
               jyppx::InvalidArgument);
  EXPECT_THROW(functional::label_smoothing_cross_entropy(
                   logits, target, std::numeric_limits<double>::quiet_NaN()),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      functional::label_smoothing_cross_entropy(logits, target, 0.1, "average"),
      jyppx::InvalidArgument);
  EXPECT_THROW(
      functional::label_smoothing_cross_entropy(torch::Tensor{}, target),
      jyppx::InvalidArgument);
  EXPECT_THROW(
      functional::label_smoothing_cross_entropy(logits, torch::Tensor{}),
      jyppx::InvalidArgument);
  EXPECT_THROW(jyppx::nn::LabelSmoothingCrossEntropy(0.1, "average"),
               jyppx::InvalidArgument);
}

TEST(LabelSmoothingCrossEntropyTest, PreservesBackendFailureContextAndCause) {
  const auto logits = torch::randn({2, 3});
  const auto invalid_target = torch::tensor({{0}, {2}}, torch::kLong);

  try {
    static_cast<void>(
        functional::label_smoothing_cross_entropy(logits, invalid_target, 0.1));
    FAIL() << "Expected the backend to reject a rank-two class target";
  } catch (const jyppx::BackendError& converted) {
    EXPECT_EQ(converted.code(), jyppx::ErrorCode::backend);
    EXPECT_EQ(converted.details().component,
              "nn.loss.label_smoothing_cross_entropy");
    EXPECT_EQ(converted.details().operation,
              "torch::nn::functional::cross_entropy");
    EXPECT_FALSE(converted.details().actual.empty());
    EXPECT_THROW(std::rethrow_if_nested(converted), c10::Error);
  }
}

TEST(LabelSmoothingCrossEntropyTest, ModuleRetainsValidatedConfiguration) {
  const auto logits = torch::randn({5, 3}, torch::kFloat64);
  const auto target = torch::randint(0, 3, {5}, torch::kLong);
  const jyppx::nn::LabelSmoothingCrossEntropy criterion(0.35, "sum");

  EXPECT_DOUBLE_EQ(criterion.smoothing(), 0.35);
  EXPECT_EQ(criterion.reduction(), "sum");
  EXPECT_TRUE(torch::allclose(
      criterion.forward(logits, target),
      functional::label_smoothing_cross_entropy(logits, target, 0.35, "sum"),
      1e-12, 1e-12));
}

TEST(LabelSmoothingCrossEntropyCudaTest, MatchesHalfPrecisionLibTorch) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  const auto logits = torch::randn({16, 7}, options);
  const auto target = torch::randint(
      0, 7, {16},
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  const auto actual =
      functional::label_smoothing_cross_entropy(logits, target, 0.2);
  const auto expected = reference_loss(logits, target, 0.2, torch::kMean);

  EXPECT_TRUE(torch::allclose(actual, expected, 1e-3, 1e-3));
}

}  // namespace
