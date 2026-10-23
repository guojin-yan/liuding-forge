// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/loss/contrastive_loss.hpp>

#include <torch/cuda.h>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

namespace functional = jyppx::liuding::nn::functional;

TEST(ContrastiveLossTest, ComputesMatchingAndNonMatchingPairs) {
  const auto x1 = torch::tensor({{0.0, 0.0}, {0.0, 0.0}}, torch::kFloat64);
  const auto x2 = torch::tensor({{3.0, 0.0}, {1.0, 0.0}}, torch::kFloat64);
  const auto target = torch::tensor({1, 0}, torch::kLong);

  const auto loss = functional::contrastive_loss(x1, x2, target, 2.0, "none");
  EXPECT_NEAR(loss[0].item<double>(), 9.0, 1e-12);
  EXPECT_NEAR(loss[1].item<double>(), 1.0, 1e-12);
}

TEST(ContrastiveLossTest, SupportsAllReductionsAndBackward) {
  auto x1 = torch::tensor(
      {{0.0, 0.0}, {1.0, 0.0}},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto x2 = torch::tensor({{0.5, 0.0}, {1.5, 0.0}}, torch::kFloat64);
  const auto target = torch::tensor({1.0, 0.0}, torch::kFloat64);

  const auto none = functional::contrastive_loss(x1, x2, target, 2.0, "none");
  const auto mean = functional::contrastive_loss(x1, x2, target, 2.0, "mean");
  const auto sum = functional::contrastive_loss(x1, x2, target, 2.0, "sum");
  EXPECT_EQ(none.sizes(), torch::IntArrayRef({2}));
  EXPECT_DOUBLE_EQ(mean.item<double>(), none.mean().item<double>());
  EXPECT_DOUBLE_EQ(sum.item<double>(), none.sum().item<double>());

  mean.backward();
  ASSERT_TRUE(x1.grad().defined());
  EXPECT_TRUE(torch::isfinite(x1.grad()).all().item<bool>());
}

TEST(ContrastiveLossTest, RejectsInvalidConfigurationAndTensorContracts) {
  const auto x1 = torch::randn({4, 3}, torch::kFloat32);
  const auto x2 = torch::randn({4, 3}, torch::kFloat32);
  const auto target = torch::zeros({4}, torch::kLong);

  EXPECT_THROW(functional::contrastive_loss(x1, x2, target, 0.0),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::contrastive_loss(
                   x1, x2, target, std::numeric_limits<double>::quiet_NaN()),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::contrastive_loss(x1, x2, target, 1.0, "average"),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::contrastive_loss(torch::Tensor{}, x2, target),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::contrastive_loss(x1, torch::randn({4, 2}), target),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::contrastive_loss(x1, x2, torch::zeros({4, 1})),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::contrastive_loss(x1, x2, torch::full({4}, 2)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      functional::contrastive_loss(torch::randn({2, 4, 3}),
                                   torch::randn({2, 4, 3}), torch::zeros({2})),
      jyppx::liuding::InvalidArgument);
}

TEST(ContrastiveLossTest, ModuleValidatesAndMatchesFunctionalForm) {
  const auto x1 = torch::randn({5, 3}, torch::kFloat64);
  const auto x2 = torch::randn({5, 3}, torch::kFloat64);
  const auto target = torch::tensor({0, 1, 0, 1, 0}, torch::kLong);
  const jyppx::liuding::nn::ContrastiveLoss criterion(0.75, "sum");

  EXPECT_DOUBLE_EQ(criterion.margin(), 0.75);
  EXPECT_EQ(criterion.reduction(), "sum");
  EXPECT_TRUE(torch::allclose(
      criterion.forward(x1, x2, target),
      functional::contrastive_loss(x1, x2, target, 0.75, "sum"), 1e-12, 1e-12));
  EXPECT_THROW(jyppx::liuding::nn::ContrastiveLoss(0.75, "average"),
               jyppx::liuding::InvalidArgument);
}

TEST(ContrastiveLossCudaTest, ValidatesLabelsOnCuda) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  const auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const auto x1 = torch::randn({4, 3}, options);
  const auto x2 = torch::randn({4, 3}, options);
  const auto target = torch::tensor(
      {1, 0, 1, 0},
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  const auto loss = functional::contrastive_loss(x1, x2, target, 1.5);
  EXPECT_TRUE(torch::isfinite(loss).item<bool>());
}

}  // namespace
