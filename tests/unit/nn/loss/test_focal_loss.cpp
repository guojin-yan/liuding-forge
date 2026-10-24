// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/loss/focal_loss.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

namespace functional = jyppx::liuding::nn::functional;

TEST(FocalLossTest, GammaZeroMatchesCrossEntropy) {
  const auto logits = torch::randn({6, 4}, torch::kFloat64);
  const auto target = torch::randint(0, 4, {6}, torch::kLong);
  const auto focal = functional::focal_loss(logits, target, 1.0, 0.0);
  const auto ce = torch::nn::functional::cross_entropy(
      logits, target,
      torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
  EXPECT_TRUE(torch::allclose(focal, ce, 1e-10, 1e-10));
}

TEST(FocalLossTest, ComputesBinaryValueAndReductions) {
  const auto logits = torch::zeros({2, 1}, torch::kFloat64);
  const auto target = torch::tensor({{1.0}, {0.0}}, torch::kFloat64);
  const auto none = functional::focal_loss(logits, target, 0.25, 0.0, "none");
  EXPECT_NEAR(none[0].item<double>(), 0.25 * std::log(2.0), 1e-12);
  EXPECT_NEAR(none[1].item<double>(), 0.75 * std::log(2.0), 1e-12);
  EXPECT_DOUBLE_EQ(
      functional::focal_loss(logits, target, 0.25, 0.0, "sum").item<double>(),
      none.sum().item<double>());
}

TEST(FocalLossTest, BackwardProducesFiniteGradients) {
  auto logits = torch::randn(
      {8, 3},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto target = torch::randint(0, 3, {8}, torch::kLong);
  functional::focal_loss(logits, target, 0.75, 2.0).backward();
  ASSERT_TRUE(logits.grad().defined());
  EXPECT_TRUE(torch::isfinite(logits.grad()).all().item<bool>());
}

TEST(FocalLossTest, RejectsInvalidConfigurationAndContracts) {
  const auto logits = torch::randn({4, 3}, torch::kFloat32);
  const auto target = torch::zeros({4}, torch::kLong);
  EXPECT_THROW(functional::focal_loss(logits, target, -0.1),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::focal_loss(logits, target,
                                      std::numeric_limits<double>::quiet_NaN()),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::focal_loss(logits, target, 1.0, -1.0),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::focal_loss(logits, target, 1.0, 2.0, "average"),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::focal_loss(logits, torch::full({4}, 4)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      functional::focal_loss(torch::randn({2, 3, 4}), torch::zeros({2, 3, 1})),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::focal_loss(torch::zeros({2, 3}, torch::kLong),
                                      torch::zeros({2, 3}, torch::kLong)),
               jyppx::liuding::InvalidArgument);
}

TEST(FocalLossTest, ModuleValidatesAndMatchesFunctionalForm) {
  const auto logits = torch::randn({5, 3}, torch::kFloat64);
  const auto target = torch::tensor({0, 1, 2, 1, 0}, torch::kLong);
  const jyppx::liuding::nn::FocalLoss criterion(0.25, 2.0, "sum");
  EXPECT_DOUBLE_EQ(criterion.alpha(), 0.25);
  EXPECT_DOUBLE_EQ(criterion.gamma(), 2.0);
  EXPECT_EQ(criterion.reduction(), "sum");
  EXPECT_TRUE(torch::allclose(
      criterion.forward(logits, target),
      functional::focal_loss(logits, target, 0.25, 2.0, "sum"), 1e-12, 1e-12));
  EXPECT_THROW(jyppx::liuding::nn::FocalLoss(0.25, 2.0, "average"),
               jyppx::liuding::InvalidArgument);
}

}  // namespace
