// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/nn/loss/dice_loss.hpp>

#include <torch/cuda.h>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

namespace functional = jyppx::liuding::nn::functional;
using jyppx::liuding::nn::DiceAverage;
using jyppx::liuding::nn::DiceLossOptions;
using jyppx::liuding::nn::DiceMode;

TEST(DiceLossTest, BinaryModeComputesPerSampleStatisticsAndReductions) {
  const auto logits = torch::zeros({2, 2}, torch::kFloat64);
  const auto target =
      torch::tensor({1.0, 1.0, 1.0, 0.0}, torch::kFloat64).reshape({2, 2});
  DiceLossOptions options;
  options.mode = DiceMode::binary;
  options.smooth = 0.0;
  options.epsilon = 1e-12;
  options.reduction = "none";

  const auto losses = functional::dice_loss(logits, target, options);
  ASSERT_EQ(losses.dim(), 1);
  ASSERT_EQ(losses.size(0), 2);
  EXPECT_NEAR(losses[0].item<double>(), 1.0 / 3.0, 1e-12);
  EXPECT_NEAR(losses[1].item<double>(), 0.5, 1e-12);

  options.reduction = "mean";
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              5.0 / 12.0, 1e-12);
  options.reduction = "sum";
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              5.0 / 6.0, 1e-12);
}

TEST(DiceLossTest, MulticlassModeFreezesMacroMicroAndClassSelection) {
  const auto probabilities =
      torch::tensor({0.9, 0.8, 0.1, 0.2}, torch::kFloat64).reshape({1, 2, 2});
  const auto logits = torch::log(probabilities);
  const auto target = torch::zeros({1, 2}, torch::kLong);
  DiceLossOptions options;
  options.mode = DiceMode::multiclass;
  options.smooth = 0.0;
  options.epsilon = 1e-12;

  options.average = DiceAverage::macro;
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              (1.0 - 3.4 / 3.7 + 1.0) / 2.0, 1e-12);
  options.average = DiceAverage::micro;
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              0.15, 1e-12);

  options.include_empty_classes = false;
  const double foreground_class_loss = 1.0 - 3.4 / 3.7;
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              foreground_class_loss, 1e-12);
  options.average = DiceAverage::macro;
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              foreground_class_loss, 1e-12);

  options.include_empty_classes = true;
  options.include_background = false;
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              1.0, 1e-12);
}

TEST(DiceLossTest, MultilabelModeUsesIndependentSigmoidChannels) {
  const auto logits = torch::zeros({1, 2, 2}, torch::kFloat64);
  const auto target =
      torch::tensor({1.0, 1.0, 0.0, 1.0}, torch::kFloat64).reshape({1, 2, 2});
  DiceLossOptions options;
  options.mode = DiceMode::multilabel;
  options.smooth = 0.0;
  options.epsilon = 1e-12;

  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              5.0 / 12.0, 1e-12);
  options.average = DiceAverage::micro;
  EXPECT_NEAR(functional::dice_loss(logits, target, options).item<double>(),
              1.0 - 1.5 / 2.5, 1e-12);
}

TEST(DiceLossTest, IgnoreIndexMasksValuesAndAllIgnoredSamplesAreZero) {
  auto logits = torch::zeros(
      {1, 2, 3},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto target = torch::tensor({0, 1, -1}, torch::kLong).reshape({1, 3});
  DiceLossOptions options;
  options.mode = DiceMode::multiclass;
  options.smooth = 0.0;
  options.epsilon = 1e-12;
  options.ignore_index = -1;

  const auto loss = functional::dice_loss(logits, target, options);
  EXPECT_NEAR(loss.item<double>(), 0.5, 1e-12);
  loss.backward();
  ASSERT_TRUE(logits.grad().defined());
  EXPECT_TRUE(torch::equal(logits.grad().select(2, 2),
                           torch::zeros({1, 2}, torch::kFloat64)));

  auto ignored_logits = torch::randn(
      {2, 3, 4},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto ignored_target = torch::full({2, 4}, -1, torch::kLong);
  const auto ignored_loss =
      functional::dice_loss(ignored_logits, ignored_target, options);
  EXPECT_DOUBLE_EQ(ignored_loss.item<double>(), 0.0);
  ignored_loss.backward();
  EXPECT_TRUE(
      torch::equal(ignored_logits.grad(), torch::zeros_like(ignored_logits)));
}

TEST(DiceLossTest, EmptyClassPolicyCanExcludeTargetEmptyChannels) {
  const auto logits = torch::zeros({2, 2, 3}, torch::kFloat64);
  const auto target = torch::zeros({2, 2, 3}, torch::kFloat64);
  DiceLossOptions options;
  options.mode = DiceMode::multilabel;
  options.smooth = 0.0;
  options.epsilon = 1e-12;
  options.reduction = "none";

  const auto included = functional::dice_loss(logits, target, options);
  EXPECT_TRUE(torch::equal(included, torch::ones({2}, torch::kFloat64)));
  options.include_empty_classes = false;
  const auto excluded = functional::dice_loss(logits, target, options);
  EXPECT_TRUE(torch::equal(excluded, torch::zeros({2}, torch::kFloat64)));
}

TEST(DiceLossTest, BackwardSupportsNonContiguousSpatialLogits) {
  auto base = torch::randn(
      {2, 3, 2, 4},
      torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  const auto logits = base.transpose(2, 3);
  ASSERT_FALSE(logits.is_contiguous());
  const auto target = torch::randint(0, 3, {2, 4, 2}, torch::kLong);
  DiceLossOptions options;
  options.mode = DiceMode::multiclass;

  functional::dice_loss(logits, target, options).backward();
  ASSERT_TRUE(base.grad().defined());
  EXPECT_TRUE(torch::isfinite(base.grad()).all().item<bool>());
}

TEST(DiceLossTest, ModuleOwnsValidatedOptionsAndMatchesFunctionalForm) {
  DiceLossOptions options;
  options.mode = DiceMode::multilabel;
  options.average = DiceAverage::micro;
  options.smooth = 0.5;
  options.epsilon = 1e-9;
  options.include_background = false;
  options.include_empty_classes = false;
  options.reduction = "sum";
  const jyppx::liuding::nn::DiceLoss criterion(options);
  const auto logits = torch::randn({2, 3, 4}, torch::kFloat64);
  const auto target = torch::randint(0, 2, {2, 3, 4}, torch::kLong);

  EXPECT_EQ(criterion.options().mode, DiceMode::multilabel);
  EXPECT_EQ(criterion.options().average, DiceAverage::micro);
  EXPECT_DOUBLE_EQ(criterion.options().smooth, 0.5);
  EXPECT_DOUBLE_EQ(criterion.options().epsilon, 1e-9);
  EXPECT_FALSE(criterion.options().include_background);
  EXPECT_FALSE(criterion.options().include_empty_classes);
  EXPECT_EQ(criterion.options().reduction, "sum");
  EXPECT_TRUE(torch::allclose(criterion.forward(logits, target),
                              functional::dice_loss(logits, target, options),
                              1e-12, 1e-12));
}

TEST(DiceLossTest, RejectsInvalidConfiguration) {
  const auto logits = torch::zeros({2, 2}, torch::kFloat32);
  const auto target = torch::zeros({2, 2}, torch::kFloat32);
  DiceLossOptions options;
  options.mode = DiceMode::binary;

  options.smooth = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(functional::dice_loss(logits, target, options),
               jyppx::liuding::InvalidArgument);
  options.smooth = 1.0;
  options.epsilon = 0.0;
  EXPECT_THROW(static_cast<void>(jyppx::liuding::nn::DiceLoss(options)),
               jyppx::liuding::InvalidArgument);
  options.epsilon = 1e-7;
  options.reduction = "average";
  EXPECT_THROW(functional::dice_loss(logits, target, options),
               jyppx::liuding::InvalidArgument);
  options.reduction = "mean";
  options.ignore_index = -1;
  EXPECT_THROW(functional::dice_loss(logits, target, options),
               jyppx::liuding::InvalidArgument);
  options.ignore_index.reset();
  options.include_background = false;
  EXPECT_THROW(functional::dice_loss(logits, target, options),
               jyppx::liuding::InvalidArgument);
  options.include_background = true;
  options.mode = static_cast<DiceMode>(99);
  EXPECT_THROW(functional::dice_loss(logits, target, options),
               jyppx::liuding::InvalidArgument);
  options.mode = DiceMode::binary;
  options.average = static_cast<DiceAverage>(99);
  EXPECT_THROW(functional::dice_loss(logits, target, options),
               jyppx::liuding::InvalidArgument);
}

TEST(DiceLossTest, RejectsInvalidTensorContracts) {
  DiceLossOptions binary;
  binary.mode = DiceMode::binary;
  EXPECT_THROW(functional::dice_loss(jyppx::liuding::Tensor(), jyppx::liuding::Tensor(), binary),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::dice_loss(torch::zeros({2, 2}, torch::kLong),
                                     torch::zeros({2, 2}), binary),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(functional::dice_loss(torch::zeros({2, 2}),
                                     torch::full({2, 2}, 2), binary),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      functional::dice_loss(torch::zeros({0, 2}), torch::zeros({0, 2}), binary),
      jyppx::liuding::InvalidArgument);

  DiceLossOptions multiclass;
  multiclass.mode = DiceMode::multiclass;
  EXPECT_THROW(
      functional::dice_loss(torch::zeros({2, 3, 4}),
                            torch::zeros({2, 4}, torch::kFloat32), multiclass),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      functional::dice_loss(torch::zeros({2, 3, 4}),
                            torch::full({2, 4}, 3, torch::kLong), multiclass),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      functional::dice_loss(torch::zeros({2, 3, 4}),
                            torch::zeros({2, 5}, torch::kLong), multiclass),
      jyppx::liuding::InvalidArgument);
  multiclass.include_background = false;
  EXPECT_THROW(
      functional::dice_loss(torch::zeros({2, 1, 4}),
                            torch::zeros({2, 4}, torch::kLong), multiclass),
      jyppx::liuding::InvalidArgument);

  DiceLossOptions multilabel;
  multilabel.mode = DiceMode::multilabel;
  EXPECT_THROW(functional::dice_loss(torch::zeros({2, 3, 4}),
                                     torch::zeros({2, 3, 1}), multilabel),
               jyppx::liuding::InvalidArgument);
}

TEST(DiceLossCudaTest, MulticlassHalfMatchesCpuAndBackpropagates) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }

  torch::manual_seed(20261016);
  auto cuda_logits = torch::randn({3, 4, 5}, torch::TensorOptions()
                                                 .dtype(torch::kFloat16)
                                                 .device(torch::kCUDA)
                                                 .requires_grad(true));
  auto cuda_target = torch::randint(
      0, 4, {3, 5},
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  cuda_target.select(1, 4).fill_(-1);
  DiceLossOptions options;
  options.mode = DiceMode::multiclass;
  options.average = DiceAverage::micro;
  options.ignore_index = -1;
  options.include_background = false;

  const auto actual = functional::dice_loss(cuda_logits, cuda_target, options);
  const auto expected = functional::dice_loss(
      cuda_logits.detach().to(torch::kCPU, torch::kFloat32),
      cuda_target.to(torch::kCPU), options);
  EXPECT_TRUE(torch::allclose(actual.to(torch::kCPU, torch::kFloat32), expected,
                              2e-3, 2e-3));

  actual.backward();
  ASSERT_TRUE(cuda_logits.grad().defined());
  EXPECT_TRUE(torch::isfinite(cuda_logits.grad()).all().item<bool>());
  EXPECT_TRUE(torch::equal(cuda_logits.grad().select(2, 4),
                           torch::zeros({3, 4}, cuda_logits.options())));
}

}  // namespace
