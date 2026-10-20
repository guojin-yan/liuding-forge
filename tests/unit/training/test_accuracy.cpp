// Unit tests for the jyppx::training::metrics::Accuracy metric.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/accuracy.hpp>

#include <gtest/gtest.h>
#include <limits>

namespace {

using jyppx::Tensor;
using jyppx::training::metrics::Accuracy;

TEST(AccuracyTest, PerfectPredictionsScoreOne) {
  Accuracy accuracy;
  EXPECT_EQ(accuracy.name(), "accuracy");

  // 1D predictions are binarized with (preds > 0).
  const auto preds = torch::tensor({0.1f, 2.5f, -0.7f});
  const auto targets = torch::tensor({1.0f, 1.0f, 0.0f});
  accuracy.update(preds, targets);
  EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, TwoDimensionalPredictionsUseArgmax) {
  Accuracy accuracy;
  const auto preds = torch::tensor({{0.1f, 0.9f}, {0.8f, 0.2f}, {0.3f, 0.4f}});
  const auto targets = torch::tensor({1.0f, 0.0f, 1.0f});
  accuracy.update(preds, targets);
  // argmax: {1, 0, 1}; targets {1, 0, 1} -> all match.
  EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, PartialCorrectnessScoresFraction) {
  Accuracy accuracy;
  const auto preds = torch::tensor({1.0f, 0.2f, -2.0f, 0.0f});
  const auto targets = torch::tensor({1.0f, 0.0f, 0.0f, 1.0f});
  accuracy.update(preds, targets);
  EXPECT_DOUBLE_EQ(accuracy.compute(), 0.5);
}

TEST(AccuracyTest, AccumulatesAcrossBatches) {
  Accuracy accuracy;
  accuracy.update(torch::ones({4}), torch::ones({4}));   // 4/4
  accuracy.update(torch::zeros({4}), torch::ones({4}));  // 0/4
  EXPECT_DOUBLE_EQ(accuracy.compute(), 0.5);
}

TEST(AccuracyTest, ResetClearsState) {
  Accuracy accuracy;
  accuracy.update(torch::zeros({2}), torch::ones({2}));
  EXPECT_DOUBLE_EQ(accuracy.compute(), 0.0);
  accuracy.reset();
  accuracy.update(torch::ones({2}), torch::ones({2}));
  EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, NoUpdatesScoresZero) {
  Accuracy accuracy;
  EXPECT_DOUBLE_EQ(accuracy.compute(), 0.0);
}

TEST(AccuracyTest, RejectsAmbiguousOrInvalidLabelsWithoutChangingState) {
  Accuracy accuracy;
  accuracy.update(torch::tensor({1.0f}), torch::tensor({1}));

  EXPECT_THROW(accuracy.update(torch::tensor({1.0f}), torch::tensor({1.5f})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      accuracy.update(torch::tensor({{0.1f, 0.9f}}), torch::tensor({2})),
      jyppx::InvalidArgument);
  EXPECT_THROW(accuracy.update(torch::tensor({1.0f}), torch::tensor({1.0e30f})),
               jyppx::InvalidArgument);
  EXPECT_THROW(accuracy.update(torch::tensor({1.0f}), torch::empty({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      accuracy.update(torch::tensor({std::numeric_limits<float>::quiet_NaN()}),
                      torch::tensor({0})),
      jyppx::InvalidArgument);
  EXPECT_DOUBLE_EQ(accuracy.compute(), 1.0);
}

TEST(AccuracyTest, IgnoresOnlyACompleteEmptyPair) {
  Accuracy accuracy;
  accuracy.update(torch::empty({0}), torch::empty({0}));
  EXPECT_DOUBLE_EQ(accuracy.compute(), 0.0);
}

}  // namespace
