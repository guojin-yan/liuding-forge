// Unit tests for the extended metrics: TopKAccuracy, MSE, MAE, R2Score,
// AverageMeter and MetricCollection.

#include <jyppx/foundation/error.hpp>
#include <jyppx/training/metrics/accuracy.hpp>
#include <jyppx/training/metrics/average_meter.hpp>
#include <jyppx/training/metrics/metric_collection.hpp>
#include <jyppx/training/metrics/regression.hpp>
#include <jyppx/training/metrics/topk_accuracy.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

using jyppx::Tensor;
using namespace jyppx::training::metrics;

/**
 * @brief Minimal metric used to verify that empty stable names are rejected.
 */
class EmptyNameMetric final : public jyppx::training::Metric {
 public:
  EmptyNameMetric() : Metric("") {}

  void update(const Tensor&, const Tensor&) override {}
  double compute() override { return 0.0; }
  void reset() override {}
};

TEST(TopKAccuracyTest, CountsTopKHits) {
  TopKAccuracy top1(1);
  // Row scores; target classes.
  const auto preds = torch::tensor({{0.1f, 0.9f, 0.0f},    // argmax 1
                                    {0.8f, 0.1f, 0.1f},    // argmax 0
                                    {0.2f, 0.3f, 0.5f}});  // argmax 2
  const auto targets = torch::tensor({1.0f, 1.0f, 0.0f});
  top1.update(preds, targets);
  EXPECT_DOUBLE_EQ(top1.compute(), 1.0 / 3.0);  // only the first row hits top-1

  TopKAccuracy top2(2);
  top2.update(preds, targets);
  // Row 1: top-2 = {1,0} contains 1 -> hit (0.8 leads, the 0.1 tie resolves
  // to the earlier index). Row 2: top-2 = {2,1} misses 0.
  // Row 0 hit top-1 already: total 2/3.
  EXPECT_DOUBLE_EQ(top2.compute(), 2.0 / 3.0);
}

TEST(TopKAccuracyTest, AccumulatesAcrossBatchesAndResets) {
  TopKAccuracy top2(2);
  top2.update(torch::tensor({{0.9f, 0.1f, 0.0f}, {0.2f, 0.8f, 0.0f}}),
              torch::tensor({0.0f, 1.0f}));  // 2/2 hits
  top2.update(torch::tensor({{0.2f, 0.3f, 0.5f}}),
              torch::tensor({0.0f}));  // top-2 {2,1}: miss
  EXPECT_DOUBLE_EQ(top2.compute(), 2.0 / 3.0);

  top2.reset();
  EXPECT_DOUBLE_EQ(top2.compute(), 0.0);
}

TEST(TopKAccuracyTest, KBeyondClassCountUsesAllClasses) {
  TopKAccuracy top5(5);
  top5.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({0.0f}));
  EXPECT_DOUBLE_EQ(top5.compute(), 1.0);  // with 2 classes, top-5 == top-2
}

TEST(TopKAccuracyTest, RejectsOneDimensionalPredictions) {
  TopKAccuracy top3(3);
  EXPECT_THROW(top3.update(torch::tensor({0.1f, 0.9f}), torch::tensor({0.0f})),
               jyppx::InvalidArgument);
  EXPECT_THROW(TopKAccuracy invalid(0), jyppx::InvalidArgument);
}

TEST(TopKAccuracyTest, RejectsAmbiguousOrInvalidInputsWithoutChangingState) {
  TopKAccuracy top2(2);
  top2.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({0}));

  EXPECT_THROW(top2.update(torch::ones({1, 2, 1}), torch::tensor({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      top2.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({0.5f})),
      jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({2})),
               jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::empty({0, 2}), torch::tensor({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      top2.update(
          torch::tensor({{std::numeric_limits<float>::quiet_NaN(), 0.1f}}),
          torch::tensor({0})),
      jyppx::InvalidArgument);
  EXPECT_DOUBLE_EQ(top2.compute(), 1.0);
}

TEST(TopKAccuracyTest, RejectsUndefinedComplexAndNegativeInputs) {
  TopKAccuracy top2(2);
  top2.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({0}));

  EXPECT_THROW(top2.update(Tensor{}, torch::tensor({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::tensor({{0.9f, 0.1f}}), Tensor{}),
               jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::ones({1, 2}, torch::kComplexFloat),
                           torch::tensor({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::tensor({{0.9f, 0.1f}}),
                           torch::ones({1}, torch::kComplexFloat)),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      top2.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({-1.0f})),
      jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({-1})),
               jyppx::InvalidArgument);
  EXPECT_THROW(top2.update(torch::tensor({{0.9f, 0.1f}, {0.1f, 0.9f}}),
                           torch::tensor({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      top2.update(torch::tensor({{0.9f, 0.1f}}),
                  torch::tensor({std::numeric_limits<float>::quiet_NaN()})),
      jyppx::InvalidArgument);
  EXPECT_DOUBLE_EQ(top2.compute(), 1.0);
}

TEST(TopKAccuracyTest, IgnoresACompleteEmptyPair) {
  TopKAccuracy top2(2);
  top2.update(torch::empty({0, 2}), torch::empty({0}));
  EXPECT_DOUBLE_EQ(top2.compute(), 0.0);
}

TEST(MSETest, ComputesMeanSquaredError) {
  MSE mse;
  mse.update(torch::tensor({1.0f, 2.0f}),
             torch::tensor({0.0f, 4.0f}));                   // 1 + 4
  mse.update(torch::tensor({3.0f}), torch::tensor({3.0f}));  // 0
  EXPECT_DOUBLE_EQ(mse.compute(), 5.0 / 3.0);
  EXPECT_EQ(mse.name(), "mse");

  mse.reset();
  EXPECT_DOUBLE_EQ(mse.compute(), 0.0);
}

TEST(MAETest, ComputesMeanAbsoluteError) {
  MAE mae;
  mae.update(torch::tensor({1.0f, 2.0f}),
             torch::tensor({0.0f, 4.0f}));                    // 1 + 2
  mae.update(torch::tensor({-3.0f}), torch::tensor({0.0f}));  // 3
  EXPECT_DOUBLE_EQ(mae.compute(), 6.0 / 3.0);
  EXPECT_EQ(mae.name(), "mae");
}

TEST(R2ScoreTest, PerfectPredictionScoresOne) {
  R2Score r2;
  r2.update(torch::tensor({1.0f, 2.0f, 3.0f}),
            torch::tensor({1.0f, 2.0f, 3.0f}));
  EXPECT_DOUBLE_EQ(r2.compute(), 1.0);
}

TEST(R2ScoreTest, KnownFitScoreIsExact) {
  // targets {0, 2, 4}; predictions {1, 2, 3} -> SS_res = 1 + 0 + 1 = 2,
  // SS_tot = 8; R^2 = 1 - 2/8 = 0.75.
  R2Score r2;
  r2.update(torch::tensor({1.0f}), torch::tensor({0.0f}));
  r2.update(torch::tensor({2.0f, 3.0f}), torch::tensor({2.0f, 4.0f}));
  EXPECT_DOUBLE_EQ(r2.compute(), 0.75);
}

TEST(R2ScoreTest, MeanPredictorScoresZero) {
  R2Score r2;
  r2.update(torch::tensor({2.0f, 2.0f, 2.0f}),
            torch::tensor({0.0f, 2.0f, 4.0f}));
  EXPECT_NEAR(r2.compute(), 0.0, 1e-12);
}

TEST(R2ScoreTest, UsesStableMomentsForLargeOffsetTargets) {
  constexpr double base = 1.0e12;
  R2Score r2;
  r2.update(torch::tensor({base, base + 1.0}, torch::kFloat64),
            torch::tensor({base, base + 1.0}, torch::kFloat64));
  r2.update(torch::tensor({base + 3.0}, torch::kFloat64),
            torch::tensor({base + 2.0}, torch::kFloat64));
  EXPECT_NEAR(r2.compute(), 0.5, 1e-12);
}

TEST(RegressionMetricTest, RejectsInvalidInputWithoutChangingState) {
  MSE mse;
  mse.update(torch::tensor({1.0}), torch::tensor({0.0}));
  EXPECT_THROW(mse.update(torch::tensor({1.0}), torch::empty({0})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      mse.update(torch::tensor({std::numeric_limits<double>::infinity()}),
                 torch::tensor({0.0})),
      jyppx::InvalidArgument);
  EXPECT_THROW(mse.update(torch::tensor({1.0e308}, torch::kFloat64),
                          torch::tensor({-1.0e308}, torch::kFloat64)),
               jyppx::NumericalError);
  EXPECT_DOUBLE_EQ(mse.compute(), 1.0);
}

TEST(RegressionMetricTest, EmptyPairsAreNoOps) {
  MSE mse;
  MAE mae;
  R2Score r2;
  const auto empty = torch::empty({0});

  mse.update(empty, empty);
  mae.update(empty, empty);
  r2.update(empty, empty);
  EXPECT_DOUBLE_EQ(mse.compute(), 0.0);
  EXPECT_DOUBLE_EQ(mae.compute(), 0.0);
  EXPECT_DOUBLE_EQ(r2.compute(), 0.0);
}

TEST(RegressionMetricTest, RejectsUndefinedComplexAndMismatchedInputs) {
  MSE mse;
  EXPECT_THROW(mse.update(Tensor{}, torch::tensor({0.0f})),
               jyppx::InvalidArgument);
  EXPECT_THROW(mse.update(torch::tensor({0.0f}), Tensor{}),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      mse.update(torch::ones({1}, torch::kComplexFloat), torch::tensor({0.0f})),
      jyppx::InvalidArgument);
  EXPECT_THROW(
      mse.update(torch::tensor({0.0f}), torch::ones({1}, torch::kComplexFloat)),
      jyppx::InvalidArgument);
  EXPECT_THROW(mse.update(torch::tensor({0.0f, 1.0f}), torch::tensor({0.0f})),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      mse.update(torch::tensor({0.0f}),
                 torch::tensor({std::numeric_limits<float>::infinity()})),
      jyppx::InvalidArgument);
  EXPECT_DOUBLE_EQ(mse.compute(), 0.0);
}

TEST(MAETest, RejectsNonFiniteBatchAccumulatorAndResets) {
  MAE mae;
  EXPECT_THROW(mae.update(torch::tensor({1.0e308}, torch::kFloat64),
                          torch::tensor({-1.0e308}, torch::kFloat64)),
               jyppx::NumericalError);
  EXPECT_DOUBLE_EQ(mae.compute(), 0.0);
  mae.update(torch::tensor({2.0}), torch::tensor({1.0}));
  mae.reset();
  EXPECT_DOUBLE_EQ(mae.compute(), 0.0);
}

TEST(R2ScoreTest, DefinesConstantTargetSemanticsAndResets) {
  R2Score exact;
  exact.update(torch::tensor({2.0, 2.0}), torch::tensor({2.0, 2.0}));
  EXPECT_DOUBLE_EQ(exact.compute(), 1.0);

  R2Score inexact;
  inexact.update(torch::tensor({1.0, 1.0}), torch::tensor({2.0, 2.0}));
  EXPECT_DOUBLE_EQ(inexact.compute(), 0.0);
  inexact.reset();
  EXPECT_DOUBLE_EQ(inexact.compute(), 0.0);
}

TEST(AverageMeterTest, WeightedAveraging) {
  AverageMeter meter;
  meter.update(1.0, 3);
  meter.update(2.0, 1);
  EXPECT_DOUBLE_EQ(meter.average(), 1.25);
  EXPECT_EQ(meter.count(), 4);
  EXPECT_DOUBLE_EQ(meter.last_value(), 2.0);

  meter.reset();
  EXPECT_DOUBLE_EQ(meter.average(), 0.0);
  EXPECT_EQ(meter.count(), 0);
}

TEST(AverageMeterTest, AcceptsTensorMeans) {
  AverageMeter meter;
  meter.update(torch::tensor({1.0f, 3.0f}), 1);
  EXPECT_DOUBLE_EQ(meter.average(), 2.0);
}

TEST(AverageMeterTest, RejectsInvalidCounts) {
  AverageMeter meter;
  EXPECT_THROW(meter.update(1.0, 0), jyppx::InvalidArgument);
}

TEST(AverageMeterTest, RejectsNonFiniteValuesWithoutChangingState) {
  AverageMeter meter;
  meter.update(2.0, 2);
  EXPECT_THROW(meter.update(std::numeric_limits<double>::infinity()),
               jyppx::InvalidArgument);
  EXPECT_THROW(
      meter.update(torch::tensor({std::numeric_limits<double>::quiet_NaN()},
                                 torch::kFloat64)),
      jyppx::InvalidArgument);
  EXPECT_DOUBLE_EQ(meter.average(), 2.0);
  EXPECT_EQ(meter.count(), 2);
  EXPECT_DOUBLE_EQ(meter.last_value(), 2.0);
}

TEST(AverageMeterTest, RejectsCountAndWeightedSumOverflow) {
  AverageMeter count_meter;
  count_meter.update(0.0, std::numeric_limits<std::int64_t>::max());
  EXPECT_THROW(count_meter.update(0.0, 1), jyppx::ResourceExhausted);
  EXPECT_EQ(count_meter.count(), std::numeric_limits<std::int64_t>::max());
  EXPECT_DOUBLE_EQ(count_meter.average(), 0.0);

  AverageMeter sum_meter;
  EXPECT_THROW(sum_meter.update(std::numeric_limits<double>::max(), 2),
               jyppx::NumericalError);
  EXPECT_EQ(sum_meter.count(), 0);
  EXPECT_DOUBLE_EQ(sum_meter.average(), 0.0);
}

TEST(AverageMeterTest, RejectsUndefinedEmptyAndComplexTensors) {
  AverageMeter meter;
  EXPECT_THROW(meter.update(Tensor{}), jyppx::InvalidArgument);
  EXPECT_THROW(meter.update(torch::empty({0})), jyppx::InvalidArgument);
  EXPECT_THROW(meter.update(torch::ones({1}, torch::kComplexFloat)),
               jyppx::InvalidArgument);
  EXPECT_EQ(meter.count(), 0);
}

TEST(MetricErrorTest, PreservesStableCodeAndStructuredContext) {
  AverageMeter meter;
  bool caught = false;
  try {
    meter.update(1.0, 0);
  } catch (const jyppx::InvalidArgument& error) {
    caught = true;
    EXPECT_EQ(error.code(), jyppx::ErrorCode::invalid_argument);
    EXPECT_EQ(error.details().component, "training.metrics.average_meter");
    EXPECT_EQ(error.details().operation, "update.count");
    EXPECT_EQ(error.details().expected, "a positive observation count");
    EXPECT_EQ(error.details().actual, "0");
  }
  EXPECT_TRUE(caught);
}

TEST(MetricCollectionTest, UpdatesAndComputesAllMetrics) {
  MetricCollection collection;
  collection.add(std::make_shared<MSE>());
  collection.add(std::make_shared<MAE>());

  collection.update(torch::tensor({1.0f, 2.0f}), torch::tensor({0.0f, 4.0f}));
  const auto values = collection.compute();
  ASSERT_EQ(values.size(), 2);
  EXPECT_DOUBLE_EQ(values.at("mse"), 2.5);
  EXPECT_DOUBLE_EQ(values.at("mae"), 1.5);

  collection.reset();
  for (const auto& [name, value] : collection.compute()) {
    EXPECT_DOUBLE_EQ(value, 0.0) << name;
  }
}

TEST(MetricCollectionTest, RejectsDuplicateAndUnknownNames) {
  MetricCollection collection;
  collection.add(std::make_shared<MSE>());
  EXPECT_THROW(collection.add(std::make_shared<MSE>()), jyppx::InvalidArgument);
  EXPECT_THROW(collection.add(nullptr), jyppx::InvalidArgument);
  EXPECT_THROW(collection.add(std::make_shared<EmptyNameMetric>()),
               jyppx::InvalidArgument);
  EXPECT_THROW(collection.at("missing"), jyppx::InvalidArgument);
  EXPECT_EQ(collection.size(), 1);
}

TEST(MetricCollectionTest, MixesClassificationAndRegression) {
  MetricCollection collection;
  collection.add(std::make_shared<Accuracy>());
  collection.add(std::make_shared<TopKAccuracy>(2));

  collection.update(torch::tensor({{0.9f, 0.1f}, {0.2f, 0.8f}}),
                    torch::tensor({0.0f, 0.0f}));
  const auto values = collection.compute();
  EXPECT_DOUBLE_EQ(values.at("accuracy"), 0.5);
  EXPECT_DOUBLE_EQ(values.at("top2_accuracy"), 1.0);
}

}  // namespace
