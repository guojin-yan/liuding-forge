// Unit tests for the extended metrics: TopKAccuracy, MSE, MAE, R2Score,
// AverageMeter and MetricCollection.

#include <jyppx/training/metrics/average_meter.hpp>
#include <jyppx/training/metrics/accuracy.hpp>
#include <jyppx/training/metrics/metric_collection.hpp>
#include <jyppx/training/metrics/regression.hpp>
#include <jyppx/training/metrics/topk_accuracy.hpp>
#include <jyppx/foundation/error.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using jyppx::Tensor;
using namespace jyppx::training::metrics;

TEST(TopKAccuracyTest, CountsTopKHits) {
    TopKAccuracy top1(1);
    // Row scores; target classes.
    const auto preds = torch::tensor({{0.1f, 0.9f, 0.0f},   // argmax 1
                                      {0.8f, 0.1f, 0.1f},   // argmax 0
                                      {0.2f, 0.3f, 0.5f}}); // argmax 2
    const auto targets = torch::tensor({1.0f, 1.0f, 0.0f});
    top1.update(preds, targets);
    EXPECT_DOUBLE_EQ(top1.compute(), 1.0 / 3.0); // only the first row hits top-1

    TopKAccuracy top2(2);
    top2.update(preds, targets);
    // Row 1: top-2 = {1,0} contains 1 -> hit (0.8 leads, the 0.1 tie resolves
    // to the earlier index). Row 2: top-2 = {2,1} misses 0.
    // Row 0 hit top-1 already: total 2/3.
    EXPECT_DOUBLE_EQ(top2.compute(), 2.0 / 3.0);
}

TEST(TopKAccuracyTest, AccumulatesAcrossBatchesAndResets) {
    TopKAccuracy top2(2);
    top2.update(
        torch::tensor({{0.9f, 0.1f, 0.0f}, {0.2f, 0.8f, 0.0f}}),
        torch::tensor({0.0f, 1.0f})); // 2/2 hits
    top2.update(
        torch::tensor({{0.2f, 0.3f, 0.5f}}), torch::tensor({0.0f})); // top-2 {2,1}: miss
    EXPECT_DOUBLE_EQ(top2.compute(), 2.0 / 3.0);

    top2.reset();
    EXPECT_DOUBLE_EQ(top2.compute(), 0.0);
}

TEST(TopKAccuracyTest, KBeyondClassCountUsesAllClasses) {
    TopKAccuracy top5(5);
    top5.update(torch::tensor({{0.9f, 0.1f}}), torch::tensor({0.0f}));
    EXPECT_DOUBLE_EQ(top5.compute(), 1.0); // with 2 classes, top-5 == top-2
}

TEST(TopKAccuracyTest, RejectsOneDimensionalPredictions) {
    TopKAccuracy top3(3);
    EXPECT_THROW(
        top3.update(torch::tensor({0.1f, 0.9f}), torch::tensor({0.0f})),
        jyppx::InvalidArgument);
    EXPECT_THROW(TopKAccuracy invalid(0), jyppx::InvalidArgument);
}

TEST(MSETest, ComputesMeanSquaredError) {
    MSE mse;
    mse.update(torch::tensor({1.0f, 2.0f}), torch::tensor({0.0f, 4.0f})); // 1 + 4
    mse.update(torch::tensor({3.0f}), torch::tensor({3.0f}));             // 0
    EXPECT_DOUBLE_EQ(mse.compute(), 5.0 / 3.0);
    EXPECT_EQ(mse.name(), "mse");

    mse.reset();
    EXPECT_DOUBLE_EQ(mse.compute(), 0.0);
}

TEST(MAETest, ComputesMeanAbsoluteError) {
    MAE mae;
    mae.update(torch::tensor({1.0f, 2.0f}), torch::tensor({0.0f, 4.0f})); // 1 + 2
    mae.update(torch::tensor({-3.0f}), torch::tensor({0.0f}));            // 3
    EXPECT_DOUBLE_EQ(mae.compute(), 6.0 / 3.0);
    EXPECT_EQ(mae.name(), "mae");
}

TEST(R2ScoreTest, PerfectPredictionScoresOne) {
    R2Score r2;
    r2.update(torch::tensor({1.0f, 2.0f, 3.0f}), torch::tensor({1.0f, 2.0f, 3.0f}));
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
    r2.update(torch::tensor({2.0f, 2.0f, 2.0f}), torch::tensor({0.0f, 2.0f, 4.0f}));
    EXPECT_NEAR(r2.compute(), 0.0, 1e-12);
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

    collection.update(
        torch::tensor({1.0f, 2.0f}), torch::tensor({0.0f, 4.0f}));
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
    EXPECT_THROW(collection.at("missing"), jyppx::InvalidArgument);
    EXPECT_EQ(collection.size(), 1);
}

TEST(MetricCollectionTest, MixesClassificationAndRegression) {
    MetricCollection collection;
    collection.add(std::make_shared<Accuracy>());
    collection.add(std::make_shared<TopKAccuracy>(2));

    collection.update(
        torch::tensor({{0.9f, 0.1f}, {0.2f, 0.8f}}), torch::tensor({0.0f, 0.0f}));
    const auto values = collection.compute();
    EXPECT_DOUBLE_EQ(values.at("accuracy"), 0.5);
    EXPECT_DOUBLE_EQ(values.at("top2_accuracy"), 1.0);
}

} // namespace
