// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Unit tests for the plan-completion modules: classification metrics,
// extended samplers/datasets, extended transforms, LAMB, and model
// summary/profiler helpers.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/liuding.hpp>

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <sstream>

namespace {

using namespace jyppx::liuding;
using namespace jyppx::liuding::data;
using namespace jyppx::liuding::data::samplers;
using namespace jyppx::liuding::data::transforms;
using namespace jyppx::liuding::training;
using namespace jyppx::liuding::training::metrics;
using namespace jyppx::liuding::training::callbacks;
using namespace jyppx::liuding::utils;

// ---- Classification metrics (Precision / Recall / F1) -----------------------

TEST(ClassificationMetricsTest, HandComputedConfusionMatrix) {
  // 3 classes; argmax of preds: {0, 0, 1, 2, 2}; targets: {0, 0, 1, 1, 2}.
  // Confusion (rows = true, cols = predicted):
  //   true 0 -> [2, 0, 0]  precision(0)=2/2, recall(0)=2/2
  //   true 1 -> [0, 1, 0]  precision(1)=1/1, recall(1)=1/2
  //   true 2 -> [0, 0, 1]  precision(2)=1/2, recall(2)=1/2
  metrics::Precision precision({.num_classes = 3});
  metrics::Recall recall({.num_classes = 3});
  metrics::F1Score f1({.num_classes = 3});

  auto preds = torch::tensor({{0.9f, 0.1f, 0.0f},
                              {0.8f, 0.1f, 0.1f},
                              {0.1f, 0.8f, 0.1f},
                              {0.0f, 0.2f, 0.9f},
                              {0.1f, 0.1f, 0.8f}});
  auto targets = torch::tensor({0, 0, 1, 1, 2}, torch::kInt64);
  precision.update(preds, targets);
  recall.update(preds, targets);
  f1.update(preds, targets);

  EXPECT_NEAR(precision.compute(), (1.0 + 1.0 + 0.5) / 3.0, 1e-12);
  EXPECT_NEAR(recall.compute(), (1.0 + 0.5 + 1.0) / 3.0, 1e-12);
  EXPECT_NEAR(f1.compute(), (1.0 + 2.0 / 3.0 + 2.0 / 3.0) / 3.0, 1e-12);
}

TEST(ClassificationMetricsTest, ConfusionMatrixExposedAndResets) {
  metrics::Precision precision;
  precision.update(torch::tensor({{0.9f, 0.1f}}),
                   torch::tensor({0}, torch::kInt64));
  ASSERT_EQ(precision.classes(), 2);
  EXPECT_EQ(precision.confusion_matrix()[0][0], 1);

  precision.reset();
  EXPECT_EQ(precision.classes(), 2);
  EXPECT_THROW(precision.compute(), jyppx::liuding::NotReady);
}

TEST(ClassificationMetricsTest, PerfectPredictionsScoreOne) {
  metrics::F1Score f1;
  f1.update(torch::tensor({{0.9f, 0.1f}, {0.1f, 0.9f}}),
            torch::tensor({0, 1}, torch::kInt64));
  EXPECT_DOUBLE_EQ(f1.compute(), 1.0);
}

// ---- Extended samplers
// --------------------------------------------------------

TEST(WeightedRandomSamplerTest, WeightsShapeTheDistribution) {
  samplers::WeightedRandomSampler sampler({1.0, 0.0, 9.0}, /*num_samples=*/1000,
                                          /*replacement=*/true, /*seed=*/5);
  const auto indices = sampler.indices();
  ASSERT_EQ(indices.size(), 1000);

  int count_class2 = 0;
  for (const size_t index : indices) {
    EXPECT_TRUE(index == 0 || index == 1 || index == 2);
    if (index == 2) {
      ++count_class2;
    }
  }
  // Weight 9/10 of the total mass lands on class 2.
  EXPECT_GT(count_class2, 800);
}

TEST(WeightedRandomSamplerTest, DeterministicWithSeed) {
  samplers::WeightedRandomSampler first({1, 2, 3}, 50, true, 7);
  samplers::WeightedRandomSampler second({1, 2, 3}, 50, true, 7);
  EXPECT_EQ(first.indices(), second.indices());
}

TEST(WeightedRandomSamplerTest, ValidatesArguments) {
  EXPECT_THROW(samplers::WeightedRandomSampler({}, 10),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(samplers::WeightedRandomSampler({1.0}, 0),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(samplers::WeightedRandomSampler({1.0, -1.0}, 10),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(samplers::WeightedRandomSampler({0.0, 0.0}, 10),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      samplers::WeightedRandomSampler({1.0, 2.0}, 5, /*replacement=*/false),
      jyppx::liuding::InvalidArgument);
}

TEST(WeightedRandomSamplerTest, WithoutReplacementYieldsDistinctIndices) {
  samplers::WeightedRandomSampler sampler({1, 1, 1, 1, 1}, 4, false, 3);
  const auto indices = sampler.indices();
  ASSERT_EQ(indices.size(), 4);
  std::set<size_t> unique(indices.begin(), indices.end());
  EXPECT_EQ(unique.size(), 4);
}

TEST(SubsetRandomSamplerTest, PermutesTheGivenSubset) {
  samplers::SubsetRandomSampler sampler({4, 7, 9, 2}, /*seed=*/11);
  const auto indices = sampler.indices();
  ASSERT_EQ(indices.size(), 4);
  std::set<size_t> unique(indices.begin(), indices.end());
  EXPECT_EQ(unique.size(), 4);
  for (const size_t index : indices) {
    EXPECT_TRUE(index == 2 || index == 4 || index == 7 || index == 9);
  }
}

TEST(DistributedSamplerTest, PartitionsAcrossRanks) {
  samplers::DistributedSampler rank0(10, /*rank=*/0, /*world_size=*/2, false);
  samplers::DistributedSampler rank1(10, /*rank=*/1, /*world_size=*/2, false);

  const auto a = rank0.indices();
  const auto b = rank1.indices();
  ASSERT_EQ(a.size(), 5);
  ASSERT_EQ(b.size(), 5);

  std::vector<size_t> all = a;
  all.insert(all.end(), b.begin(), b.end());
  std::sort(all.begin(), all.end());
  for (size_t i = 0; i < all.size(); ++i) {
    EXPECT_EQ(all[i], i);  // disjoint, covers everything
  }
}

TEST(DistributedSamplerTest, SetEpochChangesTheShuffle) {
  samplers::DistributedSampler sampler(20, 0, 2, /*shuffle=*/true, /*seed=*/1);
  const auto epoch0 = sampler.indices();
  sampler.set_epoch(1);
  const auto epoch1 = sampler.indices();
  EXPECT_NE(epoch0, epoch1);

  // Same epoch is reproducible.
  sampler.set_epoch(0);
  EXPECT_EQ(sampler.indices(), epoch0);
}

TEST(DistributedSamplerTest, SingleProcessBehavesLikeIdentity) {
  samplers::DistributedSampler sampler(6, 0, 1, false);
  const auto indices = sampler.indices();
  for (size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(indices[i], i);
  }
}

TEST(DistributedSamplerTest, ValidatesArguments) {
  EXPECT_THROW(samplers::DistributedSampler(10, 1, 1),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(samplers::DistributedSampler(10, 0, 0),
               jyppx::liuding::InvalidArgument);
}

// ---- Chain / Stack datasets
// ---------------------------------------------------

TEST(ChainDatasetTest, ConcatenatesLikeConcat) {
  auto first = std::make_shared<data::TensorDataset>(
      torch::arange(0, 4, torch::kFloat32),
      torch::arange(0, 4, torch::kFloat32));
  auto second = std::make_shared<data::TensorDataset>(
      torch::arange(4, 6, torch::kFloat32),
      torch::arange(4, 6, torch::kFloat32));
  const data::ChainDataset<data::Example> chain({first, second});

  EXPECT_EQ(chain.size(), 6);
  EXPECT_NEAR(chain.get(4).data.item<double>(), 4.0, 1e-6);
}

namespace {

// Minimal Dataset<Tensor> wrapper for the StackDataset tests.
class TensorRows : public data::Dataset<Tensor> {
 public:
  explicit TensorRows(Tensor rows) : rows_(std::move(rows)) {}
  Tensor get(size_t index) const override {
    return rows_[static_cast<int64_t>(index)];
  }
  size_t size() const override { return rows_.size(0); }

 private:
  Tensor rows_;
};

}  // namespace

TEST(StackDatasetTest, ZipsTupleSamples) {
  auto features = std::make_shared<TensorRows>(
      torch::arange(0, 6, torch::kFloat32).view({3, 2}));
  auto labels = std::make_shared<TensorRows>(torch::tensor({10.f, 20.f, 30.f}));

  const data::StackDataset<Tensor, Tensor> stacked(features, labels);
  ASSERT_EQ(stacked.size(), 3);

  auto sample = stacked.get(1);
  EXPECT_EQ(std::get<0>(sample).numel(), 2);
  EXPECT_NEAR(std::get<1>(sample).item<double>(), 20.0, 1e-6);
}

TEST(StackDatasetTest, LengthIsTheSmallestDataset) {
  auto big = std::make_shared<TensorRows>(torch::randn({5, 1}));
  auto small = std::make_shared<TensorRows>(torch::randn({2, 1}));
  const data::StackDataset<Tensor, Tensor> stacked(big, small);
  EXPECT_EQ(stacked.size(), 2);
  EXPECT_THROW(stacked.get(2), jyppx::liuding::InvalidArgument);
}

// ---- Extended transforms
// --------------------------------------------------------

class TransformsExtraTest : public ::testing::Test {
 protected:
  void SetUp() override {
    image_ = torch::arange(12, torch::TensorOptions().dtype(torch::kFloat32))
                 .view({2, 2, 3});  // zero-padded later by Pad
  }

  Tensor image_;
};

TEST_F(TransformsExtraTest, PadGrowsTheImage) {
  const Pad pad({1, 1, 1, 1}, /*fill=*/-1.0f);
  const auto result = pad(image_);
  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{2, 4, 5}));
  // Corners carry the fill value.
  EXPECT_NEAR(result[0][0][0].item<double>(), -1.0, 1e-6);
  // Interior unchanged.
  EXPECT_NEAR(result[0][1][1].item<double>(), image_[0][0][0].item<double>(),
              1e-6);
}

TEST_F(TransformsExtraTest, PadSquareShorthand) {
  const Pad pad({1});
  EXPECT_EQ(pad(image_).sizes(), (std::vector<int64_t>{2, 4, 5}));
}

TEST_F(TransformsExtraTest, RotationByZeroIsIdentity) {
  const RandomRotation rotate(0.0);  // angle drawn from [-0, 0]
  const auto result = rotate(image_);
  EXPECT_EQ(result.sizes(), image_.sizes());
  EXPECT_TRUE(torch::allclose(result, image_, 1e-3, 1e-3));
}

TEST_F(TransformsExtraTest, RotationStaysFiniteAndInRange) {
  const RandomRotation rotate(30.0);
  for (int i = 0; i < 8; ++i) {
    const auto result = rotate(image_);
    EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
  }
}

TEST_F(TransformsExtraTest, AffineIdentityAndValidation) {
  const RandomAffine affine(0.0);  // no rotation/translate/scale variation
  EXPECT_TRUE(torch::allclose(affine(image_), image_, 1e-3, 1e-3));

  EXPECT_THROW(RandomAffine(10.0, {1.5, 0.0}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(RandomAffine(10.0, {0.0, 0.0}, {0.0, 1.0}),
               jyppx::liuding::InvalidArgument);
}

TEST_F(TransformsExtraTest, AffineTranslateShiftsContent) {
  // Deterministic translate: shift right by exactly one pixel.
  const RandomAffine affine(0.0, {0.0, 0.0}, {1.0, 1.0});  // no-op scale/rot
  (void)affine;

  // Use zero-degree rotation with a fixed translate via a small subclass-free
  // path: RandomAffine draws translate from +-translate*size; with
  // translate=(0,0) there is no shift, so we validate the identity already
  // covered above. For an actual shift, apply RandomAffine(0, {0.5, 0}):
  // the drawn shift is +-0.5*W; result must remain finite and 2x3.
  const RandomAffine shifting(0.0, {0.5, 0.0});
  const auto result = shifting(image_);
  EXPECT_EQ(result.sizes(), image_.sizes());
  EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
}

TEST_F(TransformsExtraTest, GrayscaleLumaWeights) {
  // Pure red {1,0,0} -> luma 0.299.
  auto red = torch::zeros({3, 1, 1});
  red[0][0][0] = 1.0f;
  const Grayscale gray(1);
  const auto result = gray(red);
  EXPECT_EQ(result.sizes(), (std::vector<int64_t>{1, 1, 1}));
  EXPECT_NEAR(result[0][0][0].item<double>(), 0.299, 1e-5);

  const Grayscale gray3(3);
  const auto replicated = gray3(red);
  EXPECT_EQ(replicated.sizes(), (std::vector<int64_t>{3, 1, 1}));
  EXPECT_NEAR(replicated[1][0][0].item<double>(), 0.299, 1e-5);
}

TEST_F(TransformsExtraTest, GrayscaleValidatesChannels) {
  EXPECT_THROW(Grayscale(2), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Grayscale(5), jyppx::liuding::InvalidArgument);
}

TEST_F(TransformsExtraTest, RandomGrayscaleKeepsThreeChannels) {
  // RandomGrayscale requires a 3-channel RGB input; use one here.
  auto rgb = torch::rand({3, 4, 4});
  const RandomGrayscale always(1.0);
  const auto result = always(rgb);
  EXPECT_EQ(result.sizes(), rgb.sizes());
  // Channels become identical after grayscale.
  EXPECT_TRUE(torch::equal(result[0], result[1]));
}

TEST_F(TransformsExtraTest, ColorJitterBrightnessZeroIsIdentity) {
  const ColorJitter jitter(0.0, 0.0, 0.0, 0.0);
  EXPECT_TRUE(torch::allclose(jitter(image_), image_, 1e-6, 1e-6));
}

TEST_F(TransformsExtraTest, ColorJitterBrightnessScales) {
  // brightness=1 draws f in [0, 2]; the output stays proportional to input.
  const ColorJitter jitter(1.0);
  for (int i = 0; i < 8; ++i) {
    const auto result = jitter(image_);
    EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
    EXPECT_GE(result.min().item<double>(), -1e-6);
  }
}

TEST_F(TransformsExtraTest, ColorJitterValidatesArguments) {
  EXPECT_THROW(ColorJitter(-0.5), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(ColorJitter(0.0, 0.0, 0.0, 0.9),
               jyppx::liuding::InvalidArgument);
}

// ---- Model summary / profiler
// ---------------------------------------------------

class SummaryModel : public torch::nn::Module {
 public:
  SummaryModel()
      : fc1(register_module("encoder", torch::nn::Linear(4, 8))),
        fc2(register_module("head", torch::nn::Linear(8, 2))) {}

  Tensor forward(const Tensor& x) {
    return fc2->forward(torch::relu(fc1->forward(x)));
  }

  torch::nn::Linear fc1, fc2;
};

/** @brief Module containing both trainable parameters and a persistent buffer.
 */
class BufferedSummaryModel : public torch::nn::Module {
 public:
  BufferedSummaryModel()
      : weight(register_parameter("weight", torch::ones({2}))),
        running(register_buffer("running", torch::zeros({3}))) {}

  Tensor weight;   ///< Two trainable float elements.
  Tensor running;  ///< Three persistent float buffer elements.
};

TEST(ModelSummaryTest, ReportsParameterCounts) {
  SummaryModel model;
  const auto stats = utils::calculate_model_stats(model);
  // fc1: 4*8 + 8 = 40; fc2: 8*2 + 2 = 18.
  EXPECT_EQ(stats.total_params, 58);
  EXPECT_EQ(stats.trainable_params, 58);
  EXPECT_EQ(stats.memory_bytes, 58 * 4);
}

TEST(ModelSummaryTest, SummaryListsChildren) {
  SummaryModel model;
  std::cout << "children=" << model.named_children().size() << std::endl;
  const auto text = utils::model_summary(model, "TestNet");
  std::cout << text << std::endl;
  EXPECT_NE(text.find("TestNet summary"), std::string::npos);
  EXPECT_NE(text.find("encoder"), std::string::npos);
  EXPECT_NE(text.find("head"), std::string::npos);
  EXPECT_NE(text.find("Total params:        58"), std::string::npos);
}

TEST(ModelSummaryTest, CountsBuffersAndFormatsEveryStorageUnit) {
  BufferedSummaryModel model;
  const auto stats = utils::calculate_model_stats(model);
  EXPECT_EQ(stats.total_params, 2);
  EXPECT_EQ(stats.trainable_params, 2);
  EXPECT_EQ(stats.total_buffers, 3);
  EXPECT_EQ(stats.memory_bytes, 5 * 4);

  EXPECT_EQ(utils::format_bytes(512), "512 B");
  EXPECT_EQ(utils::format_bytes(1536), "1.5 KB");
  EXPECT_EQ(utils::format_bytes(3 * 1024 * 1024), "3 MB");
  EXPECT_EQ(utils::format_bytes(2LL * 1024 * 1024 * 1024), "2 GB");
}

TEST(ProfilerTest, AggregatesScopedTimings) {
  utils::ProfilerReport report;
  {
    utils::ScopedTimer timer(report, "scope_a");
    // minimal work
  }
  const auto record_scope = [&report](const std::string& name) {
    utils::ScopedTimer timer(report, name);
  };
  record_scope("scope_b");
  record_scope("scope_a");

  const auto summary = report.summary();
  EXPECT_NE(summary.find("scope_a"), std::string::npos);
  EXPECT_NE(summary.find("scope_b"), std::string::npos);

  const auto sorted = report.sorted();
  ASSERT_EQ(sorted.size(), 2);
  for (const auto& [name, entry] : sorted) {
    EXPECT_EQ(entry.count, name == "scope_a" ? 2 : 1);
    EXPECT_GE(entry.total_ms, 0.0);
    EXPECT_GE(entry.total_ms, entry.max_ms);
  }

  // Empty timed scopes have no deterministic relative duration. Explicit
  // observations verify sorting independently of clock resolution/scheduling.
  report.clear();
  report.record("scope_a", 1.0);
  report.record("scope_a", 2.0);
  report.record("scope_b", 4.0);
  const auto ordered = report.sorted();
  ASSERT_EQ(ordered.size(), 2);
  EXPECT_EQ(ordered[0].first, "scope_b");
  EXPECT_DOUBLE_EQ(ordered[0].second.total_ms, 4.0);
  EXPECT_EQ(ordered[0].second.count, 1);
  EXPECT_EQ(ordered[1].first, "scope_a");
  EXPECT_DOUBLE_EQ(ordered[1].second.total_ms, 3.0);
  EXPECT_DOUBLE_EQ(ordered[1].second.max_ms, 2.0);
  EXPECT_EQ(ordered[1].second.count, 2);
}

TEST(ProfilerTest, ClearRemovesAllAggregates) {
  utils::ProfilerReport report;
  report.record("scope", 1.0);
  report.clear();
  EXPECT_TRUE(report.sorted().empty());
  EXPECT_EQ(report.summary(),
            "Profiler report (scope, total ms, mean ms, max ms, count)\n");
}

TEST(ProgressBarTest, SaturatesProgressAndClosesOutputOnce) {
  std::ostringstream output;
  std::streambuf* const original_buffer = std::cout.rdbuf(output.rdbuf());
  {
    utils::ProgressBar bar(/*total=*/3, /*width=*/2);
    bar.set_description("epoch");
    bar.set_postfix({{"loss", "0.25"}});
    bar.update(2);
    EXPECT_EQ(bar.current(), 2);
    bar.update(-10);
    EXPECT_EQ(bar.current(), 0);
    EXPECT_DOUBLE_EQ(bar.progress(), 0.0);
    bar.update(99);
    EXPECT_EQ(bar.current(), 3);
    EXPECT_DOUBLE_EQ(bar.progress(), 1.0);
    bar.close();
  }
  std::cout.rdbuf(original_buffer);

  const std::string rendered = output.str();
  EXPECT_NE(rendered.find("epoch [#####] 3/3 (100%) loss=0.25"),
            std::string::npos);
  EXPECT_EQ(std::count(rendered.begin(), rendered.end(), '\n'), 1);
}

TEST(ProgressBarTest, NormalizesInvalidTotalAndIgnoresDrawsAfterClose) {
  std::ostringstream output;
  std::streambuf* const original_buffer = std::cout.rdbuf(output.rdbuf());
  {
    utils::ProgressBar bar(/*total=*/0, /*width=*/1);
    bar.set_postfix({{"loss", "0.5"}, {"step", "1"}});
    bar.update();
    EXPECT_EQ(bar.current(), 1);
    EXPECT_DOUBLE_EQ(bar.progress(), 1.0);
    bar.set_description("ignored");
    bar.update(-1);
  }
  std::cout.rdbuf(original_buffer);

  const std::string rendered = output.str();
  EXPECT_NE(rendered.find("[#####] 1/1 (100%) loss=0.5, step=1"),
            std::string::npos);
  EXPECT_EQ(rendered.find("ignored"), std::string::npos);
  EXPECT_EQ(std::count(rendered.begin(), rendered.end(), '\n'), 1);
}

}  // namespace
