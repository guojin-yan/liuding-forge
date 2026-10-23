// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/ranking.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

class MemoryStateStore final : public jyppx::liuding::StateWriter,
                               public jyppx::liuding::StateReader {
 public:
  /** @brief Creates an empty store governed by one participant manifest. */
  explicit MemoryStateStore(jyppx::liuding::StateManifest manifest)
      : manifest_(std::move(manifest)) {}

  /** @brief Copies one declared, bounded state blob into memory. */
  void write_blob(std::string_view name,
                  std::span<const std::byte> data) override {
    const auto field =
        std::find_if(manifest_.fields.begin(), manifest_.fields.end(),
                     [name](const jyppx::liuding::StateFieldDescriptor& descriptor) {
                       return descriptor.name == name;
                     });
    if (field == manifest_.fields.end() || data.size() > field->max_bytes ||
        !blobs_
             .emplace(std::string(name),
                      std::vector<std::byte>(data.begin(), data.end()))
             .second) {
      throw jyppx::liuding::SerializationError("invalid ranking metric test write");
    }
  }

  /** @brief Returns the stored participant manifest. */
  const jyppx::liuding::StateManifest& manifest() const noexcept override {
    return manifest_;
  }

  /** @brief Reports whether a named state blob is available. */
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }

  /** @brief Returns a named blob after enforcing the caller's byte limit. */
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max_bytes) const override {
    const auto blob = blobs_.find(std::string(name));
    if (blob == blobs_.end() || blob->second.size() > max_bytes) {
      throw jyppx::liuding::SerializationError("invalid ranking metric test read");
    }
    return blob->second;
  }

  /** @brief Replaces the final encoded binary label with an invalid value. */
  void corrupt_final_label() {
    blobs_.at("observations").back() = std::byte{2};
  }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Exact manifest stored with the blobs.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Owned blobs indexed by stable field name.
};

jyppx::liuding::training::metrics::ExactRankingResourceOptions limits(
    std::size_t max_observations) {
  return {.max_classes = 32,
          .max_observations = max_observations,
          .max_topology_bytes = 64ULL * 1024ULL,
          .max_observation_bytes =
              1024ULL + 9ULL * static_cast<std::uint64_t>(max_observations)};
}

torch::Tensor scores() {
  return torch::tensor({0.9, 0.8, 0.8, 0.1}, torch::kFloat64);
}

torch::Tensor labels() { return torch::tensor({1, 0, 1, 0}, torch::kInt64); }

double pairwise_auroc_reference(const std::vector<double>& input_scores,
                                const std::vector<std::int64_t>& input_labels) {
  long double wins = 0.0L;
  std::size_t positives = 0;
  std::size_t negatives = 0;
  for (const auto label : input_labels) {
    positives += label == 1 ? 1U : 0U;
    negatives += label == 0 ? 1U : 0U;
  }
  for (std::size_t left = 0; left < input_scores.size(); ++left) {
    if (input_labels[left] != 1) {
      continue;
    }
    for (std::size_t right = 0; right < input_scores.size(); ++right) {
      if (input_labels[right] != 0) {
        continue;
      }
      if (input_scores[left] > input_scores[right]) {
        wins += 1.0L;
      } else if (input_scores[left] == input_scores[right]) {
        wins += 0.5L;
      }
    }
  }
  return static_cast<double>(wins /
                             (static_cast<long double>(positives) * negatives));
}

TEST(RankingMetricTest, PublishesStableDescriptorAndGroupedThresholdCurve) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAUROC metric({.metric_id = "ranking.validation"}, limits(8));
  metric.update(scores(), labels());

  EXPECT_EQ(binary_ranking_score_name(BinaryRankingScore::auroc), "auroc");
  EXPECT_EQ(undefined_ranking_policy_name(UndefinedRankingPolicy::error),
            "error");
  EXPECT_EQ(metric.participant_id(), "ranking.validation");
  EXPECT_EQ(metric.descriptor().type_id, "jyppx.metric.binary-auroc/v1");
  EXPECT_EQ(metric.descriptor().score, BinaryRankingScore::auroc);
  EXPECT_EQ(metric.observations(), 4U);
  EXPECT_EQ(metric.positive_observations(), 2U);
  EXPECT_EQ(metric.negative_observations(), 2U);
  EXPECT_EQ(metric.describe_state().restore_level, jyppx::liuding::RestoreLevel::exact);
  EXPECT_NO_THROW(jyppx::liuding::validate_checkpoint_participant(metric));

  EXPECT_EQ(metric.threshold_curve(),
            (std::vector<BinaryRankingCurvePoint>{{.threshold = 0.9,
                                                   .true_positives = 1,
                                                   .false_positives = 0,
                                                   .true_negatives = 2,
                                                   .false_negatives = 1},
                                                  {.threshold = 0.8,
                                                   .true_positives = 2,
                                                   .false_positives = 1,
                                                   .true_negatives = 1,
                                                   .false_negatives = 0},
                                                  {.threshold = 0.1,
                                                   .true_positives = 2,
                                                   .false_positives = 2,
                                                   .true_negatives = 0,
                                                   .false_negatives = 0}}));
  EXPECT_THROW(BinaryAUROC({.metric_id = "invalid/id"}),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(BinaryAUROC({}, {.max_observations = 2,
                                .max_topology_bytes = 1024,
                                .max_observation_bytes = 16}),
               jyppx::liuding::InvalidArgument);
}

TEST(RankingMetricTest, ComputesTieAwareAuRocAndAveragePrecision) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAUROC auroc({.metric_id = "ranking.auroc"});
  BinaryAveragePrecision average_precision(
      {.metric_id = "ranking.average_precision"});
  auroc.update(scores(), labels());
  average_precision.update(scores(), labels());

  EXPECT_NEAR(auroc.compute(), 0.875, 1e-12);
  EXPECT_NEAR(average_precision.compute(), 5.0 / 6.0, 1e-12);

  BinaryAUROC reversed({.metric_id = "ranking.reversed"});
  reversed.update(scores().flip({0}), labels().flip({0}));
  EXPECT_NEAR(reversed.compute(), auroc.compute(), 1e-12);
}

TEST(RankingMetricTest, MatchesIndependentPairwiseDatasetReference) {
  using namespace jyppx::liuding::training::metrics;
  const std::vector<double> input_scores{0.95, 0.8,  0.8, 0.72, 0.51, 0.51,
                                         0.4,  0.33, 0.2, 0.2,  0.05};
  const std::vector<std::int64_t> input_labels{1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0};
  BinaryAUROC metric({.metric_id = "ranking.reference.differential"});
  metric.update(torch::tensor(input_scores, torch::kFloat64),
                torch::tensor(input_labels, torch::kInt64));

  EXPECT_NEAR(metric.compute(),
              pairwise_auroc_reference(input_scores, input_labels), 1e-12);
}

TEST(RankingMetricTest, ComputesCanonicalOrderingBoundaries) {
  using namespace jyppx::liuding::training::metrics;
  const auto target = torch::tensor({1, 1, 0, 0}, torch::kInt64);
  BinaryAUROC perfect({.metric_id = "ranking.perfect"});
  BinaryAUROC inverse({.metric_id = "ranking.inverse"});
  BinaryAUROC tied({.metric_id = "ranking.tied"});
  BinaryAveragePrecision tied_ap({.metric_id = "ranking.tied.ap"});
  perfect.update(torch::tensor({0.9, 0.8, 0.2, 0.1}), target);
  inverse.update(torch::tensor({0.2, 0.1, 0.9, 0.8}), target);
  tied.update(torch::ones({4}), target);
  tied_ap.update(torch::ones({4}), target);

  EXPECT_DOUBLE_EQ(perfect.compute(), 1.0);
  EXPECT_DOUBLE_EQ(inverse.compute(), 0.0);
  EXPECT_DOUBLE_EQ(tied.compute(), 0.5);
  EXPECT_DOUBLE_EQ(tied_ap.compute(), 0.5);
}

TEST(RankingMetricTest, IsInvariantToBatchPartitionAndMergeOrder) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAveragePrecision single({.metric_id = "ranking.merge"});
  BinaryAveragePrecision first({.metric_id = "ranking.merge"});
  BinaryAveragePrecision second({.metric_id = "ranking.merge"});
  single.update(scores(), labels());
  first.update(scores().narrow(0, 0, 2), labels().narrow(0, 0, 2));
  second.update(scores().narrow(0, 2, 2), labels().narrow(0, 2, 2));

  BinaryAveragePrecision forward({.metric_id = "ranking.merge"});
  forward.merge(first);
  forward.merge(second);
  BinaryAveragePrecision reverse({.metric_id = "ranking.merge"});
  reverse.merge(second);
  reverse.merge(first);

  EXPECT_EQ(forward.threshold_curve(), single.threshold_curve());
  EXPECT_EQ(reverse.threshold_curve(), single.threshold_curve());
  EXPECT_DOUBLE_EQ(forward.compute(), single.compute());
  EXPECT_DOUBLE_EQ(reverse.compute(), single.compute());

  BinaryAveragePrecision incompatible({.metric_id = "ranking.other"});
  const auto before = forward.threshold_curve();
  EXPECT_THROW(forward.merge(incompatible), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(forward.threshold_curve(), before);

  BinaryAveragePrecision bounded({.metric_id = "ranking.merge"}, limits(3));
  bounded.merge(first);
  const auto bounded_before = bounded.threshold_curve();
  EXPECT_THROW(bounded.merge(second), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(bounded.threshold_curve(), bounded_before);
}

TEST(RankingMetricTest, FreezesEmptyIgnoreAndUndefinedClassSemantics) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAUROC empty({.metric_id = "ranking.empty"});
  EXPECT_THROW(empty.compute(), jyppx::liuding::NotReady);
  EXPECT_THROW(empty.threshold_curve(), jyppx::liuding::NotReady);

  BinaryAUROC ignored({.metric_id = "ranking.ignored", .ignore_index = -1});
  ignored.update(torch::tensor({0.1, 0.2}),
                 torch::tensor({-1, -1}, torch::kInt64));
  EXPECT_EQ(ignored.observations(), 0U);
  EXPECT_THROW(ignored.compute(), jyppx::liuding::NotReady);

  const auto positives = torch::tensor({1, 1}, torch::kInt64);
  BinaryAUROC undefined_error({.metric_id = "ranking.undefined.error"});
  BinaryAUROC undefined_zero(
      {.metric_id = "ranking.undefined.zero",
       .undefined_policy = UndefinedRankingPolicy::zero});
  BinaryAUROC undefined_one({.metric_id = "ranking.undefined.one",
                             .undefined_policy = UndefinedRankingPolicy::one});
  for (auto* metric : {&undefined_error, &undefined_zero, &undefined_one}) {
    metric->update(torch::tensor({0.2, 0.8}), positives);
  }
  EXPECT_THROW(undefined_error.compute(), jyppx::liuding::NumericalError);
  EXPECT_DOUBLE_EQ(undefined_zero.compute(), 0.0);
  EXPECT_DOUBLE_EQ(undefined_one.compute(), 1.0);

  BinaryAveragePrecision all_positive({.metric_id = "ranking.ap.all_positive"});
  all_positive.update(torch::tensor({0.2, 0.8}), positives);
  EXPECT_DOUBLE_EQ(all_positive.compute(), 1.0);

  const auto negatives = torch::tensor({0, 0}, torch::kInt64);
  BinaryAveragePrecision no_positive({.metric_id = "ranking.ap.no_positive"});
  BinaryAveragePrecision no_positive_zero(
      {.metric_id = "ranking.ap.no_positive.zero",
       .undefined_policy = UndefinedRankingPolicy::zero});
  no_positive.update(torch::tensor({0.2, 0.8}), negatives);
  no_positive_zero.update(torch::tensor({0.2, 0.8}), negatives);
  EXPECT_THROW(no_positive.compute(), jyppx::liuding::NumericalError);
  EXPECT_DOUBLE_EQ(no_positive_zero.compute(), 0.0);
}

TEST(RankingMetricTest, RejectsInvalidOrOversizedUpdateWithoutMutation) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAUROC metric({.metric_id = "ranking.atomic"}, limits(3));
  metric.update(torch::tensor({0.9}), torch::tensor({1}, torch::kInt64));
  const auto before = metric.threshold_curve();

  EXPECT_THROW(
      metric.update(torch::tensor({0.8, 0.7}), torch::tensor({0.0, 1.0})),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      metric.update(torch::ones({1, 2}), torch::tensor({0, 1}, torch::kInt64)),
      jyppx::liuding::InvalidArgument);
  EXPECT_THROW(metric.update(torch::tensor({0.8, 0.7}),
                             torch::tensor({0, 2}, torch::kInt64)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      metric.update(torch::tensor({std::numeric_limits<double>::infinity()}),
                    torch::tensor({0}, torch::kInt64)),
      jyppx::liuding::NumericalError);
  EXPECT_THROW(metric.update(torch::tensor({0.8, 0.7, 0.6}),
                             torch::tensor({0, 1, 0}, torch::kInt64)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(metric.threshold_curve(), before);
}

TEST(RankingMetricTest, RestoresExactTypedStateAtomically) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAveragePrecision source({.metric_id = "ranking.checkpoint"}, limits(8));
  source.update(scores(), labels());
  auto manifest = source.describe_state();
  std::sort(manifest.fields.begin(), manifest.fields.end(),
            [](const auto& left, const auto& right) {
              return left.name < right.name;
            });
  MemoryStateStore store(std::move(manifest));
  source.save_state(store);

  BinaryAveragePrecision restored({.metric_id = "ranking.checkpoint"},
                                  limits(8));
  restored.update(torch::tensor({0.5}), torch::tensor({1}, torch::kInt64));
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.threshold_curve(), source.threshold_curve());
  EXPECT_DOUBLE_EQ(restored.compute(), source.compute());

  BinaryAveragePrecision incompatible(
      {.metric_id = "ranking.checkpoint",
       .undefined_policy = UndefinedRankingPolicy::zero},
      limits(8));
  EXPECT_THROW(incompatible.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(incompatible.observations(), 0U);

  const auto restored_before = restored.threshold_curve();
  store.corrupt_final_label();
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(restored.threshold_curve(), restored_before);
}

TEST(RankingMetricTest, ComputesMulticlassAggregationsFromOneVsRestState) {
  using namespace jyppx::liuding::training::metrics;
  const auto predictions = torch::tensor({{0.9, 0.2, 0.1},
                                          {0.8, 0.7, 0.3},
                                          {0.4, 0.6, 0.8},
                                          {0.3, 0.5, 0.7},
                                          {0.2, 0.4, 0.6}},
                                         torch::kFloat64);
  const auto targets = torch::tensor({0, 1, 2, 0, 2}, torch::kInt64);
  const ClasswiseRankingMetricOptions base{.metric_id = "ranking.multiclass",
                                           .num_classes = 3,
                                           .average = RankingAverage::macro};
  MulticlassAUROC macro(base);
  macro.update(predictions, targets);

  std::vector<double> expected;
  for (std::int64_t class_index = 0; class_index < 3; ++class_index) {
    BinaryAUROC binary({.metric_id = "ranking.reference"});
    binary.update(predictions.select(1, class_index),
                  targets.eq(class_index).to(torch::kInt64));
    expected.push_back(binary.compute());
  }
  EXPECT_EQ(ranking_average_name(RankingAverage::macro), "macro");
  EXPECT_EQ(classwise_ranking_task_name(ClasswiseRankingTask::multiclass),
            "multiclass");
  EXPECT_EQ(macro.descriptor().type_id, "jyppx.metric.multiclass-auroc/v1");
  EXPECT_EQ(macro.descriptor().num_classes, 3U);
  EXPECT_EQ(macro.observations(), 15U);
  EXPECT_EQ(macro.observations(0), 5U);
  EXPECT_EQ(macro.positive_observations(0), 2U);
  EXPECT_EQ(macro.per_class_scores(), expected);
  EXPECT_NEAR(macro.compute(), (expected[0] + expected[1] + expected[2]) / 3.0,
              1e-12);

  auto weighted_options = base;
  weighted_options.metric_id = "ranking.multiclass.weighted";
  weighted_options.average = RankingAverage::weighted;
  MulticlassAUROC weighted(weighted_options);
  weighted.update(predictions, targets);
  EXPECT_NEAR(weighted.compute(),
              (2.0 * expected[0] + expected[1] + 2.0 * expected[2]) / 5.0,
              1e-12);

  auto micro_options = base;
  micro_options.metric_id = "ranking.multiclass.micro";
  micro_options.average = RankingAverage::micro;
  MulticlassAUROC micro(micro_options);
  micro.update(predictions, targets);
  BinaryAUROC pooled({.metric_id = "ranking.pooled"});
  pooled.update(predictions.reshape({-1}),
                torch::one_hot(targets, 3).reshape({-1}));
  EXPECT_DOUBLE_EQ(micro.compute(), pooled.compute());
  EXPECT_EQ(micro.micro_threshold_curve(), pooled.threshold_curve());

  auto none_options = base;
  none_options.metric_id = "ranking.multiclass.none";
  none_options.average = RankingAverage::none;
  MulticlassAUROC none(none_options);
  none.update(predictions, targets);
  EXPECT_THROW(none.compute(), jyppx::liuding::InvalidArgument);
  EXPECT_EQ(none.per_class_scores(), expected);
}

TEST(RankingMetricTest, FreezesMultilabelIgnoreAndUndefinedClassSemantics) {
  using namespace jyppx::liuding::training::metrics;
  const auto predictions =
      torch::tensor({{0.9, 0.1}, {0.8, 0.7}, {0.2, 0.6}}, torch::kFloat64);
  const auto targets = torch::tensor({{1, 0}, {1, 1}, {0, -1}}, torch::kInt64);
  MultilabelAveragePrecision metric(
      {.metric_id = "ranking.multilabel",
       .num_classes = 2,
       .average = RankingAverage::macro,
       .undefined_policy = UndefinedRankingPolicy::error,
       .ignore_index = -1});
  metric.update(predictions, targets);

  EXPECT_EQ(metric.descriptor().type_id,
            "jyppx.metric.multilabel-average-precision/v1");
  EXPECT_EQ(metric.observations(), 5U);
  EXPECT_EQ(metric.observations(0), 3U);
  EXPECT_EQ(metric.observations(1), 2U);
  EXPECT_EQ(metric.per_class_scores(), (std::vector<double>{1.0, 1.0}));
  EXPECT_DOUBLE_EQ(metric.compute(), 1.0);

  const auto sparse_targets =
      torch::tensor({{1, -1}, {0, -1}, {1, -1}}, torch::kInt64);
  MultilabelAUROC zero_policy({.metric_id = "ranking.multilabel.zero",
                               .num_classes = 2,
                               .average = RankingAverage::macro,
                               .undefined_policy = UndefinedRankingPolicy::zero,
                               .ignore_index = -1});
  zero_policy.update(predictions, sparse_targets);
  EXPECT_EQ(zero_policy.per_class_scores(), (std::vector<double>{0.5, 0.0}));
  EXPECT_DOUBLE_EQ(zero_policy.compute(), 0.25);
  EXPECT_THROW(zero_policy.threshold_curve(1), jyppx::liuding::NotReady);
  EXPECT_THROW(zero_policy.threshold_curve(2), jyppx::liuding::InvalidArgument);

  MultilabelAUROC error_policy(
      {.metric_id = "ranking.multilabel.error",
       .num_classes = 2,
       .average = RankingAverage::macro,
       .undefined_policy = UndefinedRankingPolicy::error,
       .ignore_index = -1});
  error_policy.update(predictions, sparse_targets);
  EXPECT_THROW(error_policy.compute(), jyppx::liuding::NumericalError);

  MultilabelAUROC empty({.metric_id = "ranking.multilabel.empty",
                         .num_classes = 2,
                         .average = RankingAverage::macro,
                         .undefined_policy = UndefinedRankingPolicy::zero,
                         .ignore_index = -1});
  empty.update(predictions, torch::full({3, 2}, -1, torch::kInt64));
  EXPECT_EQ(empty.observations(), 0U);
  EXPECT_THROW(empty.compute(), jyppx::liuding::NotReady);
}

TEST(RankingMetricTest, MergesAndRestoresClasswiseExactStateAtomically) {
  using namespace jyppx::liuding::training::metrics;
  const auto predictions = torch::tensor(
      {{0.9, 0.2, 0.1}, {0.8, 0.7, 0.3}, {0.4, 0.6, 0.8}}, torch::kFloat64);
  const auto targets = torch::tensor({0, 1, 2}, torch::kInt64);
  const ClasswiseRankingMetricOptions options{
      .metric_id = "ranking.classwise.state",
      .num_classes = 3,
      .average = RankingAverage::micro,
      .undefined_policy = UndefinedRankingPolicy::error};
  MulticlassAveragePrecision full(options, limits(12));
  full.update(predictions, targets);
  MulticlassAveragePrecision first(options, limits(12));
  MulticlassAveragePrecision second(options, limits(12));
  first.update(predictions.narrow(0, 0, 1), targets.narrow(0, 0, 1));
  second.update(predictions.narrow(0, 1, 2), targets.narrow(0, 1, 2));
  first.merge(second);
  EXPECT_EQ(first.per_class_scores(), full.per_class_scores());
  EXPECT_DOUBLE_EQ(first.compute(), full.compute());

  MemoryStateStore store(full.describe_state());
  full.save_state(store);
  MulticlassAveragePrecision restored(options, limits(12));
  restored.update(predictions.narrow(0, 0, 1), targets.narrow(0, 0, 1));
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.per_class_scores(), full.per_class_scores());
  EXPECT_DOUBLE_EQ(restored.compute(), full.compute());

  auto incompatible_options = options;
  incompatible_options.average = RankingAverage::macro;
  MulticlassAveragePrecision incompatible(incompatible_options, limits(12));
  EXPECT_THROW(incompatible.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(incompatible.observations(), 0U);

  const auto before = restored.per_class_scores();
  store.corrupt_final_label();
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(restored.per_class_scores(), before);

  MulticlassAveragePrecision bounded(options, limits(8));
  bounded.update(predictions.narrow(0, 0, 1), targets.narrow(0, 0, 1));
  EXPECT_THROW(bounded.merge(second), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(bounded.observations(), 3U);
}

TEST(RankingMetricTest, RejectsInvalidClasswiseUpdatesWithoutMutation) {
  using namespace jyppx::liuding::training::metrics;
  MulticlassAUROC metric({.metric_id = "ranking.classwise.atomic",
                          .num_classes = 2,
                          .average = RankingAverage::macro},
                         limits(5));
  metric.update(torch::tensor({{0.9, 0.1}, {0.2, 0.8}}),
                torch::tensor({0, 1}, torch::kInt64));
  const auto before = metric.per_class_scores();

  EXPECT_THROW(metric.update(torch::tensor({{0.7, 0.3}}),
                             torch::tensor({2}, torch::kInt64)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(metric.update(torch::tensor({0.7, 0.3}),
                             torch::tensor({0}, torch::kInt64)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      metric.update(
          torch::tensor({{std::numeric_limits<double>::infinity(), 0.3}}),
          torch::tensor({0}, torch::kInt64)),
      jyppx::liuding::NumericalError);
  EXPECT_THROW(metric.update(torch::tensor({{0.7, 0.3}}),
                             torch::tensor({0}, torch::kInt64)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(metric.observations(), 4U);
  EXPECT_EQ(metric.per_class_scores(), before);

  EXPECT_THROW(MulticlassAUROC({.num_classes = 1}), jyppx::liuding::InvalidArgument);
  MulticlassAUROC pair_limited({.num_classes = 3}, limits(2));
  EXPECT_THROW(pair_limited.update(torch::tensor({{0.7, 0.2, 0.1}}),
                                   torch::tensor({0}, torch::kInt64)),
               jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(pair_limited.observations(), 0U);
}

TEST(RankingMetricTest, ResetPreservesDescriptorAndClearsReadiness) {
  using namespace jyppx::liuding::training::metrics;
  BinaryAUROC metric({.metric_id = "ranking.reset"});
  metric.update(scores(), labels());
  metric.reset();

  EXPECT_EQ(metric.participant_id(), "ranking.reset");
  EXPECT_EQ(metric.observations(), 0U);
  EXPECT_THROW(metric.compute(), jyppx::liuding::NotReady);
}

TEST(RankingMetricTest, AcceptsCudaScoresAndLabelsWhenAvailable) {
  using namespace jyppx::liuding::training::metrics;
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  BinaryAUROC metric({.metric_id = "ranking.cuda"});
  metric.update(scores().to(torch::kCUDA), labels().to(torch::kCUDA));

  EXPECT_NEAR(metric.compute(), 0.875, 1e-12);
  EXPECT_EQ(metric.observations(), 4U);

  MultilabelAUROC classwise({.metric_id = "ranking.cuda.multilabel",
                             .num_classes = 2,
                             .average = RankingAverage::macro});
  classwise.update(
      torch::tensor({{0.9, 0.1}, {0.8, 0.7}, {0.2, 0.6}}).to(torch::kCUDA),
      torch::tensor({{1, 0}, {1, 1}, {0, 0}}, torch::kInt64).to(torch::kCUDA));
  EXPECT_EQ(classwise.observations(), 6U);
  EXPECT_EQ(classwise.per_class_scores(), (std::vector<double>{1.0, 1.0}));
}

}  // namespace
