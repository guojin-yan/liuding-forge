// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/classification.hpp>

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
      throw jyppx::liuding::SerializationError("invalid metric test write");
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
      throw jyppx::liuding::SerializationError("invalid metric test read");
    }
    return blob->second;
  }

  /** @brief Corrupts the high byte of the final encoded int64 count. */
  void corrupt_final_count() { blobs_.at("counts").back() = std::byte{0xff}; }

 private:
  jyppx::liuding::StateManifest manifest_;  ///< Exact manifest stored with the blobs.
  std::map<std::string, std::vector<std::byte>>
      blobs_;  ///< Owned blobs indexed by stable field name.
};

torch::Tensor logits() {
  return torch::tensor({{0.9, 0.1, 0.0},
                        {0.8, 0.1, 0.1},
                        {0.1, 0.8, 0.1},
                        {0.0, 0.2, 0.9},
                        {0.1, 0.1, 0.8}},
                       torch::kFloat64);
}

torch::Tensor targets() {
  return torch::tensor({0, 0, 1, 1, 2}, torch::kInt64);
}

jyppx::liuding::training::metrics::ClassificationMetricOptions options(
    std::string metric_id,
    jyppx::liuding::training::metrics::ClassificationAverage average =
        jyppx::liuding::training::metrics::ClassificationAverage::macro) {
  return {
      .metric_id = std::move(metric_id), .num_classes = 3, .average = average};
}

TEST(ClassificationMetricTest, PublishesStableFixedTopologyDescriptors) {
  using namespace jyppx::liuding::training::metrics;
  Precision metric(options("metric.validation"));

  EXPECT_EQ(classification_average_name(ClassificationAverage::weighted),
            "weighted");
  EXPECT_EQ(zero_division_policy_name(ZeroDivisionPolicy::error), "error");
  EXPECT_EQ(classification_score_name(ClassificationScore::precision),
            "precision");
  EXPECT_EQ(metric.participant_id(), "metric.validation");
  EXPECT_EQ(metric.descriptor().type_id, "jyppx.metric.precision/v1");
  EXPECT_EQ(metric.descriptor().num_classes, 3U);
  EXPECT_EQ(metric.classes(), 3U);
  EXPECT_EQ(metric.confusion_matrix().size(), 3U);
  EXPECT_EQ(metric.describe_state().restore_level, jyppx::liuding::RestoreLevel::exact);
  EXPECT_NO_THROW(jyppx::liuding::validate_checkpoint_participant(metric));

  EXPECT_THROW(Precision({.num_classes = 0}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(Precision({.metric_id = "invalid/id", .num_classes = 2}),
               jyppx::liuding::InvalidArgument);
}

TEST(ClassificationMetricTest, ComputesExplicitAveragePolicies) {
  using namespace jyppx::liuding::training::metrics;
  Precision macro(options("precision.macro"));
  Precision micro(options("precision.micro", ClassificationAverage::micro));
  Precision weighted(
      options("precision.weighted", ClassificationAverage::weighted));
  Precision per_class(options("precision.none", ClassificationAverage::none));
  for (auto* metric : {&macro, &micro, &weighted, &per_class}) {
    metric->update(logits(), targets());
  }

  EXPECT_NEAR(macro.compute(), 5.0 / 6.0, 1e-12);
  EXPECT_NEAR(micro.compute(), 4.0 / 5.0, 1e-12);
  EXPECT_NEAR(weighted.compute(), 9.0 / 10.0, 1e-12);
  EXPECT_EQ(per_class.per_class_scores(), (std::vector<double>{1.0, 1.0, 0.5}));
  EXPECT_THROW(per_class.compute(), jyppx::liuding::InvalidArgument);

  Recall recall(options("recall.macro"));
  F1Score f1(options("f1.macro"));
  recall.update(logits(), targets());
  f1.update(logits(), targets());
  EXPECT_NEAR(recall.compute(), 5.0 / 6.0, 1e-12);
  EXPECT_NEAR(f1.compute(), 7.0 / 9.0, 1e-12);
}

TEST(ClassificationMetricTest, IsInvariantToBatchPartitioning) {
  using namespace jyppx::liuding::training::metrics;
  F1Score single(options("f1.partition"));
  F1Score split(options("f1.partition"));
  single.update(logits(), targets());
  split.update(logits().narrow(0, 0, 2), targets().narrow(0, 0, 2));
  split.update(logits().narrow(0, 2, 3), targets().narrow(0, 2, 3));

  EXPECT_EQ(split.confusion_matrix(), single.confusion_matrix());
  EXPECT_DOUBLE_EQ(split.compute(), single.compute());
}

TEST(ClassificationMetricTest, MergeIsAssociativeAndCommutativeForCounts) {
  using namespace jyppx::liuding::training::metrics;
  Recall single(options("recall.merge"));
  Recall first(options("recall.merge"));
  Recall second(options("recall.merge"));
  single.update(logits(), targets());
  first.update(logits().narrow(0, 0, 2), targets().narrow(0, 0, 2));
  second.update(logits().narrow(0, 2, 3), targets().narrow(0, 2, 3));

  Recall forward(options("recall.merge"));
  forward.merge(first);
  forward.merge(second);
  Recall reverse(options("recall.merge"));
  reverse.merge(second);
  reverse.merge(first);

  EXPECT_EQ(forward.confusion_matrix(), single.confusion_matrix());
  EXPECT_EQ(reverse.confusion_matrix(), single.confusion_matrix());
  EXPECT_DOUBLE_EQ(forward.compute(), single.compute());
  EXPECT_DOUBLE_EQ(reverse.compute(), single.compute());

  Recall incompatible(options("recall.other"));
  const auto before = forward.confusion_matrix();
  EXPECT_THROW(forward.merge(incompatible), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(forward.confusion_matrix(), before);
}

TEST(ClassificationMetricTest, EmptyAndAllIgnoredStateIsNotReady) {
  using namespace jyppx::liuding::training::metrics;
  Precision empty({.metric_id = "precision.empty", .num_classes = 2});
  EXPECT_THROW(empty.compute(), jyppx::liuding::NotReady);
  EXPECT_THROW(empty.per_class_scores(), jyppx::liuding::NotReady);

  Precision ignored(
      {.metric_id = "precision.ignored", .num_classes = 2, .ignore_index = -1});
  ignored.update(torch::tensor({{1.0, 0.0}, {0.0, 1.0}}),
                 torch::tensor({-1, -1}, torch::kInt64));
  EXPECT_EQ(ignored.total_observations(), 0);
  EXPECT_THROW(ignored.compute(), jyppx::liuding::NotReady);
}

TEST(ClassificationMetricTest, AppliesExplicitZeroDivisionPolicy) {
  using namespace jyppx::liuding::training::metrics;
  const auto one_logit = torch::tensor({{1.0, 0.0}});
  const auto one_target = torch::tensor({0}, torch::kInt64);
  Precision zero({.metric_id = "precision.zero", .num_classes = 2});
  Precision one({.metric_id = "precision.one",
                 .num_classes = 2,
                 .zero_division = ZeroDivisionPolicy::one});
  Precision error({.metric_id = "precision.error",
                   .num_classes = 2,
                   .zero_division = ZeroDivisionPolicy::error});
  zero.update(one_logit, one_target);
  one.update(one_logit, one_target);
  error.update(one_logit, one_target);

  EXPECT_DOUBLE_EQ(zero.compute(), 0.5);
  EXPECT_DOUBLE_EQ(one.compute(), 1.0);
  EXPECT_THROW(error.compute(), jyppx::liuding::NumericalError);
}

TEST(ClassificationMetricTest, RejectsInvalidInputWithoutMutation) {
  using namespace jyppx::liuding::training::metrics;
  Precision metric(options("precision.atomic"));
  metric.update(logits().narrow(0, 0, 1), targets().narrow(0, 0, 1));
  const auto before = metric.confusion_matrix();

  EXPECT_THROW(metric.update(logits(), targets().to(torch::kFloat32)),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(metric.update(torch::ones({5, 2}), targets()),
               jyppx::liuding::InvalidArgument);
  EXPECT_THROW(metric.update(logits(), torch::tensor({0, 0, 1, 1, 3})),
               jyppx::liuding::InvalidArgument);
  auto non_finite = logits().clone();
  non_finite.index_put_({0, 0}, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(metric.update(non_finite, targets()), jyppx::liuding::NumericalError);
  EXPECT_EQ(metric.confusion_matrix(), before);
}

TEST(ClassificationMetricTest, RestoresExactTypedStateAtomically) {
  using namespace jyppx::liuding::training::metrics;
  F1Score source(options("f1.checkpoint"));
  source.update(logits(), targets());
  MemoryStateStore store(source.describe_state());
  source.save_state(store);

  F1Score restored(options("f1.checkpoint"));
  restored.update(logits().narrow(0, 0, 1), targets().narrow(0, 0, 1));
  restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only);
  EXPECT_EQ(restored.confusion_matrix(), source.confusion_matrix());
  EXPECT_DOUBLE_EQ(restored.compute(), source.compute());

  F1Score incompatible({.metric_id = "f1.checkpoint", .num_classes = 2});
  const auto incompatible_before = incompatible.confusion_matrix();
  EXPECT_THROW(incompatible.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::CompatibilityError);
  EXPECT_EQ(incompatible.confusion_matrix(), incompatible_before);

  const auto restored_before = restored.confusion_matrix();
  store.corrupt_final_count();
  EXPECT_THROW(restored.load_state(store, jyppx::liuding::RestorePolicy::exact_only),
               jyppx::liuding::SerializationError);
  EXPECT_EQ(restored.confusion_matrix(), restored_before);
}

TEST(ClassificationMetricTest, ResetPreservesTopologyAndClearsReadiness) {
  using namespace jyppx::liuding::training::metrics;
  F1Score metric(options("f1.reset"));
  metric.update(logits(), targets());
  metric.reset();

  EXPECT_EQ(metric.classes(), 3U);
  EXPECT_EQ(metric.total_observations(), 0);
  EXPECT_EQ(metric.confusion_matrix(), (std::vector<std::vector<std::int64_t>>{
                                           {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
  EXPECT_THROW(metric.compute(), jyppx::liuding::NotReady);
}

}  // namespace
