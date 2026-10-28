// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/segmentation.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <span>
#include <utility>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training::metrics;

class MemoryState final : public StateReader, public StateWriter {
 public:
  explicit MemoryState(const CheckpointParticipant& metric)
      : state(metric.describe_state()) {}
  const StateManifest& manifest() const noexcept override { return state; }
  bool contains(std::string_view name) const override {
    return blobs.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t max) const override {
    const auto& b = blobs.at(std::string(name));
    if (b.size() > max)
      throw ResourceExhausted("test state exceeds read limit");
    return b;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs.emplace(std::string(name),
                  std::vector<std::byte>(bytes.begin(), bytes.end()));
  }
  void set_count(std::size_t slot, std::int64_t value) {
    for (unsigned i = 0; i < 8; ++i)
      blobs.at("counts")[8 * slot + i] = static_cast<std::byte>(
          (static_cast<std::uint64_t>(value) >> (8U * i)) & 255U);
  }
  StateManifest state;
  std::map<std::string, std::vector<std::byte>> blobs;
};

Tensor prediction() {
  return torch::tensor({0, 1, 2, 0, 1, 1, 2, 0}, torch::kInt64)
      .reshape({2, 2, 2});
}
Tensor target() {
  return torch::tensor({0, 1, 1, 1, -1, 2, 2, 0}, torch::kInt64)
      .reshape({2, 2, 2});
}
SegmentationMetricOptions options(
    SegmentationScore score = SegmentationScore::iou) {
  return {.metric_id = "metric.mask",
          .num_classes = 4,
          .score = score,
          .input = SegmentationInput::class_indices,
          .ignore_index = -1};
}
std::vector<std::int64_t> counts(const SemanticSegmentationMetric& metric) {
  std::vector<std::int64_t> result{metric.total_pixels()};
  for (const auto& c : metric.class_counts()) {
    result.push_back(c.intersection);
    result.push_back(c.target);
    result.push_back(c.predicted);
  }
  return result;
}

void manual(bool cuda) {
  const auto device = torch::Device(cuda ? "cuda:0" : "cpu");
  auto p = prediction().to(device), t = target().to(device);
  SemanticSegmentationMetric iou(options(), {.chunk_pixels = 1});
  SemanticSegmentationMetric dice(options(SegmentationScore::dice),
                                  {.chunk_pixels = 3});
  iou.update(p, t);
  dice.update(p, t);
  EXPECT_EQ(iou.total_pixels(), 7);
  // True support [2,3,2,0], predicted support [3,2,2,0], TP [2,1,1,0].
  const auto values = iou.per_class_scores();
  EXPECT_DOUBLE_EQ(*values[0], 2. / 3);
  EXPECT_DOUBLE_EQ(*values[1], 1. / 4);
  EXPECT_DOUBLE_EQ(*values[2], 1. / 3);
  EXPECT_FALSE(values[3]);
  EXPECT_NEAR(iou.compute(), 5. / 12, 1e-15);
  EXPECT_NEAR(dice.compute(), 17. / 30, 1e-15);
  EXPECT_EQ(counts(iou),
            (std::vector<std::int64_t>{7, 2, 2, 3, 1, 3, 2, 1, 2, 2, 0, 0, 0}));
}
TEST(SemanticSegmentation, ManualDatasetIoUDiceCPU) { manual(false); }
TEST(SemanticSegmentation, ManualDatasetIoUDiceCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  manual(true);
}

TEST(SemanticSegmentation, BackgroundExclusionRetainsForegroundMistakes) {
  auto o = options();
  o.include_background = false;
  SemanticSegmentationMetric metric(o);
  metric.update(prediction(), target());
  EXPECT_FALSE(metric.per_class_scores()[0]);
  EXPECT_NEAR(metric.compute(), 7. / 24, 1e-15);
  o.average = ClassificationAverage::micro;
  SemanticSegmentationMetric micro(o);
  micro.update(prediction(), target());
  EXPECT_DOUBLE_EQ(micro.compute(), 2. / 7);
}
TEST(SemanticSegmentation, EmptyIgnoreAbsentAndNoIncludedDenominator) {
  SemanticSegmentationMetric empty(options());
  EXPECT_THROW(empty.compute(), NotReady);
  empty.update(torch::empty({0, 2, 3}, torch::kInt64),
               torch::empty({0, 2, 3}, torch::kInt64));
  empty.update(torch::full({1, 2, 3}, 999, torch::kInt64),
               torch::full({1, 2, 3}, -1, torch::kInt64));
  EXPECT_EQ(empty.total_pixels(), 0);
  EXPECT_THROW(empty.per_class_scores(), NotReady);
  for (auto policy : {AbsentClassPolicy::exclude, AbsentClassPolicy::zero,
                      AbsentClassPolicy::one, AbsentClassPolicy::error}) {
    auto o = options();
    o.absent = policy;
    SemanticSegmentationMetric metric(o);
    metric.update(prediction(), target());
    if (policy == AbsentClassPolicy::error)
      EXPECT_THROW(metric.compute(), NumericalError);
    else
      EXPECT_NEAR(metric.compute(),
                  policy == AbsentClassPolicy::exclude ? 5. / 12
                  : policy == AbsentClassPolicy::zero  ? 5. / 16
                                                       : 9. / 16,
                  1e-15);
  }
  auto o = options();
  o.include_background = false;
  SemanticSegmentationMetric background(o);
  background.update(torch::zeros({1, 2, 2}, torch::kInt64),
                    torch::zeros({1, 2, 2}, torch::kInt64));
  EXPECT_THROW(background.compute(), NotReady);
  o.average = ClassificationAverage::none;
  SemanticSegmentationMetric none(o);
  none.update(prediction(), target());
  EXPECT_THROW(none.compute(), InvalidArgument);
  EXPECT_EQ(none.per_class_scores().size(), 4);
}

void logits_test(bool cuda) {
  const auto device = torch::Device(cuda ? "cuda:0" : "cpu");
  auto p = prediction();
  auto t = target().transpose(1, 2).to(device);
  auto logits = torch::nn::functional::one_hot(p, 4)
                    .permute({0, 3, 1, 2})
                    .to(torch::kFloat)
                    .transpose(2, 3)
                    .to(device);
  logits.set_requires_grad(true);
  auto o = options();
  o.input = SegmentationInput::logits;
  SemanticSegmentationMetric actual(o, {.chunk_pixels = 1});
  actual.update(logits, t);
  SemanticSegmentationMetric expected(options());
  expected.update(p, target());
  EXPECT_EQ(counts(actual), counts(expected));
  EXPECT_FALSE(logits.grad().defined());
  auto tie = torch::zeros({1, 4, 1, 3}, torch::TensorOptions().device(device));
  actual.reset();
  actual.update(
      tie,
      torch::zeros({1, 1, 3},
                   torch::TensorOptions().dtype(torch::kInt64).device(device)));
  EXPECT_EQ(actual.class_counts()[0].intersection, 3);
  EXPECT_DOUBLE_EQ(actual.compute(), 1.);
}
TEST(SemanticSegmentation, NoncontiguousLogitsTiesAndNoGradCPU) {
  logits_test(false);
}
TEST(SemanticSegmentation, NoncontiguousLogitsTiesAndNoGradCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  logits_test(true);
}

TEST(SemanticSegmentation, PartitionMergeOrderAndStateContinuationAreExact) {
  SemanticSegmentationMetric whole(options()), a(options()), b(options()),
      ab(options()), ba(options());
  whole.update(prediction(), target());
  a.update(prediction().narrow(0, 0, 1), target().narrow(0, 0, 1));
  b.update(prediction().narrow(0, 1, 1), target().narrow(0, 1, 1));
  ab.merge(a);
  ab.merge(b);
  ba.merge(b);
  ba.merge(a);
  EXPECT_EQ(counts(ab), counts(whole));
  EXPECT_EQ(counts(ba), counts(whole));
  MemoryState state(a);
  a.save_state(state);
  SemanticSegmentationMetric restored(options(), {.chunk_pixels = 1});
  restored.load_state(state, RestorePolicy::exact_only);
  restored.merge(b);
  EXPECT_EQ(counts(restored), counts(whole));
  EXPECT_DOUBLE_EQ(restored.compute(), whole.compute());
  EXPECT_NO_THROW(validate_checkpoint_participant(restored));
  restored.merge(restored);
  EXPECT_EQ(restored.total_pixels(), 14);
  EXPECT_DOUBLE_EQ(restored.compute(), whole.compute());
  restored.reset();
  EXPECT_THROW(restored.compute(), NotReady);
}

TEST(SemanticSegmentation, AllTopologyDifferencesRejectRestoreAndMerge) {
  SemanticSegmentationMetric base(options());
  base.update(prediction(), target());
  MemoryState state(base);
  base.save_state(state);
  std::vector<SegmentationMetricOptions> alternatives;
  auto o = options();
  o.metric_id = "metric.other";
  alternatives.push_back(o);
  o = options();
  o.num_classes = 5;
  alternatives.push_back(o);
  o = options();
  o.score = SegmentationScore::dice;
  alternatives.push_back(o);
  o = options();
  o.input = SegmentationInput::logits;
  alternatives.push_back(o);
  o = options();
  o.average = ClassificationAverage::micro;
  alternatives.push_back(o);
  o = options();
  o.absent = AbsentClassPolicy::zero;
  alternatives.push_back(o);
  o = options();
  o.ignore_index.reset();
  alternatives.push_back(o);
  o = options();
  o.include_background = false;
  alternatives.push_back(o);
  o = options();
  o.background_index = 1;
  alternatives.push_back(o);
  for (const auto& option : alternatives) {
    SemanticSegmentationMetric other(option);
    EXPECT_THROW(other.load_state(state, RestorePolicy::exact_only),
                 CompatibilityError);
    EXPECT_THROW(base.merge(other), CompatibilityError);
    EXPECT_EQ(other.total_pixels(), 0);
  }
}

TEST(SemanticSegmentation, CorruptStateAndOverflowLeaveLiveCountsUnchanged) {
  SemanticSegmentationMetric metric(options());
  metric.update(prediction(), target());
  const auto before = counts(metric);
  MemoryState valid(metric);
  metric.save_state(valid);
  for (int mode = 0; mode < 8; ++mode) {
    auto bad = valid;
    if (mode == 0) bad.blobs["counts"].pop_back();
    if (mode == 1) bad.set_count(1, -1);
    if (mode == 2) bad.set_count(1, 4);
    if (mode == 3) bad.set_count(2, 1);
    if (mode == 4) bad.blobs["topology"][0] = std::byte{'!'};
    if (mode == 5) bad.state.schema_version = 2;
    if (mode == 6) bad.state.fields.pop_back();
    if (mode == 7) bad.state.fields[0].required = false;
    EXPECT_THROW(metric.load_state(bad, RestorePolicy::exact_only), Error);
    EXPECT_EQ(counts(metric), before);
  }
  auto large = valid;
  std::fill(large.blobs["counts"].begin(), large.blobs["counts"].end(),
            std::byte{0});
  large.set_count(0, std::numeric_limits<std::int64_t>::max());
  for (std::size_t slot = 1; slot <= 3; ++slot)
    large.set_count(slot, std::numeric_limits<std::int64_t>::max());
  metric.load_state(large, RestorePolicy::exact_only);
  const auto max_counts = counts(metric);
  EXPECT_THROW(metric.update(prediction(), target()), ResourceExhausted);
  EXPECT_THROW(metric.merge(metric), ResourceExhausted);
  EXPECT_EQ(counts(metric), max_counts);
  EXPECT_THROW(metric.load_state(valid, static_cast<RestorePolicy>(0)),
               CompatibilityError);
  EXPECT_EQ(counts(metric), max_counts);
}

TEST(SemanticSegmentation, InputErrorsAreAtomicAcrossMultipleTiles) {
  SemanticSegmentationMetric metric(options(), {.chunk_pixels = 1});
  metric.update(prediction(), target());
  const auto before = counts(metric);
  auto p = prediction().clone();
  p[1][1][1] = 4;
  EXPECT_THROW(metric.update(p, target()), InvalidArgument);
  auto t = target().clone();
  t[1][1][1] = 999;
  EXPECT_THROW(metric.update(prediction(), t), InvalidArgument);
  EXPECT_THROW(metric.update(prediction().to(torch::kFloat), target()),
               InvalidArgument);
  EXPECT_THROW(metric.update(prediction(), target().to(torch::kBool)),
               InvalidArgument);
  EXPECT_THROW(metric.update(prediction().reshape({1, 2, 4}), target()),
               InvalidArgument);
  EXPECT_THROW(metric.update(Tensor{}, target()), InvalidArgument);
  EXPECT_EQ(counts(metric), before);
  auto o = options();
  o.input = SegmentationInput::logits;
  SemanticSegmentationMetric logits(o, {.chunk_pixels = 1});
  auto x = torch::zeros({2, 4, 2, 2});
  x[1][3][1][1] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(logits.update(x, target()), NumericalError);
  EXPECT_EQ(logits.total_pixels(), 0);
}

TEST(SemanticSegmentation, LimitsAndDescriptorNamesAreExplicit) {
  EXPECT_EQ(segmentation_score_name(SegmentationScore::dice), "dice");
  EXPECT_EQ(segmentation_input_name(SegmentationInput::class_indices),
            "class_indices");
  EXPECT_EQ(absent_class_policy_name(AbsentClassPolicy::exclude), "exclude");
  EXPECT_THROW(SemanticSegmentationMetric({.num_classes = 0}), InvalidArgument);
  EXPECT_THROW(SemanticSegmentationMetric({.metric_id = "bad/id"}),
               InvalidArgument);
  EXPECT_THROW(
      SemanticSegmentationMetric({.score = static_cast<SegmentationScore>(0)}),
      InvalidArgument);
  EXPECT_THROW(SemanticSegmentationMetric({.background_index = 2}),
               InvalidArgument);
  EXPECT_THROW(SemanticSegmentationMetric(options(), {.chunk_pixels = 0}),
               InvalidArgument);
  EXPECT_THROW(SemanticSegmentationMetric(options(), {.max_classes = 3}),
               ResourceExhausted);
  EXPECT_THROW(SemanticSegmentationMetric(options(), {.max_state_bytes = 103}),
               ResourceExhausted);
  SemanticSegmentationMetric metric(options(), {.max_batch_pixels = 7});
  EXPECT_THROW(metric.update(prediction(), target()), ResourceExhausted);
  EXPECT_EQ(metric.total_pixels(), 0);
  EXPECT_EQ(metric.describe_state().fields[1].max_bytes, 104);
  SemanticSegmentationMetric many({.num_classes = 4096});
  EXPECT_EQ(many.describe_state().fields[1].max_bytes, 8 + 24 * 4096);
}

void reference_test(bool cuda) {
  const auto device = torch::Device(cuda ? "cuda:0" : "cpu");
  const auto path = std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                    "metrics/semantic-segmentation-reference.json";
  std::ifstream stream(path);
  ASSERT_TRUE(stream.good());
  const auto reference = nlohmann::json::parse(stream);
  std::vector<std::int64_t> p, t;
  for (int i = 0; i < 646; ++i) {
    p.push_back((i * 37 + i / 7) % 4);
    t.push_back(i % 13 == 0 ? -1 : (i * 11 + i / 3) % 4);
  }
  const auto predictions = torch::tensor(p).reshape({2, 17, 19}).to(device);
  const auto targets = torch::tensor(t).reshape({2, 17, 19}).to(device);
  for (const auto& c : reference.at("cases")) {
    auto o = options(c.at("dice").get<bool>() ? SegmentationScore::dice
                                              : SegmentationScore::iou);
    o.num_classes = 5;
    o.include_background = c.at("background");
    o.background_index = 2;
    o.absent = c.at("absent") == "exclude" ? AbsentClassPolicy::exclude
               : c.at("absent") == "zero"  ? AbsentClassPolicy::zero
                                           : AbsentClassPolicy::one;
    o.average = c.at("average") == "micro"   ? ClassificationAverage::micro
                : c.at("average") == "macro" ? ClassificationAverage::macro
                                             : ClassificationAverage::weighted;
    SemanticSegmentationMetric metric(o, {.chunk_pixels = 7}), a(o), b(o),
        merged(o);
    metric.update(predictions, targets);
    a.update(predictions.narrow(0, 0, 1), targets.narrow(0, 0, 1));
    b.update(predictions.narrow(0, 1, 1), targets.narrow(0, 1, 1));
    merged.merge(b);
    merged.merge(a);
    EXPECT_EQ(metric.total_pixels(),
              reference.at("valid_pixels").get<std::int64_t>());
    EXPECT_EQ(counts(metric), counts(merged));
    EXPECT_NEAR(metric.compute(), c.at("score").get<double>(), 1e-15)
        << c.dump();
    for (std::size_t i = 0; i < 5; ++i) {
      const auto& expected = reference.at("counts")[i];
      const auto& actual = metric.class_counts()[i];
      EXPECT_EQ(actual.intersection,
                expected.at("intersection").get<std::int64_t>());
      EXPECT_EQ(actual.target, expected.at("target").get<std::int64_t>());
      EXPECT_EQ(actual.predicted, expected.at("predicted").get<std::int64_t>());
    }
  }
}
TEST(SemanticSegmentation, IndependentSetRationalReferenceCPU) {
  reference_test(false);
}
TEST(SemanticSegmentation, IndependentSetRationalReferenceCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  reference_test(true);
}
}  // namespace
