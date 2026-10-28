// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/metrics/detection.hpp>

#include <bit>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training::metrics;
using DetectionMetric = DetectionAveragePrecisionMetric;
DetectionMetricOptions options() {
  return {.dataset_id = "fixture.dataset",
          .class_vocabulary_id = "fixture.classes",
          .num_classes = 2,
          .iou_thresholds = {.5, .75}};
}
DetectionImage first() {
  return {.image_id = 1,
          .targets = {{{0, 0, 10, 10}, 0}},
          .predictions = {{{20, 20, 30, 30}, .9, 0}, {{0, 0, 10, 10}, .1, 0}}};
}
DetectionImage second() {
  return {.image_id = 2,
          .targets = {{{0, 0, 10, 10}, 0}},
          .predictions = {{{0, 0, 10, 10}, .8, 0}}};
}
void ingest(DetectionMetric& m, std::initializer_list<DetectionImage> images) {
  m.update(std::span(images.begin(), images.size()));
}
class MemoryState final : public StateReader, public StateWriter {
 public:
  explicit MemoryState(const CheckpointParticipant& m)
      : state(m.describe_state()) {}
  const StateManifest& manifest() const noexcept override { return state; }
  bool contains(std::string_view n) const override {
    return blobs.contains(std::string(n));
  }
  std::span<const std::byte> read_blob(std::string_view n,
                                       std::uint64_t max) const override {
    const auto& b = blobs.at(std::string(n));
    if (b.size() > max) throw ResourceExhausted("Test read bound");
    return b;
  }
  void write_blob(std::string_view n, std::span<const std::byte> b) override {
    blobs.emplace(std::string(n), std::vector<std::byte>(b.begin(), b.end()));
  }
  void slot(std::size_t index, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i)
      blobs.at("matches")[index * 8 + i] =
          static_cast<std::byte>((value >> (8U * i)) & 255U);
  }
  StateManifest state;
  std::map<std::string, std::vector<std::byte>> blobs;
};
TEST(DetectionAP, DatasetMatchingIsNotMeanBatchAP) {
  DetectionMetric whole(options()), a(options()), b(options()),
      merged(options());
  ingest(whole, {first(), second()});
  ingest(a, {first()});
  ingest(b, {second()});
  merged.merge(b);
  merged.merge(a);
  EXPECT_NEAR(whole.compute(), 2. / 3, 1e-14);
  EXPECT_NE(whole.compute(), (a.compute() + b.compute()) / 2);
  EXPECT_DOUBLE_EQ(whole.compute(), merged.compute());
  const auto s = whole.summarize();
  EXPECT_EQ(s.images, 2);
  EXPECT_EQ(s.per_class[0].ground_truth, 2);
  EXPECT_EQ(s.per_class[0].predictions, 3);
  EXPECT_EQ(s.per_class[0].recall_by_iou[1], 1.);
  EXPECT_FALSE(s.per_class[1].average_precision);
  MemoryState x(whole), y(merged);
  whole.save_state(x);
  merged.save_state(y);
  EXPECT_EQ(x.blobs, y.blobs);
  EXPECT_EQ(whole.state_bytes(), x.blobs.at("matches").size());
}
TEST(DetectionAP, ScoreTiesUseImageIdentityAndRowBeforeMergeOrder) {
  auto x = first(), y = second();
  x.predictions = {{{0, 0, 10, 10}, .5, 0}};
  y.predictions = {{{20, 20, 30, 30}, .5, 0}};
  DetectionMetric a(options()), b(options());
  ingest(a, {y, x});
  ingest(b, {x});
  ingest(b, {y});
  EXPECT_DOUBLE_EQ(a.compute(), b.compute());
  EXPECT_NEAR(a.compute(), 51. / 101, 1e-14);
  auto o = options();
  o.max_detections_per_image_class = 1;
  DetectionMetric truncated(o);
  ingest(truncated, {first(), second()});
  EXPECT_NEAR(truncated.compute(), 51. / 202, 1e-14);
  x.predictions = {{{20, 20, 30, 30}, .5, 0}, {{0, 0, 10, 10}, .5, 0}};
  DetectionMetric row(o);
  ingest(row, {x});
  EXPECT_DOUBLE_EQ(row.compute(), 0.);
}
TEST(DetectionAP, CrowdUsesRepeatIntersectionOverDetectionArea) {
  auto x = second();
  x.targets.push_back({{20, 20, 120, 120}, 0, true});
  x.predictions.insert(x.predictions.begin(), {{{25, 25, 30, 30}, .99, 0},
                                               {{35, 35, 40, 40}, .98, 0}});
  DetectionMetric crowd(options());
  ingest(crowd, {x});
  EXPECT_DOUBLE_EQ(crowd.compute(), 1.);
  EXPECT_EQ(crowd.summarize().per_class[0].ground_truth, 1);
  x.targets.back().crowd = false;
  x.targets.back().ignore = true;
  x.targets.back().box = {20, 20, 30, 30};
  x.predictions[0].box = x.predictions[1].box = {20, 20, 30, 30};
  DetectionMetric ignored(options());
  ingest(ignored, {x});
  EXPECT_DOUBLE_EQ(ignored.compute(), .5);
}
TEST(DetectionAP, EligibleGroundTruthWinsAndEqualIoUSelectsLastAnnotation) {
  auto o = options();
  o.iou_thresholds = {.5};
  DetectionImage x{
      .image_id = 8,
      .targets = {{{0, 0, 10, 10}, 0}, {{5, 0, 15, 10}, 0}},
      .predictions = {{{2.5, 0, 12.5, 10}, .9, 0}, {{0, 0, 10, 10}, .8, 0}}};
  DetectionMetric tied(o);
  ingest(tied, {x});
  EXPECT_DOUBLE_EQ(tied.compute(), 1.);
  x.targets[1].ignore = true;
  x.predictions.resize(1);
  DetectionMetric priority(o);
  ingest(priority, {x});
  EXPECT_DOUBLE_EQ(priority.compute(), 1.);
}
TEST(DetectionAP, InclusiveAreaAndExplicitAnnotationArea) {
  auto o = options();
  o.min_area = o.max_area = 100;
  auto x = second();
  x.predictions.insert(x.predictions.begin(), {{20, 20, 220, 220}, .99, 0});
  DetectionMetric boundary(o);
  ingest(boundary, {x});
  EXPECT_DOUBLE_EQ(boundary.compute(), 1.);
  x.targets[0].area = 99;
  DetectionMetric annotation(o);
  ingest(annotation, {x});
  EXPECT_THROW(annotation.compute(), NotReady);
  x.targets.push_back({{40, 40, 50, 50}, 0});
  DetectionMetric unmatched(o);
  ingest(unmatched, {x});
  EXPECT_DOUBLE_EQ(unmatched.compute(), 0.);
}
TEST(DetectionAP, EmptyAbsentAndSourcePoliciesAreExplicit) {
  auto o = options();
  o.sources = {"a", "b", "empty"};
  auto x = second();
  x.source_index = 1;
  DetectionMetric m(o);
  EXPECT_THROW(m.compute(), NotReady);
  ingest(m, {x});
  EXPECT_EQ(m.summarize(1).mean_average_precision, 1.);
  EXPECT_FALSE(m.summarize(0).mean_average_precision);
  EXPECT_THROW(m.summarize(3), InvalidArgument);
  o.no_ground_truth = DetectionNoGroundTruth::zero;
  DetectionMetric zero(o);
  ingest(zero, {x});
  EXPECT_EQ(zero.compute(), .5);
  o.no_ground_truth = DetectionNoGroundTruth::error;
  DetectionMetric error(o);
  EXPECT_FALSE(error.summarize().mean_average_precision);
  ingest(error, {x});
  EXPECT_THROW(error.compute(), NumericalError);
}
TEST(DetectionAP, CompleteImageIdentityRejectsDuplicateAndOverlappingMerge) {
  DetectionMetric a(options()), b(options());
  auto x = first();
  x.image_id = std::numeric_limits<std::uint64_t>::max();
  ingest(a, {x});
  const auto before = a.compute();
  EXPECT_THROW(ingest(a, {second(), x}), InvalidArgument);
  EXPECT_EQ(a.image_count(), 1);
  EXPECT_EQ(a.compute(), before);
  ingest(b, {x});
  EXPECT_THROW(a.merge(b), InvalidArgument);
  EXPECT_THROW(a.merge(a), InvalidArgument);
  EXPECT_EQ(a.image_count(), 1);
  a.reset();
  EXPECT_EQ(a.state_bytes(), 16);
  EXPECT_THROW(a.compute(), NotReady);
}
TEST(DetectionAP, InvalidInputsLeaveAllStateUnchangedEvenPastMaxDet) {
  auto o = options();
  o.max_detections_per_image_class = 1;
  DetectionMetric m(o);
  ingest(m, {second()});
  auto x = first();
  x.predictions.back().score = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(ingest(m, {x}), InvalidArgument);
  x = first();
  x.predictions.back().box[2] = 0;
  EXPECT_THROW(ingest(m, {x}), InvalidArgument);
  x = first();
  x.targets[0].area = -1;
  EXPECT_THROW(ingest(m, {x}), InvalidArgument);
  x = first();
  x.predictions[0].class_index = 2;
  EXPECT_THROW(ingest(m, {x}), InvalidArgument);
  x = first();
  x.source_index = 0;
  EXPECT_THROW(ingest(m, {x}), InvalidArgument);
  EXPECT_EQ(m.image_count(), 1);
  EXPECT_DOUBLE_EQ(m.compute(), 1.);
}
TEST(DetectionAP, StateWorkspaceAndOperationLimitsFailAtomically) {
  auto o = options();
  DetectionMetric m(o, {.max_state_bytes = 100});
  EXPECT_THROW(ingest(m, {first()}), ResourceExhausted);
  EXPECT_EQ(m.image_count(), 0);
  DetectionMetric work(o, {.max_workspace_bytes = 1});
  EXPECT_THROW(ingest(work, {first()}), ResourceExhausted);
  DetectionMetric time(o, {.max_match_operations_per_update = 1});
  EXPECT_THROW(ingest(time, {second()}), ResourceExhausted);
  DetectionMetric retained(o, {.max_retained_predictions = 1});
  EXPECT_THROW(ingest(retained, {second(), first()}), ResourceExhausted);
  EXPECT_EQ(retained.image_count(), 0);
  ingest(work, {{.image_id = 8}});
  EXPECT_THROW(work.compute(), ResourceExhausted);
  DetectionMetric images(o, {.max_images = 1});
  EXPECT_THROW(ingest(images, {first(), second()}), ResourceExhausted);
}
TEST(DetectionAP, CocoFP64RecallGridBoundaryAndAllSixtyFourThresholdBits) {
  auto o = options();
  o.iou_thresholds.clear();
  for (int k = 1; k <= 64; ++k) o.iou_thresholds.push_back(k / 64.);
  DetectionImage image{.image_id = 1};
  for (int i = 0; i < 20; ++i) {
    const DetectionBox box{20. * i, 0., 20. * i + 10., 10.};
    image.targets.push_back({box, 0});
    if (i < 7) image.predictions.push_back({box, 1. - i / 100., 0});
  }
  DetectionMetric metric(o), restored(o);
  ingest(metric, {image});
  EXPECT_NEAR(metric.compute(), 35. / 101, 1e-14);
  const auto result = metric.summarize();
  ASSERT_EQ(result.per_class[0].ap_by_iou.size(), 64);
  EXPECT_NEAR(*result.per_class[0].ap_by_iou.back(), 35. / 101, 1e-14);
  EXPECT_EQ(result.per_class[0].recall_by_iou.back(), .35);
  MemoryState state(metric);
  metric.save_state(state);
  restored.load_state(state, RestorePolicy::exact_only);
  EXPECT_EQ(metric.compute(), restored.compute());
}
TEST(DetectionAP, ConfigurationAndDatasetIdentityAreFrozen) {
  auto o = options();
  o.dataset_id.clear();
  EXPECT_THROW(DetectionMetric bad(o), InvalidArgument);
  o = options();
  o.iou_thresholds = {.5, .5};
  EXPECT_THROW(DetectionMetric bad(o), InvalidArgument);
  o.iou_thresholds = {0};
  EXPECT_THROW(DetectionMetric bad(o), InvalidArgument);
  o = options();
  o.sources = {"a", "a"};
  EXPECT_THROW(DetectionMetric bad(o), InvalidArgument);
  DetectionMetric m(options());
  ingest(m, {first()});
  for (int i = 0; i < 7; ++i) {
    o = options();
    switch (i) {
      case 0:
        o.dataset_id += "2";
        break;
      case 1:
        o.class_vocabulary_id += "2";
        break;
      case 2:
        o.max_area = 100;
        break;
      case 3:
        o.max_detections_per_image_class = 10;
        break;
      case 4:
        o.iou_thresholds = {.5};
        break;
      case 5:
        o.no_ground_truth = DetectionNoGroundTruth::zero;
        break;
      default:
        o.sources = {"a"};
    }
    DetectionMetric incompatible(o);
    EXPECT_THROW(m.merge(incompatible), CompatibilityError);
  }
}
TEST(DetectionAP, ExactStateRoundTripAndCorruptionRollback) {
  DetectionMetric m(options()),
      restored(options(), {.max_state_bytes = 1ULL << 20U});
  ingest(m, {first(), second()});
  MemoryState state(m);
  m.save_state(state);
  restored.load_state(state, RestorePolicy::exact_only);
  EXPECT_DOUBLE_EQ(restored.compute(), m.compute());
  MemoryState roundtrip(restored);
  restored.save_state(roundtrip);
  EXPECT_EQ(roundtrip.blobs, state.blobs);
  for (int i = 0; i < 8; ++i) {
    auto bad = state;
    switch (i) {
      case 0:
        bad.blobs["matches"].pop_back();
        break;
      case 1:
        bad.slot(0, 0);
        break;
      case 2:
        bad.slot(8, std::bit_cast<std::uint64_t>(
                        std::numeric_limits<double>::infinity()));
        break;
      case 3:
        bad.slot(10, 4);
        break;
      case 4:
        bad.slot(10, 1);
        bad.slot(11, 1);
        break;
      case 5:
        bad.slot(6, 0);
        bad.slot(14, 1);
        break;
      case 6:
        bad.slot(9, 1);
        break;
      default:
        bad.blobs["matches"].push_back(std::byte{0});
    }
    EXPECT_THROW(restored.load_state(bad, RestorePolicy::exact_only),
                 SerializationError)
        << i;
    EXPECT_DOUBLE_EQ(restored.compute(), m.compute());
  }
  auto bad = state;
  bad.state.schema_version = 2;
  EXPECT_THROW(restored.load_state(bad, RestorePolicy::exact_only),
               CompatibilityError);
  auto changed = options();
  changed.dataset_id += "2";
  DetectionMetric other(changed);
  EXPECT_THROW(other.load_state(state, RestorePolicy::exact_only),
               CompatibilityError);
}
void optional_near(std::optional<double> actual,
                   const nlohmann::json& expected) {
  EXPECT_EQ(actual.has_value(), !expected.is_null());
  if (actual && !expected.is_null())
    EXPECT_NEAR(*actual, expected.get<double>(), 1e-12);
}
TEST(DetectionAP, OfficialCocoReferenceAreasMaxDetSourcesAndReverseMerge) {
  std::ifstream stream(std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                       "metrics/box-detection-reference.json");
  ASSERT_TRUE(stream.good());
  const auto r = nlohmann::json::parse(stream);
  std::vector<DetectionImage> images;
  for (const auto& row : r.at("images")) {
    DetectionImage image{
        .image_id = row.at("image_id"),
        .source_index = row.at("source_index").get<std::size_t>(),
        .targets = {},
        .predictions = {}};
    for (const auto& t : row.at("targets"))
      image.targets.push_back({.box = t.at("box").get<DetectionBox>(),
                               .class_index = t.at("class_index"),
                               .crowd = t.at("crowd"),
                               .area = t.at("area").get<double>()});
    for (const auto& p : row.at("predictions"))
      image.predictions.push_back({p.at("box").get<DetectionBox>(),
                                   p.at("score"), p.at("class_index")});
    images.push_back(std::move(image));
  }
  for (const auto& c : r.at("cases")) {
    auto o = options();
    o.dataset_id = r.at("dataset_id");
    o.num_classes = r.at("num_classes");
    o.sources = r.at("sources").get<std::vector<std::string>>();
    o.iou_thresholds = {.5, .55, .6, .65, .7, .75, .8, .85, .9, .95};
    o.min_area = c.at("min_area");
    o.max_area = c.at("max_area");
    o.max_detections_per_image_class = c.at("max_det");
    DetectionMetric whole(o), merged(o);
    whole.update(images);
    for (auto it = images.rbegin(); it != images.rend(); ++it) {
      DetectionMetric shard(o);
      ingest(shard, {*it});
      merged.merge(shard);
    }
    MemoryState a(whole), b(merged);
    whole.save_state(a);
    merged.save_state(b);
    EXPECT_EQ(a.blobs, b.blobs);
    for (const auto& expected : c.at("summaries")) {
      const auto source = expected.at("source").is_null()
                              ? std::optional<std::size_t>{}
                              : std::optional<std::size_t>{
                                    expected.at("source").get<std::size_t>()};
      const auto actual = whole.summarize(source);
      optional_near(actual.mean_average_precision, expected.at("mean_ap"));
      for (std::size_t cls = 0; cls < o.num_classes; ++cls)
        for (std::size_t k = 0; k < o.iou_thresholds.size(); ++k) {
          optional_near(actual.per_class[cls].ap_by_iou[k],
                        expected.at("per_class")[cls].at("ap_by_iou")[k]);
          optional_near(actual.per_class[cls].recall_by_iou[k],
                        expected.at("per_class")[cls].at("recall_by_iou")[k]);
        }
    }
    EXPECT_FALSE(whole.summarize(2).mean_average_precision);
  }
}
}  // namespace
