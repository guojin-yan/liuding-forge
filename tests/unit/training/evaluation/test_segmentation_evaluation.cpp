// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/segmentation_evaluation.hpp>
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
#include <jyppx/liuding/experiment/evaluation.hpp>
#endif

#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training;

nn::ModelAdapter model(bool cuda, int& calls) {
  auto layer = std::make_shared<torch::nn::Conv2dImpl>(
      torch::nn::Conv2dOptions(1, 3, 1));
  layer->register_module("dropout", torch::nn::Dropout(0.9));
  {
    torch::NoGradGuard guard;
    layer->weight.copy_(torch::tensor({-1.f, 1.f, 0.f}).reshape({3, 1, 1, 1}));
    layer->bias.copy_(torch::tensor({.5f, -.5f, -2.f}));
  }
  if (cuda) layer->to(torch::kCUDA);
  return nn::adapt_module(
      layer,
      [&](torch::nn::Conv2dImpl& m,
          const nn::Invocation& input) -> c10::IValue {
        ++calls;
        if (m.is_training() || m.children().front()->is_training() ||
            torch::GradMode::is_enabled())
          throw std::runtime_error("inference guard missing");
        return m.forward(input.positional.front().toTensor());
      },
      {.positional = {ValueSchema::tensor()}}, ValueSchema::tensor());
}
Tensor images(bool cuda = false) {
  return torch::tensor({0.f, 1.f, 1.f, 0.f, 1.f, 0.f, 1.f, 0.f})
      .reshape({2, 1, 2, 2})
      .to(cuda ? torch::kCUDA : torch::kCPU);
}
Tensor labels(bool cuda = false) {
  return torch::tensor({0, 1, 0, -1, 1, 0, 1, 0}, torch::kInt64)
      .reshape({2, 2, 2})
      .to(cuda ? torch::kCUDA : torch::kCPU);
}
SegmentationEvaluationOptions options() {
  return {.metric = {.metric_id = "metric.semantic",
                     .num_classes = 3,
                     .ignore_index = -1},
          .sources = {"left", "right", "empty"}};
}
SegmentationEvaluationResult run(bool cuda, bool split,
                                 metrics::SegmentationScore score) {
  int calls = 0;
  auto m = model(cuda, calls);
  m.state().module().children().front()->eval();
  const auto x = images(cuda), y = labels(cuda);
  std::int64_t offset = 0;
  auto o = options();
  o.metric.score = score;
  const auto result = evaluate_semantic_segmentation(
      m,
      [&](std::stop_token) -> std::optional<SegmentationEvaluationBatch> {
        if (offset == 2) return std::nullopt;
        const auto n = split ? 1 : 2;
        SegmentationEvaluationBatch batch{
            .invocation = {.positional = {x.narrow(0, offset, n)}},
            .targets = y.narrow(0, offset, n),
            .source_indices =
                split
                    ? std::vector<std::size_t>{static_cast<std::size_t>(offset)}
                    : std::vector<std::size_t>{0, 1}};
        offset += n;
        return batch;
      },
      o);
  EXPECT_EQ(calls, split ? 2 : 1);
  EXPECT_TRUE(m.state().is_training());
  EXPECT_FALSE(m.state().module().children().front()->is_training());
  for (const auto& p : m.state().parameters()) EXPECT_FALSE(p.grad().defined());
  return result;
}
void check(bool cuda) {
  for (auto score :
       {metrics::SegmentationScore::iou, metrics::SegmentationScore::dice}) {
    const auto whole = run(cuda, false, score), split = run(cuda, true, score);
    EXPECT_EQ(whole.images, 2);
    EXPECT_EQ(whole.pixels_seen, 8);
    EXPECT_EQ(whole.overall.pixels, 7);
    EXPECT_EQ(whole.overall.counts[0].intersection, 3);
    EXPECT_EQ(whole.overall.counts[0].target, 4);
    EXPECT_EQ(whole.overall.counts[1].intersection, 3);
    EXPECT_EQ(whole.overall.counts[1].predicted, 4);
    EXPECT_DOUBLE_EQ(*whole.overall.score,
                     score == metrics::SegmentationScore::iou ? .75 : 6. / 7);
    EXPECT_EQ(whole.overall.score, split.overall.score);
    EXPECT_EQ(whole.by_source[1].score, 1.);
    EXPECT_FALSE(whole.by_source[2].score);
    EXPECT_FALSE(whole.overall.per_class[2]);
    EXPECT_EQ(whole.observations.size(), 16);
    for (std::size_t i = 0; i < whole.observations.size(); ++i) {
      EXPECT_EQ(whole.observations[i].observation.value,
                split.observations[i].observation.value);
      EXPECT_EQ(whole.observations[i].sample_count,
                split.observations[i].sample_count);
    }
  }
}
TEST(SegmentationEvaluation, ActualConvolutionPartitionSourcesCPU) {
  check(false);
}
TEST(SegmentationEvaluation, ActualConvolutionPartitionSourcesCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  check(true);
}
TEST(SegmentationEvaluation, ModeAndGradRestorationOnFailure) {
  int calls = 0;
  auto m = model(false, calls);
  m.state().module().children().front()->eval();
  const bool grad = torch::GradMode::is_enabled();
  auto x = images();
  x[1][0][1][1] = std::numeric_limits<float>::quiet_NaN();
  bool first = true;
  EXPECT_THROW(evaluate_semantic_segmentation(
                   m,
                   [&](auto) -> std::optional<SegmentationEvaluationBatch> {
                     if (!first) return std::nullopt;
                     first = false;
                     return SegmentationEvaluationBatch{
                         .invocation = {.positional = {x}},
                         .targets = labels()};
                   },
                   {.metric = {.num_classes = 3, .ignore_index = -1}}),
               NumericalError);
  EXPECT_TRUE(m.state().is_training());
  EXPECT_FALSE(m.state().module().children().front()->is_training());
  EXPECT_EQ(torch::GradMode::is_enabled(), grad);
  EXPECT_THROW(evaluate_semantic_segmentation(
                   m,
                   [](auto) -> std::optional<SegmentationEvaluationBatch> {
                     throw std::runtime_error("source failed");
                   }),
               std::runtime_error);
  EXPECT_TRUE(m.state().is_training());
}
TEST(SegmentationEvaluation, EmptyAndAllIgnoredRemainMissing) {
  int calls = 0;
  auto m = model(false, calls);
  const auto empty = evaluate_semantic_segmentation(
      m, [](auto) -> std::optional<SegmentationEvaluationBatch> {
        return std::nullopt;
      });
  EXPECT_EQ(calls, 0);
  EXPECT_FALSE(empty.overall.score);
  EXPECT_EQ(empty.observations[0].sample_count, 0);
  auto o = options();
  o.sources.clear();
  bool first = true;
  const auto ignored = evaluate_semantic_segmentation(
      m,
      [&](auto) -> std::optional<SegmentationEvaluationBatch> {
        if (!first) return std::nullopt;
        first = false;
        return SegmentationEvaluationBatch{
            .invocation = {.positional = {images()}},
            .targets = torch::full({2, 2, 2}, -1, torch::kInt64)};
      },
      o);
  EXPECT_EQ(ignored.pixels_seen, 8);
  EXPECT_EQ(ignored.overall.pixels, 0);
  EXPECT_FALSE(ignored.overall.score);
}
TEST(SegmentationEvaluation, LimitsAndSourceIDsRejectBeforeInference) {
  int calls = 0;
  auto m = model(false, calls);
  const auto source = [](auto) -> std::optional<SegmentationEvaluationBatch> {
    return SegmentationEvaluationBatch{.invocation = {.positional = {images()}},
                                       .targets = labels(),
                                       .source_indices = {0, 1}};
  };
  auto o = options();
  o.max_images = 1;
  EXPECT_THROW(evaluate_semantic_segmentation(m, source, o), ResourceExhausted);
  o = options();
  o.max_pixels = 7;
  EXPECT_THROW(evaluate_semantic_segmentation(m, source, o), ResourceExhausted);
  o = options();
  o.max_counts_bytes = 1;
  EXPECT_THROW(evaluate_semantic_segmentation(m, source, o), ResourceExhausted);
  o = options();
  o.sources = {"left"};
  EXPECT_THROW(evaluate_semantic_segmentation(m, source, o), InvalidArgument);
  o = options();
  o.sources = {"a", "a"};
  EXPECT_THROW(evaluate_semantic_segmentation(m, source, o), InvalidArgument);
  EXPECT_EQ(calls, 0);
  o = options();
  o.max_batches = 1;
  EXPECT_THROW(evaluate_semantic_segmentation(m, source, o), ResourceExhausted);
  EXPECT_EQ(calls, 1);
  EXPECT_THROW(evaluate_semantic_segmentation(m, {}), InvalidArgument);
}
TEST(SegmentationEvaluation, CancellationAndDeadlineNeverReturnPartialReport) {
  int calls = 0;
  auto m = model(false, calls);
  std::stop_source stop;
  stop.request_stop();
  const auto empty = [](auto) -> std::optional<SegmentationEvaluationBatch> {
    return std::nullopt;
  };
  EXPECT_THROW(evaluate_semantic_segmentation(m, empty, {}, stop.get_token()),
               Cancelled);
  EXPECT_THROW(evaluate_semantic_segmentation(m, empty, {}, {},
                                              std::chrono::steady_clock::now()),
               DeadlineExceeded);
  std::stop_source later;
  EXPECT_THROW(evaluate_semantic_segmentation(
                   m,
                   [&](auto) -> std::optional<SegmentationEvaluationBatch> {
                     later.request_stop();
                     return std::nullopt;
                   },
                   {}, later.get_token()),
               Cancelled);
  EXPECT_EQ(calls, 0);
}
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
TEST(SegmentationEvaluation, RealObservationsReachVersionedQualityReport) {
  const auto value = run(false, true, metrics::SegmentationScore::iou);
  experiment::EvaluationSuite suite{
      .suite_id = "mask.suite",
      .task = "semantic_segmentation",
      .dataset_digest = {.hex = std::string(64, 'a')},
      .environment_id = "cpu.test"};
  const auto& overall = value.observations.front();
  suite.rules.push_back({.rule_id = "macro.iou",
                         .definition = overall.definition,
                         .context = overall.context,
                         .minimum_samples = 7,
                         .minimum = .70});
  experiment::EvaluationIdentity id{
      .evaluation_id = "mask.eval",
      .run_id = "mask.run",
      .model_digest = {.hex = std::string(64, 'b')},
      .build_fingerprint = "test-build",
      .adapter_id = "training.semantic.v1",
      .config_digest = {.hex = std::string(64, 'c')}};
  const auto report =
      experiment::run_evaluation(suite, id, [&](const auto&, auto, auto) {
        return std::vector<TelemetryMetricPoint>{overall};
      });
  EXPECT_EQ(report.verdict, experiment::QualityVerdict::pass);
}
#endif
}  // namespace
