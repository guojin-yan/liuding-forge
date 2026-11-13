// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/detection_evaluation.hpp>
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
DetectionEvaluationOptions options() {
  return {.metric = {.dataset_id = "synthetic.boxes.v1",
                     .class_vocabulary_id = "synthetic.classes.v1",
                     .num_classes = 2,
                     .iou_thresholds = {.5, .75},
                     .sources = {"a", "b", "empty"}},
          .expected_images = 2};
}
Tensor features(bool cuda = false) {
  return torch::tensor({20.f, 20.f, 30.f, 30.f, .9f,  0.f,  0.f,
                        10.f, 10.f, .1f,  0.f,  0.f,  10.f, 10.f,
                        .8f,  20.f, 20.f, 30.f, 30.f, .05f})
      .reshape({2, 2, 5})
      .to(cuda ? torch::kCUDA : torch::kCPU);
}
nn::ModelAdapter model(bool cuda, int& calls) {
  auto layer = std::make_shared<torch::nn::LinearImpl>(5, 5);
  layer->register_module("dropout", torch::nn::Dropout(.9));
  {
    torch::NoGradGuard guard;
    layer->weight.copy_(torch::eye(5));
    layer->bias.zero_();
  }
  if (cuda) layer->to(torch::kCUDA);
  return nn::adapt_module(
      layer,
      [&](torch::nn::LinearImpl& m,
          const nn::Invocation& input) -> c10::IValue {
        ++calls;
        if (m.is_training() || m.children().front()->is_training() ||
            torch::GradMode::is_enabled())
          throw std::runtime_error("inference guards missing");
        const auto y = m.forward(input.positional.front().toTensor());
        c10::impl::GenericList output(c10::AnyType::get());
        for (std::int64_t i = 0; i < y.size(0); ++i)
          output.push_back(c10::ivalue::Tuple::create(
              {y[i].narrow(1, 0, 4), y[i].select(1, 4),
               torch::zeros({y.size(1)}, y.options().dtype(torch::kInt32))}));
        return output;
      },
      {.positional = {ValueSchema::tensor()}},
      ValueSchema::list(
          ValueSchema::tuple({ValueSchema::tensor(), ValueSchema::tensor(),
                              ValueSchema::tensor()})));
}
DetectionEvaluationBatch batch(const Tensor& x, std::int64_t start = 0,
                               std::int64_t n = 2) {
  DetectionEvaluationBatch b{
      .invocation = {.positional = {x.narrow(0, start, n)}}, .images = {}};
  for (std::int64_t i = start; i < start + n; ++i)
    b.images.push_back({.image_id = static_cast<std::uint64_t>(i + 1),
                        .source_index = static_cast<std::size_t>(i),
                        .targets = {{{0, 0, 10, 10}, 0}},
                        .predictions = {}});
  return b;
}
DetectionEvaluationResult run(bool cuda, bool split) {
  int calls = 0;
  auto m = model(cuda, calls);
  m.state().module().children().front()->eval();
  const auto x = features(cuda);
  std::int64_t cursor = 0;
  const auto r = evaluate_detection(
      m,
      [&](auto) -> std::optional<DetectionEvaluationBatch> {
        if (cursor == 2) return std::nullopt;
        const auto n = split ? 1 : 2;
        auto b = batch(x, cursor, n);
        cursor += n;
        return b;
      },
      options());
  EXPECT_EQ(calls, split ? 2 : 1);
  EXPECT_TRUE(m.state().is_training());
  EXPECT_FALSE(m.state().module().children().front()->is_training());
  for (const auto& p : m.state().parameters()) EXPECT_FALSE(p.grad().defined());
  return r;
}
void check(bool cuda) {
  const auto all = run(cuda, false), split = run(cuda, true);
  ASSERT_TRUE(all.overall.mean_average_precision);
  EXPECT_NEAR(*all.overall.mean_average_precision, 2. / 3, 1e-14);
  EXPECT_EQ(all.overall.mean_average_precision,
            split.overall.mean_average_precision);
  EXPECT_EQ(all.by_source[0].mean_average_precision, .5);
  EXPECT_EQ(all.by_source[1].mean_average_precision, 1.);
  EXPECT_FALSE(all.by_source[2].mean_average_precision);
  EXPECT_EQ(all.overall.images, 2);
  EXPECT_EQ(all.observations.size(), 12);
  for (std::size_t i = 0; i < all.observations.size(); ++i) {
    EXPECT_EQ(all.observations[i].observation.value,
              split.observations[i].observation.value);
    EXPECT_EQ(all.observations[i].sample_count,
              split.observations[i].sample_count);
  }
  EXPECT_EQ(all.observations[0].definition.parameters.at("dataset_id"),
            options().metric.dataset_id);
}
TEST(DetectionEvaluation, ActualLinearInferencePartitionSourcesCPU) {
  check(false);
}
TEST(DetectionEvaluation, ActualLinearInferencePartitionSourcesCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  check(true);
}
TEST(DetectionEvaluation, ErrorsRestoreModesAndGradWithoutPartialReport) {
  int calls = 0;
  auto m = model(false, calls);
  m.state().module().children().front()->eval();
  const auto grad = torch::GradMode::is_enabled();
  auto x = features();
  x[0][1][4] = std::numeric_limits<float>::quiet_NaN();
  bool first = true;
  EXPECT_THROW(evaluate_detection(
                   m,
                   [&](auto) -> std::optional<DetectionEvaluationBatch> {
                     if (!first) return std::nullopt;
                     first = false;
                     return batch(x);
                   },
                   options()),
               InvalidArgument);
  EXPECT_TRUE(m.state().is_training());
  EXPECT_FALSE(m.state().module().children().front()->is_training());
  EXPECT_EQ(torch::GradMode::is_enabled(), grad);
  EXPECT_THROW(evaluate_detection(
                   m,
                   [](auto) -> std::optional<DetectionEvaluationBatch> {
                     throw std::runtime_error("source failed");
                   },
                   options()),
               std::runtime_error);
  EXPECT_TRUE(m.state().is_training());
}
TEST(DetectionEvaluation, ResourceAndSourceErrorsRejectBeforeInvocation) {
  int calls = 0;
  auto m = model(false, calls);
  const auto x = features();
  const auto source = [&](auto) -> std::optional<DetectionEvaluationBatch> {
    return batch(x);
  };
  auto o = options();
  o.limits.max_images = 1;
  o.expected_images.reset();
  EXPECT_THROW(evaluate_detection(m, source, o), ResourceExhausted);
  o = options();
  o.max_summary_values = 1;
  EXPECT_THROW(evaluate_detection(m, source, o), ResourceExhausted);
  o = options();
  o.metric.sources = {"a"};
  EXPECT_THROW(evaluate_detection(m, source, o), InvalidArgument);
  EXPECT_EQ(calls, 0);
  o = options();
  o.max_conversion_bytes = 1;
  EXPECT_THROW(evaluate_detection(m, source, o), ResourceExhausted);
  EXPECT_EQ(calls, 1);
  EXPECT_THROW(evaluate_detection(m, {}, options()), InvalidArgument);
}
TEST(DetectionEvaluation, ExpectedDatasetSizeEOFAndDuplicateImages) {
  int calls = 0;
  auto m = model(false, calls);
  const auto x = features();
  bool first = true;
  EXPECT_THROW(evaluate_detection(
                   m,
                   [&](auto) -> std::optional<DetectionEvaluationBatch> {
                     if (!first) return std::nullopt;
                     first = false;
                     return batch(x, 0, 1);
                   },
                   options()),
               InvalidArgument);
  first = true;
  EXPECT_THROW(evaluate_detection(
                   m,
                   [&](auto) -> std::optional<DetectionEvaluationBatch> {
                     if (!first) return std::nullopt;
                     first = false;
                     auto b = batch(x);
                     b.images[1].image_id = b.images[0].image_id;
                     return b;
                   },
                   options()),
               InvalidArgument);
  auto o = options();
  o.expected_images.reset();
  const auto empty = evaluate_detection(
      m,
      [](auto) -> std::optional<DetectionEvaluationBatch> {
        return std::nullopt;
      },
      o);
  EXPECT_FALSE(empty.overall.mean_average_precision);
}
TEST(DetectionEvaluation, CancellationAndDeadlineDoNotReturnPartialReports) {
  int calls = 0;
  auto m = model(false, calls);
  const auto source = [&](auto) -> std::optional<DetectionEvaluationBatch> {
    return batch(features());
  };
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(evaluate_detection(m, source, options(), stop.get_token()),
               Cancelled);
  EXPECT_THROW(evaluate_detection(
                   m, source, options(), {},
                   std::chrono::steady_clock::now() - std::chrono::seconds(1)),
               DeadlineExceeded);
  EXPECT_EQ(calls, 0);
  std::stop_source middle;
  int loads = 0;
  EXPECT_THROW(evaluate_detection(
                   m,
                   [&](auto) -> std::optional<DetectionEvaluationBatch> {
                     if (loads++ == 1) {
                       middle.request_stop();
                       return std::nullopt;
                     }
                     return batch(features(), 0, 1);
                   },
                   options(), middle.get_token()),
               Cancelled);
  EXPECT_EQ(calls, 1);
}
TEST(DetectionEvaluation,
     RejectsMalformedOutputDtypeShapeAndCallerPredictions) {
  for (int mode = 0; mode < 5; ++mode) {
    auto module = std::make_shared<torch::nn::Module>();
    auto m = nn::adapt_module(
        module,
        [&](torch::nn::Module&, const nn::Invocation&) -> c10::IValue {
          if (mode == 0) return torch::zeros({1});
          c10::impl::GenericList out(c10::AnyType::get());
          for (int i = 0; i < 2; ++i)
            out.push_back(c10::ivalue::Tuple::create(
                {torch::zeros({1, mode == 1 ? 5 : 4}),
                 torch::zeros({mode == 2 ? 2 : 1}),
                 torch::zeros({1},
                              mode == 3 ? torch::kFloat32 : torch::kInt64)}));
          return out;
        },
        {.positional = {ValueSchema::tensor()}}, ValueSchema::any());
    bool first = true;
    EXPECT_THROW(
        evaluate_detection(
            m,
            [&](auto) -> std::optional<DetectionEvaluationBatch> {
              if (!first) return std::nullopt;
              first = false;
              auto b = batch(features());
              if (mode == 4)
                b.images[0].predictions.push_back({{0, 0, 1, 1}, .5, 0});
              return b;
            },
            options()),
        InvalidArgument)
        << mode;
  }
}
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
TEST(DetectionEvaluation, ActualAPObservationsReachVersionedQualityReport) {
  const auto value = run(false, true);
  const auto& overall = value.observations.front();
  experiment::EvaluationSuite suite{
      .suite_id = "boxes.suite",
      .task = "axis_aligned_box_detection",
      .dataset_digest = {.hex = std::string(64, 'a')},
      .environment_id = "cpu.test"};
  suite.rules.push_back({.rule_id = "macro.ap",
                         .definition = overall.definition,
                         .context = overall.context,
                         .minimum_samples = 2,
                         .minimum = .65});
  experiment::EvaluationIdentity id{
      .evaluation_id = "boxes.eval",
      .run_id = "boxes.run",
      .model_digest = {.hex = std::string(64, 'b')},
      .build_fingerprint = "test-build",
      .adapter_id = "training.boxes.v1",
      .config_digest = {.hex = std::string(64, 'c')}};
  const auto report =
      experiment::run_evaluation(suite, id, [&](const auto&, auto, auto) {
        return std::vector<TelemetryMetricPoint>{overall};
      });
  EXPECT_EQ(report.verdict, experiment::QualityVerdict::pass);
}
#endif
}  // namespace
