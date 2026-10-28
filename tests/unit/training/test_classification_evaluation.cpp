// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
#include <jyppx/liuding/experiment/evaluation.hpp>
#endif

#include <torch/torch.h>

#include <chrono>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::training;

nn::ModelAdapter identity() {
  auto module = std::make_shared<torch::nn::Module>();
  module->register_module("dropout", torch::nn::Dropout(0.8));
  return nn::adapt_module(
      module,
      [](torch::nn::Module& module, const nn::Invocation& call) -> c10::IValue {
        if (torch::GradMode::is_enabled() || module.is_training() ||
            module.children().front()->is_training())
          throw std::runtime_error("inference guard not active");
        return call.positional.front();
      },
      {.positional = {ValueSchema::tensor()}}, ValueSchema::tensor());
}

Tensor logits() {
  return torch::tensor({{3., 1., 0.},
                        {0., 2., 1.},
                        {0., 0., 1.},
                        {2., 2., 0.},
                        {9., 0., 0.},
                        {0., 0., 0.}});
}
Tensor labels() { return torch::tensor({0, 1, 1, 1, -1, 2}, torch::kInt64); }

ClassificationEvaluationResult evaluate(bool cuda, std::int64_t batch_size) {
  auto model = identity();
  const auto device = torch::Device(cuda ? torch::kCUDA : torch::kCPU);
  auto x = logits().to(device), y = labels().to(device);
  const std::vector<std::size_t> sources{0, 1, 0, 1, 0, 0};
  std::int64_t offset = 0;
  return evaluate_classification(
      model,
      [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
        if (offset == 6) return std::nullopt;
        const auto end = std::min(offset + batch_size, std::int64_t{6});
        ClassificationEvaluationBatch batch{
            .invocation = {.positional = {x.slice(0, offset, end)}},
            .targets = y.slice(0, offset, end),
            .source_indices = {sources.begin() + offset,
                               sources.begin() + end}};
        offset = end;
        return batch;
      },
      {.num_classes = 3,
       .ignore_index = -1,
       .sources = {"camera-a", "camera-b", "empty"}});
}

void check(bool cuda) {
  const auto whole = evaluate(cuda, 6), split = evaluate(cuda, 2);
  EXPECT_EQ(whole.batches, 1);
  EXPECT_EQ(split.batches, 3);
  EXPECT_EQ(whole.rows_seen, 6);
  EXPECT_EQ(whole.overall.samples, 5);
  const std::vector<std::vector<std::int64_t>> expected{
      {1, 0, 0}, {1, 1, 1}, {1, 0, 0}};
  EXPECT_EQ(whole.overall.confusion, expected);
  EXPECT_EQ(split.overall.confusion, expected);
  EXPECT_DOUBLE_EQ(*whole.overall.accuracy, 2.0 / 5);
  EXPECT_NEAR(*whole.overall.macro_precision, 4.0 / 9, 1e-15);
  EXPECT_NEAR(*whole.overall.macro_recall, 4.0 / 9, 1e-15);
  EXPECT_NEAR(*whole.overall.macro_f1, 1.0 / 3, 1e-15);
  EXPECT_EQ(whole.overall.classes[1].support, 3);
  EXPECT_EQ(whole.overall.classes[0].predicted, 3);
  EXPECT_EQ(whole.by_source[0].samples, 3);
  EXPECT_EQ(whole.by_source[1].samples, 2);
  EXPECT_FALSE(whole.by_source[2].accuracy);
  EXPECT_EQ(whole.observations.size(), 16);
  for (std::size_t i = 0; i < 3; ++i)
    EXPECT_EQ(whole.by_source[i].confusion, split.by_source[i].confusion);
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(whole.observations[i].observation.value,
              split.observations[i].observation.value);
    EXPECT_EQ(whole.observations[i].sample_count,
              split.observations[i].sample_count);
  }
  EXPECT_EQ(whole.observations.back().observation.status,
            TelemetryValueStatus::missing);
}

TEST(ClassificationEvaluation, ActualForwardCountsSlicesAndPartitionCPU) {
  check(false);
}
TEST(ClassificationEvaluation, ActualForwardCountsSlicesAndPartitionCUDA) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  check(true);
}

TEST(ClassificationEvaluation,
     RestoresMixedModesAndGradientsAfterSuccessAndError) {
  auto model = identity();
  auto& root = model.state().module();
  root.train();
  root.children().front()->eval();
  auto p = root.register_parameter("p", torch::ones({1}));
  p.mutable_grad() = torch::full({1}, 7.);
  bool first = true;
  const auto source =
      [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
    if (!first) return std::nullopt;
    first = false;
    return ClassificationEvaluationBatch{
        .invocation = {.positional = {logits()}}, .targets = labels()};
  };
  const bool grad = torch::GradMode::is_enabled();
  evaluate_classification(model, source,
                          {.num_classes = 3, .ignore_index = -1});
  EXPECT_TRUE(root.is_training());
  EXPECT_FALSE(root.children().front()->is_training());
  EXPECT_EQ(torch::GradMode::is_enabled(), grad);
  EXPECT_EQ(p.grad().item<float>(), 7.);
  EXPECT_THROW(
      evaluate_classification(
          model,
          [](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
            throw std::runtime_error("source");
          }),
      std::runtime_error);
  EXPECT_TRUE(root.is_training());
  EXPECT_FALSE(root.children().front()->is_training());
  EXPECT_EQ(torch::GradMode::is_enabled(), grad);
}

TEST(ClassificationEvaluation, EmptyAndAllIgnoredStayMissing) {
  auto model = identity();
  const auto empty = evaluate_classification(
      model,
      [](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
        return std::nullopt;
      });
  EXPECT_EQ(empty.batches, 0);
  EXPECT_FALSE(empty.overall.accuracy);
  EXPECT_EQ(empty.observations[0].sample_count, 0);
  bool first = true;
  const auto ignored = evaluate_classification(
      model,
      [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
        if (!first) return std::nullopt;
        first = false;
        return ClassificationEvaluationBatch{
            .invocation = {.positional = {torch::zeros({3, 2})}},
            .targets = torch::full({3}, -1, torch::kInt64)};
      },
      {.ignore_index = -1, .zero_division = metrics::ZeroDivisionPolicy::one});
  EXPECT_EQ(ignored.rows_seen, 3);
  EXPECT_EQ(ignored.overall.samples, 0);
  EXPECT_FALSE(ignored.overall.macro_f1);
}

TEST(ClassificationEvaluation, ZeroDivisionPoliciesIncludeAbsentClasses) {
  auto model = identity();
  for (auto policy :
       {metrics::ZeroDivisionPolicy::zero, metrics::ZeroDivisionPolicy::one,
        metrics::ZeroDivisionPolicy::error}) {
    bool first = true;
    const auto source =
        [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
      if (!first) return std::nullopt;
      first = false;
      return ClassificationEvaluationBatch{
          .invocation = {.positional = {torch::tensor({{1., 0.}})}},
          .targets = torch::zeros({1}, torch::kInt64)};
    };
    if (policy == metrics::ZeroDivisionPolicy::error)
      EXPECT_THROW(
          evaluate_classification(model, source, {.zero_division = policy}),
          NumericalError);
    else {
      const auto result =
          evaluate_classification(model, source, {.zero_division = policy});
      EXPECT_DOUBLE_EQ(*result.overall.macro_f1,
                       policy == metrics::ZeroDivisionPolicy::one ? 1. : .5);
    }
  }
}

TEST(ClassificationEvaluation, RejectsMalformedTargetsOutputsAndSlices) {
  auto model = identity();
  const auto run = [&](Tensor x, Tensor y, std::vector<std::size_t> ids,
                       std::vector<std::string> names) {
    bool first = true;
    return evaluate_classification(
        model,
        [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
          if (!first) return std::nullopt;
          first = false;
          return ClassificationEvaluationBatch{
              .invocation = {.positional = {x}},
              .targets = y,
              .source_indices = ids};
        },
        {.num_classes = 3, .ignore_index = -1, .sources = names});
  };
  EXPECT_THROW(run(logits(), labels().to(torch::kFloat), {}, {}),
               InvalidArgument);
  EXPECT_THROW(run(logits(), labels().reshape({2, 3}), {}, {}),
               InvalidArgument);
  EXPECT_THROW(run(logits().slice(1, 0, 2), labels(), {}, {}), InvalidArgument);
  EXPECT_THROW(run(logits(), torch::full({6}, 4, torch::kInt64), {}, {}),
               InvalidArgument);
  EXPECT_THROW(run(logits(), labels(), {0}, {}), InvalidArgument);
  EXPECT_THROW(run(logits(), labels(), {}, {"a"}), InvalidArgument);
  EXPECT_THROW(run(logits(), labels(), {0, 0, 0, 0, 0, 1}, {"a"}),
               InvalidArgument);
  auto x = logits();
  x[0][0] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(run(x, labels(), {}, {}), NumericalError);
}

TEST(ClassificationEvaluation, LimitsAndOptionsFailWithoutForward) {
  auto model = identity();
  int calls = 0;
  const auto source =
      [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
    ++calls;
    return ClassificationEvaluationBatch{
        .invocation = {.positional = {logits()}}, .targets = labels()};
  };
  EXPECT_THROW(evaluate_classification(model, source, {.num_classes = 0}),
               InvalidArgument);
  EXPECT_THROW(evaluate_classification(model, source, {.sources = {"a", "a"}}),
               InvalidArgument);
  EXPECT_THROW(
      evaluate_classification(model, source,
                              {.num_classes = 3, .max_confusion_bytes = 287}),
      ResourceExhausted);
  EXPECT_EQ(calls, 0);
  EXPECT_THROW(evaluate_classification(
                   model, source, {.num_classes = 3, .max_batch_samples = 5}),
               ResourceExhausted);
  EXPECT_EQ(calls, 1);
  EXPECT_THROW(evaluate_classification(
                   model, source,
                   {.num_classes = 3, .ignore_index = -1, .max_samples = 5}),
               ResourceExhausted);
  EXPECT_THROW(evaluate_classification(
                   model, source,
                   {.num_classes = 3, .ignore_index = -1, .max_batches = 1}),
               ResourceExhausted);
  EXPECT_THROW(evaluate_classification(model, {}), InvalidArgument);
}

TEST(ClassificationEvaluation,
     CancellationAndDeadlineNeverReturnPartialEvidence) {
  auto model = identity();
  std::stop_source stop;
  stop.request_stop();
  int calls = 0;
  const auto source =
      [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
    ++calls;
    return std::nullopt;
  };
  EXPECT_THROW(evaluate_classification(model, source, {}, stop.get_token()),
               Cancelled);
  EXPECT_THROW(evaluate_classification(model, source, {}, {},
                                       std::chrono::steady_clock::now()),
               DeadlineExceeded);
  EXPECT_EQ(calls, 0);
  std::stop_source later;
  EXPECT_THROW(
      evaluate_classification(
          model,
          [&](std::stop_token) -> std::optional<ClassificationEvaluationBatch> {
            later.request_stop();
            return std::nullopt;
          },
          {}, later.get_token()),
      Cancelled);
}

#if defined(JYPPX_LIUDING_EVALUATION_REPORT_TEST)
TEST(ClassificationEvaluation, ActualPredictionToCanonicalQualityReport) {
  const auto result = evaluate(false, 2);
  experiment::EvaluationSuite suite{
      .suite_id = "classification.fixture",
      .task = "classification",
      .dataset_digest = {.hex = std::string(64, 'd')},
      .environment_id = "fixture.cpu.fp32"};
  for (std::size_t i = 0; i < result.observations.size(); ++i) {
    const auto& point = result.observations[i];
    suite.rules.push_back({.rule_id = "r" + std::to_string(i),
                           .definition = point.definition,
                           .context = point.context,
                           .minimum_samples = 1,
                           .minimum = 0.});
  }
  experiment::EvaluationIdentity id{
      .evaluation_id = "eval.fixture",
      .run_id = "run.fixture",
      .model_digest = {.hex = std::string(64, 'a')},
      .build_fingerprint = "test-build",
      .adapter_id = "training.classification.v1",
      .config_digest = {.hex = std::string(64, 'b')}};
  const auto report = experiment::run_evaluation(
      suite, id, [&](const auto&, auto, auto) { return result.observations; });
  EXPECT_EQ(report.verdict, experiment::QualityVerdict::incomplete);
  EXPECT_EQ(report.results.front().verdict, experiment::QualityVerdict::pass);
  const auto json = experiment::quality_report_json(report);
  EXPECT_EQ(experiment::quality_report_json(
                experiment::parse_quality_report_json(json)),
            json);
}
#endif
}  // namespace
