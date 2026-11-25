// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../../src/jyppx/liuding/vision/tasks/workflow/rtdetr_validation.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
using J = nlohmann::json;
t::RecipeSpec Recipe() {
  t::RecipeSpec r;
  r.model.architecture = "rtdetr_resnet18_vd_coco72";
  r.training.source = std::filesystem::absolute("reference.json");
  r.model.num_classes = r.data.num_classes = 1;
  r.data.rtdetr.enabled = true;
  r.data.letterbox.canvas = {256, 256};
  r.evaluation = {.confidence = 0, .multi_label = true, .clip_boxes = false};
  r.rtdetr_inference_validation = true;
  return r;
}
std::shared_ptr<im::AnnotationIndex> Sources() {
  auto source = std::make_shared<im::AnnotationIndex>();
  source->categories = {{.source_id = 1, .name = "object"}};
  for (int i = 0; i < 3; ++i) {
    im::VisionAnnotationRecord record;
    record.image_path = std::to_string(i) + ".png";
    record.source_image_id = 10 + i;
    record.width = 64;
    record.height = 32;
    record.objects = {
        {.source_id = 1, .box = im::ImageBox{8, 8, 40, 24}, .area = 400},
        {.source_id = 2,
         .box = im::ImageBox{40, 0, 64, 32},
         .area = 768,
         .crowd = true}};
    source->images.push_back(std::move(record));
  }
  return source;
}
im::ImageBatch Batch(const im::AnnotationIndex& source, std::size_t begin,
                     std::size_t end, const t::RecipeSpec& r) {
  std::vector<im::VisionSample> samples;
  std::vector<im::ImageBatchKey> keys;
  for (auto i = begin; i < end; ++i) {
    const auto& record = source.images.at(i);
    im::VisionSample sample;
    sample.image = torch::full({3, 32, 64}, 70, torch::kUInt8);
    sample.width = record.width;
    sample.height = record.height;
    sample.image_path = record.image_path;
    sample.categories = std::make_shared<const std::vector<im::VisionCategory>>(
        source.categories);
    sample.annotation = std::make_shared<im::VisionAnnotationRecord>(record);
    sample.geometry.emplace(im::ImageExtent{64, 32});
    keys.push_back({.id = record.image_path});
    samples.push_back(std::move(sample));
  }
  auto options = t::yolo_evaluation_options(r.data);
  options.augmentation.dataset_id = "original-split";
  return t::prepare_yolo_batch(samples, keys, options);
}
t::TaskBinding Binding(std::size_t& calls, bool false_positives = false) {
  auto module = std::make_shared<torch::nn::Module>();
  module->eval();
  nn::InvocationSchema schema;
  schema.positional = {ValueSchema::tensor(), ValueSchema::tensor()};
  auto model = nn::adapt_module(
      module, nn::ModelInvoker([&calls, module](const nn::Invocation& in) {
        EXPECT_FALSE(module->is_training());
        EXPECT_EQ(in.positional.at(1).toTensor().numel(), 0);
        ++calls;
        return in.positional.front();
      }),
      schema, ValueSchema::tensor());
  t::TaskBinding binding;
  binding.model = std::make_shared<nn::ModelAdapter>(std::move(model));
  binding.decode = [false_positives](const c10::IValue& value, auto canvas,
                                     const t::ResultOptions& options) {
    EXPECT_EQ(canvas[0], 256);
    EXPECT_EQ(canvas[1], 256);
    EXPECT_FALSE(options.clip_boxes);
    std::vector<t::ImageResult> results;
    for (std::int64_t i = 0; i < value.toTensor().size(0); ++i) {
      t::ImageResult result;
      result.extent = {256, 256};
      // Crowd prediction must be ignored using the original source GT.
      // The profile removes crowd from the transformed batch annotations.
      result.boxes = torch::tensor(
          {{160.f, 0.f, 256.f, 256.f}, {32.f, 64.f, 160.f, 192.f}});
      result.scores = torch::tensor({.95f, .9f});
      result.classes = torch::zeros({2}, torch::kInt64);
      if (false_positives) {
        result.boxes =
            torch::cat({torch::tensor({{0.f, 0.f, 8.f, 8.f}}).repeat({100, 1}),
                        result.boxes.slice(0, 1, 2)});
        result.scores = torch::cat(
            {torch::full({100}, .95f), result.scores.slice(0, 1, 2)});
        result.classes = torch::zeros({101}, torch::kInt64);
      }
      results.push_back(std::move(result));
    }
    return results;
  };
  return binding;
}
TEST(RtDetrValidation, OneForwardPerBatchKeepsTailAndSourceCrowdAcrossEpochs) {
  torch::set_num_threads(1);
  torch::NoGradGuard guard;
  const auto recipe = Recipe();
  const auto source = Sources();
  t::detail::RtDetrValidation validation(source, "original-split", recipe);
  std::size_t calls = 0;
  auto binding = Binding(calls);
  for (std::uint64_t epoch = 0; epoch < 2; ++epoch) {
    for (std::size_t begin = 0; begin < 3; begin += 2) {
      auto batch =
          Batch(*source, begin, std::min(begin + 2, std::size_t{3}), recipe);
      ASSERT_EQ(batch.items.front().annotation->objects.size(), 1U);
      data::BatchEnvelope envelope;
      envelope.value = im::image_batch_value(batch);
      nn::Invocation invocation{{batch.images, torch::ones({1, 6})}, {}};
      std::optional<c10::IValue> target;
      training::StepContext context{.phase = training::StepPhase::validation,
                                    .epoch = epoch};
      const auto output = validation.step(
          binding, {*binding.model, envelope, invocation, target, context});
      EXPECT_FALSE(output.loss.defined());
      EXPECT_FALSE(output.prediction.has_value());
      EXPECT_EQ(output.sample_count, batch.items.size());
    }
    const auto report = J::parse(validation.finish({}));
    EXPECT_EQ(report["images"], 3);
    EXPECT_DOUBLE_EQ(report["map"].get<double>(), 1);
    EXPECT_DOUBLE_EQ(report["ap50"].get<double>(), 1);
    EXPECT_EQ(report["metric_max_detections_per_image_class"], 100);
    EXPECT_EQ(calls, 2 * (epoch + 1));
  }
}
TEST(RtDetrValidation, MetricAppliesCocoMaxDet100AndRejectsIncompleteSplit) {
  torch::set_num_threads(1);
  const auto recipe = Recipe();
  const auto source = Sources();
  t::detail::RtDetrValidation validation(source, "original-split", recipe);
  std::size_t calls = 0;
  auto binding = Binding(calls, true);
  auto batch = Batch(*source, 0, 3, recipe);
  data::BatchEnvelope envelope;
  envelope.value = im::image_batch_value(batch);
  nn::Invocation invocation{{batch.images, torch::ones({1, 6})}, {}};
  std::optional<c10::IValue> target;
  training::StepContext context{.phase = training::StepPhase::validation};
  const auto input = training::StepInput{*binding.model, envelope, invocation,
                                         target, context};
  validation.step(binding, input);
  EXPECT_DOUBLE_EQ(J::parse(validation.finish({}))["map"].get<double>(), 0);
  t::detail::RtDetrValidation partial(source, "original-split", recipe);
  EXPECT_THROW(partial.finish({}), InvalidArgument);
  context.phase = training::StepPhase::training;
  EXPECT_THROW(partial.step(binding, input), InvalidArgument);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(validation.finish(stop.get_token()), Cancelled);
}
TEST(RtDetrValidation, RecipeBindsWorkloadAndRejectsIncompatiblePolicies) {
  auto r = Recipe();
  const auto json = t::recipe_json(r), id = t::recipe_identity(r);
  const auto restored = t::parse_recipe(json, std::filesystem::current_path());
  EXPECT_TRUE(restored.rtdetr_inference_validation);
  EXPECT_EQ(id, t::recipe_identity(restored));
  r.exif_policy = t::ExifPolicy::ignore;
  EXPECT_NO_THROW(t::check_recipe(r));
  r.exif_policy = t::ExifPolicy::reject_non_identity;
  r.rtdetr_inference_validation = false;
  EXPECT_EQ(t::recipe_json(r).find("rtdetr_inference_validation"),
            std::string::npos);
  EXPECT_NE(id, t::recipe_identity(r));
  for (int i = 0; i < 8; ++i) {
    r = Recipe();
    if (i == 0) r.full_validation_metrics = false;
    if (i == 1) r.evaluation.clip_boxes = true;
    if (i == 2) r.evaluation.max_detections = 100;
    if (i == 3) r.evaluation.confidence = .01;
    if (i == 4) r.selection_metric = "validation.loss";
    if (i == 5) r.evaluation.multi_label = false;
    if (i == 6) r.data.rtdetr.enabled = false;
    if (i == 7) r.exif_policy = t::ExifPolicy::encoded_coordinates;
    EXPECT_THROW(t::check_recipe(r), InvalidArgument) << i;
  }
  auto malformed = J::parse(json);
  malformed["rtdetr_inference_validation"] = 1;
  EXPECT_THROW(
      t::parse_recipe(malformed.dump(), std::filesystem::current_path()),
      InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
