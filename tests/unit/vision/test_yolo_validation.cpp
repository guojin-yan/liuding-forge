// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../../src/jyppx/liuding/vision/tasks/workflow/yolo_validation.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
using J = nlohmann::json;
t::RecipeSpec Recipe() {
  t::RecipeSpec r;
  r.model.architecture = "yolo26n";
  r.training.source = std::filesystem::absolute("reference.json");
  r.validation = r.training;
  r.model.num_classes = r.data.num_classes = 1;
  r.data.letterbox.canvas = {64, 64};
  r.yolo_detection_validation = true;
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
                     std::size_t end, const t::RecipeSpec& recipe) {
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
  auto options = t::yolo_evaluation_options(recipe.data);
  options.augmentation.dataset_id = "original-split";
  return t::prepare_yolo_batch(samples, keys, options);
}
t::TaskBinding Binding(std::size_t& forwards, std::size_t& losses,
                       bool false_positives = false) {
  auto module = std::make_shared<torch::nn::Module>();
  module->eval();
  nn::InvocationSchema schema;
  schema.positional = {ValueSchema::tensor()};
  auto model = nn::adapt_module(
      module, nn::ModelInvoker([&forwards, module](const nn::Invocation& in) {
        EXPECT_FALSE(module->is_training());
        ++forwards;
        return in.positional.front();
      }),
      schema, ValueSchema::tensor());
  t::TaskBinding binding;
  binding.model = std::make_shared<nn::ModelAdapter>(std::move(model));
  binding.step = [&losses,
                  expected = binding.model](const training::StepInput& in) {
    // The trainer input may carry the raw training model; use the EMA binding.
    EXPECT_EQ(&in.model, expected.get());
    training::StepOutput output;
    output.prediction = in.model.invoke(in.invocation);
    ++losses;
    output.loss = torch::tensor(2.5);
    output.metrics["box_loss"] = 1.25;
    output.sample_count = in.invocation.positional.front().toTensor().size(0);
    return output;
  };
  binding.decode = [false_positives](const c10::IValue& value, auto canvas,
                                     const t::ResultOptions& options) {
    EXPECT_EQ(canvas[0], 64);
    EXPECT_EQ(canvas[1], 64);
    EXPECT_TRUE(options.clip_boxes);
    std::vector<t::ImageResult> results;
    for (std::int64_t i = 0; i < value.toTensor().size(0); ++i) {
      t::ImageResult result;
      result.extent = {64, 64};
      // LetterBox adds 16 vertical pixels. The higher-score crowd is ignored.
      result.boxes =
          torch::tensor({{40.f, 16.f, 64.f, 48.f}, {8.f, 24.f, 40.f, 40.f}});
      result.scores = torch::tensor({.95f, .9f});
      result.classes = torch::zeros({2}, torch::kInt64);
      if (false_positives) {
        result.boxes = torch::cat(
            {torch::tensor({{0.f, 16.f, 2.f, 18.f}}).repeat({100, 1}),
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
TEST(YoloDetectionValidation, ReusesLossPredictionAndKeepsTailCrowdAndEpochs) {
  torch::set_num_threads(1);
  torch::NoGradGuard guard;
  const auto recipe = Recipe();
  const auto source = Sources();
  t::detail::YoloDetectionValidation validation(source, "original-split",
                                                recipe);
  std::size_t forwards = 0, losses = 0;
  auto binding = Binding(forwards, losses);
  std::size_t raw_calls = 0, raw_losses = 0;
  auto raw = Binding(raw_calls, raw_losses);
  for (std::uint64_t epoch = 0; epoch < 2; ++epoch) {
    for (std::size_t begin = 0; begin < 3; begin += 2) {
      auto batch =
          Batch(*source, begin, std::min(begin + 2, std::size_t{3}), recipe);
      // Metrics must use source crowd/area, independent of loss annotations.
      for (auto& item : batch.items) item.annotation.reset();
      data::BatchEnvelope envelope;
      envelope.value = im::image_batch_value(batch);
      nn::Invocation invocation{{batch.images}, {}};
      std::optional<c10::IValue> target;
      training::StepContext context{.phase = training::StepPhase::validation,
                                    .epoch = epoch};
      const auto output = validation.step(
          binding, {*raw.model, envelope, invocation, target, context});
      EXPECT_DOUBLE_EQ(output.loss.item<double>(), 2.5);
      EXPECT_DOUBLE_EQ(output.metrics.at("box_loss"), 1.25);
      ASSERT_TRUE(output.prediction);
      EXPECT_TRUE(
          torch::equal(output.prediction->value.toTensor(), batch.images));
      EXPECT_EQ(output.sample_count, batch.items.size());
    }
    const auto report = J::parse(validation.finish({}));
    EXPECT_EQ(report["execution"], "yolo.batched_loss_and_inference/v1");
    EXPECT_EQ(report["images"], 3);
    EXPECT_DOUBLE_EQ(report["map"].get<double>(), 1);
    EXPECT_DOUBLE_EQ(report["ap50"].get<double>(), 1);
    EXPECT_EQ(report["coco_metrics"].size(), 12U);
    EXPECT_DOUBLE_EQ(report["coco_metrics"]["AP75"].get<double>(), 1);
    EXPECT_DOUBLE_EQ(report["coco_metrics"]["AR1"].get<double>(), 0);
    EXPECT_DOUBLE_EQ(report["coco_metrics"]["AR10"].get<double>(), 1);
    EXPECT_DOUBLE_EQ(report["coco_metrics"]["AP_small"].get<double>(), 1);
    EXPECT_DOUBLE_EQ(report["coco_metrics"]["AP_medium"].get<double>(), -1);
    EXPECT_EQ(forwards, 2 * (epoch + 1));
    EXPECT_EQ(losses, forwards);
    EXPECT_EQ(raw_calls, 0U);
  }
}
TEST(YoloDetectionValidation, BoundsCocoDetectionsAndRejectsInvalidSteps) {
  torch::set_num_threads(1);
  const auto recipe = Recipe();
  const auto source = Sources();
  t::detail::YoloDetectionValidation validation(source, "original-split",
                                                recipe);
  std::size_t forwards = 0, losses = 0;
  auto binding = Binding(forwards, losses, true);
  auto batch = Batch(*source, 0, 3, recipe);
  data::BatchEnvelope envelope;
  envelope.value = im::image_batch_value(batch);
  nn::Invocation invocation{{batch.images}, {}};
  std::optional<c10::IValue> target;
  training::StepContext context{.phase = training::StepPhase::validation};
  const training::StepInput input{*binding.model, envelope, invocation, target,
                                  context};
  validation.step(binding, input);
  EXPECT_DOUBLE_EQ(J::parse(validation.finish({}))["map"].get<double>(), 0);
  t::detail::YoloDetectionValidation partial(source, "original-split", recipe);
  EXPECT_THROW(partial.finish({}), InvalidArgument);
  const auto step = binding.step;
  for (bool missing : {true, false}) {
    binding.step = [step, missing](const training::StepInput& in) {
      auto output = step(in);
      if (missing)
        output.prediction.reset();
      else
        ++output.sample_count;
      return output;
    };
    EXPECT_THROW(partial.step(binding, input), InvalidArgument);
  }
  context.phase = training::StepPhase::training;
  EXPECT_THROW(partial.step(binding, input), InvalidArgument);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(validation.finish(stop.get_token()), Cancelled);
  source->images[1].source_image_id = source->images[0].source_image_id;
  EXPECT_THROW(
      t::detail::YoloDetectionValidation(source, "original-split", recipe),
      InvalidArgument);
}
TEST(YoloDetectionValidation,
     RecipeBindsWorkloadAndRejectsUnsupportedPolicies) {
  auto r = Recipe();
  const auto json = t::recipe_json(r), id = t::recipe_identity(r);
  const auto restored = t::parse_recipe(json, std::filesystem::current_path());
  EXPECT_TRUE(restored.yolo_detection_validation);
  EXPECT_EQ(id, t::recipe_identity(restored));
  r.selection_metric = "validation.loss";
  EXPECT_NO_THROW(t::check_recipe(r));
  r = Recipe();
  r.yolo_detection_validation = false;
  EXPECT_EQ(t::recipe_json(r).find("yolo_detection_validation"),
            std::string::npos);
  EXPECT_NE(id, t::recipe_identity(r));
  for (int i = 0; i < 11; ++i) {
    r = Recipe();
    if (i == 0) r.full_validation_metrics = false;
    if (i == 1) r.evaluation.classes = {0};
    if (i == 2) r.model.architecture = "yolov8n";
    if (i == 3) r.rtdetr_inference_validation = true;
    if (i == 4) r.model.task = t::TaskKind::pose;
    if (i == 5) r.validation.reset();
    if (i == 6) r.validation->format = t::DetectionDatasetFormat::yolo;
    if (i == 7) r.exif_policy = t::ExifPolicy::encoded_coordinates;
    if (i == 8) r.extension.validation_manifest = "other.json";
    if (i == 9) r.extension.training_manifest = "other.json";
    if (i == 10) r.data.rtdetr.enabled = true;
    EXPECT_THROW(t::check_recipe(r), InvalidArgument) << i;
  }
  auto malformed = J::parse(json);
  malformed["yolo_detection_validation"] = 1;
  EXPECT_THROW(
      t::parse_recipe(malformed.dump(), std::filesystem::current_path()),
      InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
