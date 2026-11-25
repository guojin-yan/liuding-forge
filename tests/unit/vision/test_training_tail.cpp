// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/data_image/image_encoder.hpp>
#include <jyppx/liuding/vision/tasks/data/inspection.hpp>
#include <jyppx/liuding/vision/tasks/workflow/suite.hpp>

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
using J = nlohmann::json;
class VisionTrainingTail : public ::testing::Test {
 protected:
  std::filesystem::path root;
  void SetUp() override {
    torch::set_num_threads(1);
    root = std::filesystem::temp_directory_path() /
           ("liuding-training-tail-" +
            std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    for (const auto* split : {"train", "val"}) {
      const int count = std::string_view(split) == "train" ? 5 : 3;
      auto directory = root / split;
      std::filesystem::create_directory(directory);
      J annotations{{"images", J::array()},
                    {"annotations", J::array()},
                    {"categories", {{{"id", 1}, {"name", "object"}}}}};
      for (int i = 0; i < count; ++i) {
        const auto file = std::to_string(i) + ".png";
        const auto pixels = torch::full({3, 64, 64}, 70 + i, torch::kUInt8);
        im::ImageEncodeOptions options;
        options.format = im::ImageEncodeFormat::png;
        im::write_image_file(directory / file, pixels, {}, options);
        annotations["images"].push_back({{"id", i + 1},
                                         {"file_name", file},
                                         {"width", 64},
                                         {"height", 64}});
        annotations["annotations"].push_back({{"id", i + 1},
                                              {"image_id", i + 1},
                                              {"category_id", 1},
                                              {"bbox", {8, 8, 32, 32}},
                                              {"area", 1024},
                                              {"iscrowd", 0}});
      }
      std::ofstream(root / (std::string(split) + ".json")) << annotations;
    }
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }
  t::RecipeSpec Recipe() {
    t::RecipeSpec r;
    r.model.architecture = "yolov8n";
    r.model.num_classes = r.data.num_classes = 1;
    r.training.source = root / "train.json";
    r.training.image_root = root / "train";
    r.validation = r.training;
    r.validation->source = root / "val.json";
    r.validation->image_root = root / "val";
    r.validation->split = t::DatasetSplit::validation;
    r.epochs = 1;
    r.batch_size = 2;
    r.data.letterbox.canvas = {64, 64};
    r.ema = false;
    r.full_validation_metrics = false;
    r.verify_image_contents = false;
    r.scheduler.warmup_updates = 0;
    r.scheduler.final_factor = 1;
    r.early_stopping_patience = 0;
    r.save_interval_epochs = 0;
    return r;
  }
};
TEST_F(VisionTrainingTail, DropsOnlyTrainingTailAndKeepsValidationTail) {
  for (const auto drop : {false, true}) {
    SCOPED_TRACE(drop);
    auto recipe = Recipe();
    recipe.drop_last_training_batch = drop;
    std::size_t training = 0, validation = 0;
    t::TrainRequest request;
    request.recipe = recipe;
    request.output = root / (drop ? "drop" : "keep");
    request.event_observer = [&](const training::CallbackEvent& event) {
      if (event.kind != training::CallbackEventKind::batch_finished) return;
      if (event.phase == training::CallbackPhase::training) ++training;
      if (event.phase == training::CallbackPhase::validation) ++validation;
    };
    auto result = t::train_vision(request);
    EXPECT_EQ(result.fit.completed_epochs, 1U);
    EXPECT_EQ(result.fit.global_step, drop ? 2U : 3U);
    EXPECT_EQ(training, drop ? 2U : 3U);
    EXPECT_EQ(validation, 2U);
  }
}
TEST_F(VisionTrainingTail, RejectsDroppingTheEntireTrainingDataset) {
  auto recipe = Recipe();
  recipe.batch_size = 6;
  recipe.drop_last_training_batch = true;
  t::TrainRequest request;
  request.recipe = recipe;
  request.output = root / "empty";
  EXPECT_THROW(t::train_vision(request), InvalidArgument);
  EXPECT_FALSE(std::filesystem::exists(request.output));
}
TEST_F(VisionTrainingTail, JpegBackendBindsRecipeDatasetAndDecodeCache) {
  auto recipe = Recipe();
  recipe.data.augmentation.multi_source.total_epochs = recipe.epochs;
  recipe.data.augmentation.multi_source.close_epochs = 0;
  const auto original = t::recipe_json(recipe);
  const auto identity = t::recipe_identity(recipe);
  EXPECT_EQ(original.find("jpeg_backend"), std::string::npos);
  recipe.jpeg_backend = im::JpegDecodeBackend::libjpeg_turbo;
  if (!im::jpeg_decode_backend_available(recipe.jpeg_backend)) {
    EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
    return;
  }
  const auto roundtrip = t::parse_recipe(t::recipe_json(recipe), root);
  EXPECT_EQ(roundtrip.jpeg_backend, recipe.jpeg_backend);
  EXPECT_EQ(t::recipe_identity(roundtrip), t::recipe_identity(recipe));
  EXPECT_NE(identity, t::recipe_identity(recipe));
  const auto accelerated = t::load_vision_data(recipe);
  recipe.jpeg_backend = im::JpegDecodeBackend::stb;
  EXPECT_EQ(t::recipe_json(recipe), original);
  const auto standard = t::load_vision_data(recipe);
  EXPECT_NE(standard.training_identity, accelerated.training_identity);
  EXPECT_NE(standard.validation_identity, accelerated.validation_identity);
  // PNG still uses the same decoder under both explicit JPEG policies.
  EXPECT_TRUE(torch::equal(standard.training->get(0).image,
                           accelerated.training->get(0).image));
  const auto jpeg = std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                    "data_image/jpeg_reference/baseline.jpg";
  t::ImageDecodeCache cache({.ram_bytes = 1U << 20U});
  im::ImageDecodeOptions o;
  o.channels = im::ImageChannels::rgb;
  const auto stb = cache.read(jpeg, o);
  o.jpeg_backend = im::JpegDecodeBackend::libjpeg_turbo;
  const auto turbo = cache.read(jpeg, o);
  EXPECT_EQ(cache.stats().misses, 2U);
  EXPECT_EQ(cache.stats().ram_hits, 0U);
  EXPECT_TRUE(torch::equal(turbo.tensor, cache.read(jpeg, o).tensor));
  EXPECT_EQ(cache.stats().ram_hits, 1U);
  EXPECT_EQ(stb.tensor.sizes(), turbo.tensor.sizes());
}
TEST_F(VisionTrainingTail, RejectsUnknownJpegBackendValues) {
  for (const auto* value : {"null", "1", "true", "[]", "\"auto\""}) {
    const auto json =
        std::string(
            R"({"model":{"architecture":"yolov8n"},"dataset":{"format":"coco","source":"a.json"},"jpeg_backend":)") +
        value + "}";
    EXPECT_THROW(t::parse_recipe(json, root), InvalidArgument);
  }
}
TEST_F(VisionTrainingTail,
       RtDetrInferenceValidationPublishesMetricsWithoutLoss) {
  auto recipe = Recipe();
  recipe.model.architecture = "rtdetr_resnet18_vd_coco72";
  recipe.data.letterbox.canvas = {256, 256};
  recipe.data.rtdetr.enabled = true;
  recipe.data.rtdetr.photometric_probability = 0;
  recipe.data.rtdetr.zoom_out_probability = 0;
  recipe.data.rtdetr.iou_crop_probability = 0;
  recipe.data.rtdetr.horizontal_flip_probability = 0;
  recipe.rtdetr_inference_validation = true;
  recipe.full_validation_metrics = true;
  recipe.evaluation = {
      .confidence = 0, .multi_label = true, .clip_boxes = false};
  recipe.epochs = 2;
  recipe.ema = true;
  recipe.drop_last_training_batch = true;
  std::size_t batches = 0, completed = 0;
  t::TrainRequest request;
  request.recipe = recipe;
  request.output = root / "rtdetr-inference";
  request.event_observer = [&](const training::CallbackEvent& event) {
    if (event.phase != training::CallbackPhase::validation) return;
    EXPECT_FALSE(event.metrics.contains("validation.loss"));
    if (event.kind == training::CallbackEventKind::batch_finished) ++batches;
    if (event.kind == training::CallbackEventKind::validation_finished) {
      ++completed;
      EXPECT_TRUE(event.metrics.contains("validation.map"));
      EXPECT_TRUE(event.metrics.contains("validation.ap50"));
    }
  };
  const auto result = t::train_vision(request);
  EXPECT_EQ(result.fit.completed_epochs, 2U);
  EXPECT_EQ(result.fit.global_step, 4U);
  EXPECT_EQ(batches, 4U);
  EXPECT_EQ(completed, 2U);
  J report;
  std::ifstream(request.output / "validation.json") >> report;
  EXPECT_EQ(report["execution"], "rtdetr.batched_inference/v1");
  EXPECT_EQ(report["images"], 3);
  EXPECT_EQ(report["metric_max_detections_per_image_class"], 100);
}
TEST_F(VisionTrainingTail,
       BestArtifactPolicyPreservesPeriodicAndLastCheckpoints) {
  auto recipe = Recipe();
  recipe.save_interval_epochs = 1;
  recipe.data.augmentation.multi_source.total_epochs = recipe.epochs;
  recipe.data.augmentation.multi_source.close_epochs = 0;
  const auto original = t::recipe_json(recipe), id = t::recipe_identity(recipe);
  EXPECT_EQ(original.find("save_best_checkpoint"), std::string::npos);
  recipe.save_best_checkpoint = false;
  const auto restored = t::parse_recipe(t::recipe_json(recipe), root);
  EXPECT_FALSE(restored.save_best_checkpoint);
  EXPECT_EQ(t::recipe_identity(restored), t::recipe_identity(recipe));
  EXPECT_NE(id, t::recipe_identity(recipe));
  for (bool best : {false, true}) {
    recipe.save_best_checkpoint = best;
    t::TrainRequest request;
    request.recipe = recipe;
    request.output = root / (best ? "with-best" : "without-best");
    const auto result = t::train_vision(request);
    EXPECT_EQ(result.fit.completed_epochs, 1U);
    EXPECT_TRUE(
        std::filesystem::is_regular_file(request.output / "last.checkpoint"));
    EXPECT_TRUE(std::filesystem::is_regular_file(request.output /
                                                 "epoch-1.checkpoint"));
    EXPECT_EQ(std::filesystem::exists(request.output / "best.checkpoint"),
              best);
    EXPECT_EQ(std::filesystem::exists(request.output / "best.safetensors"),
              best);
  }
  auto malformed = J::parse(original);
  malformed["save_best_checkpoint"] = 0;
  EXPECT_THROW(t::parse_recipe(malformed.dump(), root), InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
