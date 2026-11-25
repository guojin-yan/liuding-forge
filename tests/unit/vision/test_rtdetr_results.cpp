// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/vision/tasks/rtdetr.hpp>
#include <jyppx/liuding/vision/tasks/workflow/recipe.hpp>

#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
using J = nlohmann::json;
Tensor Matrix(const J& value) {
  std::vector<float> values;
  for (const auto& row : value)
    for (const auto& v : row) values.push_back(v.get<float>());
  return torch::tensor(values).view(
      {static_cast<std::int64_t>(value.size()), -1});
}
TEST(RtDetrResults, OriginalTopkAndUnclippedRestorationMatchOfficialReference) {
  torch::set_num_threads(1);
  J reference;
  std::ifstream(std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                "data_image/rtdetr_results.json") >>
      reference;
  std::vector<Tensor> logits, boxes;
  for (const auto& batch : reference["logits"]) logits.push_back(Matrix(batch));
  for (const auto& batch : reference["normalized_cxcywh"])
    boxes.push_back(Matrix(batch));
  std::vector<Tensor> raw(8);
  raw[0] = torch::stack(logits).unsqueeze(0);
  raw[1] = torch::stack(boxes).unsqueeze(0);
  t::ResultOptions options;
  options.confidence = 0;
  options.max_detections = 4;
  options.multi_label = true;
  options.clip_boxes = false;
  const auto actual = t::decode_rtdetr(raw, {640, 640}, options);
  ASSERT_EQ(actual.size(), 2U);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto width = reference["sizes_wh"][i][0].get<std::uint32_t>(),
               height = reference["sizes_wh"][i][1].get<std::uint32_t>();
    const auto trace = im::GeometryTrace({width, height}).resized({640, 640});
    const auto restored = t::restore_result(actual[i], trace, 67108864, false);
    const auto& expected = reference["expected"][i];
    EXPECT_TRUE(
        torch::allclose(restored.boxes, Matrix(expected["boxes"]), 1e-6, 2e-6));
    EXPECT_TRUE(torch::equal(
        restored.classes,
        torch::tensor(expected["labels"].get<std::vector<std::int64_t>>())));
    EXPECT_TRUE(torch::allclose(
        restored.scores,
        torch::tensor(expected["scores"].get<std::vector<float>>()), 1e-6,
        1e-7));
    EXPECT_EQ(restored.extent.width, width);
    EXPECT_EQ(restored.extent.height, height);
    EXPECT_EQ(restored.boxes.size(0), 4);
    EXPECT_TRUE((restored.boxes.select(1, 0) < 0).any().item<bool>() ||
                (restored.boxes.select(1, 1) < 0).any().item<bool>());
    const auto clipped = t::restore_result(actual[i], trace);
    EXPECT_GE(clipped.boxes.min().item<float>(), 0);
    EXPECT_FALSE(torch::equal(clipped.boxes, restored.boxes));
  }
  const auto padded = im::GeometryTrace({320, 320}).padded(0, 0, 320, 320);
  EXPECT_THROW(t::restore_result(actual[0], padded, 67108864, false),
               InvalidArgument);
  const auto cropped =
      im::GeometryTrace({800, 800}).cropped(80, 80, {640, 640});
  EXPECT_THROW(t::restore_result(actual[0], cropped, 67108864, false),
               InvalidArgument);
  const auto flipped = im::GeometryTrace({640, 640}).flipped_horizontal();
  EXPECT_THROW(t::restore_result(actual[0], flipped, 67108864, false),
               InvalidArgument);
  options.clip_boxes = true;
  for (const auto& result : t::decode_rtdetr(raw, {640, 640}, options)) {
    EXPECT_GE(result.boxes.min().item<float>(), 0);
    EXPECT_LE(result.boxes.max().item<float>(), 640);
  }
}
TEST(RtDetrResults,
     RecipeBindsExplicitClippingPolicyAndRejectsUnsupportedGeometry) {
  t::RecipeSpec recipe;
  recipe.model.architecture = "rtdetr_resnet18_vd_coco72";
  recipe.training.source = std::filesystem::absolute("reference.json");
  recipe.data.rtdetr.enabled = true;
  const auto original = t::recipe_json(recipe),
             identity = t::recipe_identity(recipe);
  EXPECT_EQ(original.find("clip_boxes"), std::string::npos);
  recipe.evaluation.clip_boxes = recipe.prediction.clip_boxes = false;
  const auto restored =
      t::parse_recipe(t::recipe_json(recipe), std::filesystem::current_path());
  EXPECT_FALSE(restored.evaluation.clip_boxes);
  EXPECT_FALSE(restored.prediction.clip_boxes);
  EXPECT_EQ(t::recipe_identity(recipe), t::recipe_identity(restored));
  EXPECT_NE(identity, t::recipe_identity(recipe));
  recipe.data.rtdetr.enabled = false;
  EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
  recipe.data.rtdetr.enabled = true;
  recipe.model.architecture = "yolov8n";
  EXPECT_THROW(t::check_recipe(recipe), InvalidArgument);
  EXPECT_THROW(
      t::parse_recipe(
          R"({"model":{"architecture":"rtdetr_resnet18_vd_coco72"},"dataset":{"format":"coco","source":"ref.json"},"rtdetr_data":{},"evaluation":{"clip_boxes":1}})",
          std::filesystem::current_path()),
      InvalidArgument);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
