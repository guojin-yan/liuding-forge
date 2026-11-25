// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/vision/tasks/data/opencv_preparation.hpp>

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace jyppx::liuding::vision::tests {
namespace {
namespace t = tasks;
namespace im = data::image;
using J = nlohmann::json;
J Reference() {
  std::ifstream stream(std::filesystem::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) /
                       "vision/yolo_opencv.json");
  if (!stream) throw std::runtime_error("missing independent OpenCV fixture");
  return J::parse(stream);
}
std::array<im::VisionSample, 4> Samples(const J& ref) {
  std::array<im::VisionSample, 4> out;
  auto categories = std::make_shared<const std::vector<im::VisionCategory>>(
      std::vector<im::VisionCategory>{{.source_id = 1, .name = "one"},
                                      {.source_id = 2, .name = "two"},
                                      {.source_id = 3, .name = "three"},
                                      {.source_id = 4, .name = "four"}});
  for (int i = 0; i < 4; ++i) {
    auto& s = out[i];
    int h = ref["inputs"][i]["shape"][0], w = ref["inputs"][i]["shape"][1];
    s.width = w;
    s.height = h;
    s.image = torch::empty({h, w, 3}, torch::kUInt8);
    auto* bytes = s.image.mutable_data_ptr<std::uint8_t>();
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int c = 0; c < 3; ++c)
          bytes[(y * w + x) * 3 + c] =
              std::uint8_t((x * 17 + y * 31 + (2 - c) * 63 + i * 47) % 256);
    s.pixels.layout = im::ImageLayout::hwc;
    s.image_path = "synthetic-input-" + std::to_string(i);
    s.geometry.emplace(im::ImageExtent{s.width, s.height});
    s.categories = categories;
    auto ann = std::make_shared<im::VisionAnnotationRecord>();
    ann->width = w;
    ann->height = h;
    ann->image_path = s.image_path;
    ann->source_image_id = i + 1;
    for (const auto& raw : ref["inputs"][i]["boxes"]) {
      ann->objects.push_back(
          {.source_id = std::int64_t(ann->objects.size() + 1),
           .category = std::size_t(i),
           .box = im::ImageBox{double(raw[0].get<float>()) * w,
                               double(raw[1].get<float>()) * h,
                               double(raw[2].get<float>()) * w,
                               double(raw[3].get<float>()) * h}});
    }
    s.annotation = std::move(ann);
  }
  return out;
}
t::YoloOpenCvGeometryPlan Plan(const J& ref, int index) {
  const auto& d = ref["geometry"][index ? "b" : "a"]["draws"];
  const auto& center = ref["mosaic_center_yx"][index];
  return {.size = 64,
          .mosaic = true,
          .mosaic_center = {center[1].get<std::uint32_t>(),
                            center[0].get<std::uint32_t>()},
          .perspective = true,
          .perspective_x = d[0],
          .perspective_y = d[1],
          .rotation_degrees = d[2],
          .scale = d[3],
          .shear_x_degrees = d[4],
          .shear_y_degrees = d[5],
          .translate_x = d[6],
          .translate_y = d[7]};
}
void Check(const im::VisionSample& actual, const J& expected) {
  SCOPED_TRACE(expected.at("stage").get<std::string>());
  auto hwc = actual.image.permute({1, 2, 0}).contiguous();
  EXPECT_EQ(sha256_hex(std::span(reinterpret_cast<const std::byte*>(
                                     hwc.const_data_ptr<std::uint8_t>()),
                                 std::size_t(hwc.numel()))),
            expected["rgb_hwc_sha256"].get<std::string>());
  ASSERT_TRUE(actual.annotation);
  ASSERT_EQ(actual.annotation->objects.size(), expected["boxes"].size());
  for (std::size_t i = 0; i < actual.annotation->objects.size(); ++i) {
    const auto& object = actual.annotation->objects[i];
    const auto b = *object.box;
    const std::array coordinates{b.x_min, b.y_min, b.x_max, b.y_max};
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(coordinates[j], expected["boxes"][i][j].get<double>(), 1e-5);
    EXPECT_EQ(object.category, expected["classes"][i].get<std::size_t>());
    ASSERT_FALSE(object.augmentation_sources.empty());
  }
  im::ImageCollateOptions options;
  const std::array samples{actual};
  const std::array keys{im::ImageBatchKey{.id = "opencv-fixed"}};
  EXPECT_NO_THROW(im::collate_images(samples, keys, options));
}
TEST(YoloOpenCv, DisabledBuildRejectsExplicitOperationsWithoutFallback) {
#ifdef JYPPX_TEST_VISION_OPENCV
  EXPECT_TRUE(t::yolo_opencv_available());
#else
  EXPECT_FALSE(t::yolo_opencv_available());
  EXPECT_THROW(t::prepare_yolo_opencv_geometry({}, {}), NotImplemented);
  EXPECT_THROW(t::prepare_yolo_opencv_color({}, {}), NotImplemented);
#endif
}
TEST(YoloOpenCv,
     FixedLargeMosaicAndGeometryMatchIndependentOriginalPixelsAndBoxes) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::prepare_yolo_opencv_geometry({}, {}), NotImplemented);
    return;
  }
  const auto ref = Reference();
  const auto samples = Samples(ref);
  auto before = samples[0].image.clone();
  for (int i = 0; i < 2; ++i) {
    const auto result = t::prepare_yolo_opencv_geometry(samples, Plan(ref, i));
    Check(result, ref["stages"][i ? "b-pretransform" : "a-pretransform"]);
    EXPECT_FALSE(result.geometry);
    EXPECT_FALSE(result.annotation->source_image_id);
    EXPECT_EQ(result.annotation->objects[0].augmentation_sources[0].image_path,
              "synthetic-input-0");
    const auto& actual =
        result.annotation->objects[0].augmentation_sources[0].matrix;
    const auto& matrix = ref["geometry"][i ? "b" : "a"]["matrix"];
    const double dx = ref["mosaic_center_yx"][i][1].get<int>() - 64;
    const double dy = ref["mosaic_center_yx"][i][0].get<int>() - 38;
    for (int row = 0; row < 3; ++row) {
      const double m0 = matrix[row][0], m1 = matrix[row][1],
                   m2 = matrix[row][2];
      EXPECT_NEAR(actual[row * 3], m0 * 64 / 29, 1e-10);
      EXPECT_NEAR(actual[row * 3 + 1], m1 * 38 / 17, 1e-10);
      EXPECT_NEAR(actual[row * 3 + 2], m0 * dx + m1 * dy + m2, 1e-10);
    }
  }
  EXPECT_TRUE(torch::equal(before, samples[0].image));
  const auto single = t::prepare_yolo_opencv_geometry(
      std::span(samples).first(1), {.size = 64, .warp = false});
  Check(single, ref["stages"]["no-mosaic-letterbox"]);
  ASSERT_TRUE(single.geometry);
  EXPECT_TRUE(single.geometry->retains_full_source());
}
TEST(YoloOpenCv, CompletePartnerMixHsvAndFlipsMatchOriginalBytesAndProvenance) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::prepare_yolo_opencv_color({}, {}), NotImplemented);
    return;
  }
  const auto ref = Reference();
  const auto samples = Samples(ref);
  const auto a = t::prepare_yolo_opencv_geometry(samples, Plan(ref, 0)),
             b = t::prepare_yolo_opencv_geometry(samples, Plan(ref, 1));
  const auto before = a.image.clone();
  const auto zero = t::prepare_yolo_opencv_color(a, {.hsv = true});
  Check(zero, ref["stages"]["hsv-zero"]);
  EXPECT_FALSE(torch::equal(zero.image, a.image));
  const double ratio = ref["mixup"]["ratio"];
  const auto mixed =
      t::prepare_yolo_opencv_color(a, {.mixup_ratio = ratio}, &b);
  Check(mixed, ref["stages"]["mixup"]);
  ASSERT_EQ(mixed.annotation->objects.size(), 8U);
  EXPECT_DOUBLE_EQ(mixed.annotation->objects[0].augmentation_sources[0].weight,
                   ratio);
  EXPECT_DOUBLE_EQ(mixed.annotation->objects[4].augmentation_sources[0].weight,
                   1 - ratio);
  const auto hsv =
      t::prepare_yolo_opencv_color(mixed, {.hue_shift = -.8 * .01373,
                                           .saturation_gain = 1 + .45 * .64481,
                                           .value_gain = 1 - .2 * .56565});
  Check(hsv, ref["stages"]["hsv"]);
  const auto flipped = t::prepare_yolo_opencv_color(
      hsv, {.horizontal_flip = true, .vertical_flip = true});
  Check(flipped, ref["stages"]["flips"]);
  EXPECT_TRUE(torch::equal(before, a.image));
  EXPECT_FALSE(flipped.geometry);
}
TEST(YoloOpenCv, RejectsRichInputsInvalidPlansCancellationAndResourceOverruns) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::prepare_yolo_opencv_geometry({}, {}), NotImplemented);
    return;
  }
  const auto ref = Reference();
  auto samples = Samples(ref);
  const auto plan = Plan(ref, 0);
  EXPECT_THROW(
      t::prepare_yolo_opencv_geometry(std::span(samples).first(1), plan),
      InvalidArgument);
  auto invalid = plan;
  invalid.scale = 0;
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, invalid),
               InvalidArgument);
  invalid = plan;
  invalid.mosaic_center[0] = 0;
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, invalid),
               InvalidArgument);
  auto ann =
      std::make_shared<im::VisionAnnotationRecord>(*samples[0].annotation);
  ann->objects[0].keypoints.push_back({{1, 1}, 2});
  samples[0].annotation = ann;
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, plan), InvalidArgument);
  samples = Samples(ref);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(
      t::prepare_yolo_opencv_geometry(samples, plan, {}, stop.get_token()),
      Cancelled);
  im::JointTransformOptions limits;
  limits.limits.max_working_bytes = 1;
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, plan, limits),
               ResourceExhausted);
  limits = {};
  limits.limits.max_output_bytes = 100;
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, plan, limits),
               ResourceExhausted);
  limits = {};
  limits.limits.max_objects = 3;
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, plan, limits),
               ResourceExhausted);
  const auto output = t::prepare_yolo_opencv_geometry(samples, plan);
  EXPECT_THROW(t::prepare_yolo_opencv_color(output, {.mixup_ratio = .5}),
               InvalidArgument);
  EXPECT_THROW(t::prepare_yolo_opencv_color(output, {.hue_shift = 2}),
               InvalidArgument);
}
TEST(YoloOpenCv, RetainsOutputLeaseAndReleasesFailedReservations) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::prepare_yolo_opencv_geometry({}, {}), NotImplemented);
    return;
  }
  const auto ref = Reference();
  const auto samples = Samples(ref);
  im::JointTransformOptions options;
  options.resource_budget = std::make_shared<im::ImageResourceBudget>();
  {
    const auto output =
        t::prepare_yolo_opencv_geometry(samples, Plan(ref, 0), options);
    EXPECT_EQ(
        options.resource_budget->usage(im::ImageResourceKind::pixels).current,
        64U * 64);
    EXPECT_EQ(
        options.resource_budget->usage(im::ImageResourceKind::decoded_bytes)
            .current,
        64U * 64 * 3);
    EXPECT_EQ(options.resource_budget
                  ->usage(im::ImageResourceKind::codec_scratch_bytes)
                  .current,
              0U);
  }
  EXPECT_EQ(
      options.resource_budget->usage(im::ImageResourceKind::pixels).current,
      0U);
  EXPECT_EQ(options.resource_budget->usage(im::ImageResourceKind::decoded_bytes)
                .current,
            0U);
  im::ImageResourceBudgetLimits small;
  small.max_codec_scratch_bytes = 1;
  options.resource_budget = std::make_shared<im::ImageResourceBudget>(small);
  EXPECT_THROW(t::prepare_yolo_opencv_geometry(samples, Plan(ref, 0), options),
               ResourceExhausted);
  EXPECT_EQ(
      options.resource_budget->usage(im::ImageResourceKind::pixels).current,
      0U);
  EXPECT_EQ(options.resource_budget->usage(im::ImageResourceKind::decoded_bytes)
                .current,
            0U);
}
TEST(YoloOpenCv, AffineIdentityAndZeroRatioPartnerKeepCanonicalBoxes) {
  if (!t::yolo_opencv_available()) {
    EXPECT_THROW(t::prepare_yolo_opencv_geometry({}, {}), NotImplemented);
    return;
  }
  const auto ref = Reference();
  const auto samples = Samples(ref);
  const auto single = std::span(samples).first(1);
  const auto letterbox =
      t::prepare_yolo_opencv_geometry(single, {.size = 64, .warp = false});
  const auto affine = t::prepare_yolo_opencv_geometry(single, {.size = 64});
  EXPECT_TRUE(torch::equal(letterbox.image, affine.image));
  ASSERT_EQ(affine.annotation->objects.size(), 1U);
  EXPECT_DOUBLE_EQ(affine.annotation->objects[0].box->x_min,
                   letterbox.annotation->objects[0].box->x_min);
  EXPECT_FALSE(affine.geometry);
  const auto other = t::prepare_yolo_opencv_geometry(samples, Plan(ref, 1));
  const auto partner =
      t::prepare_yolo_opencv_color(affine, {.mixup_ratio = 0}, &other);
  EXPECT_TRUE(torch::equal(partner.image, other.image));
  ASSERT_EQ(partner.annotation->objects.size(),
            other.annotation->objects.size());
  EXPECT_DOUBLE_EQ(
      partner.annotation->objects[0].augmentation_sources[0].weight, 1);
  EXPECT_NE(partner.annotation->objects[0].augmentation_sources[0].matrix,
            affine.annotation->objects[0].augmentation_sources[0].matrix);
}
}  // namespace
}  // namespace jyppx::liuding::vision::tests
