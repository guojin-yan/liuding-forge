// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/extended_annotations.hpp>

#include <gtest/gtest.h>
#include <stop_token>
#include <string>

namespace {
using namespace jyppx::liuding::data::image;

TextAnnotationReadOptions DotaOptions() {
  TextAnnotationReadOptions options;
  options.image_path = "images/0001.png";
  options.width = 100;
  options.height = 80;
  options.categories = {{17, "plane"}, {23, "ship"}};
  return options;
}

TEST(ExtendedImageAnnotationTest, DotaParsesHeadersQuadrilateralAndDifficulty) {
  const auto result = parse_dota_1x_annotations(
      "imagesource: GoogleEarth\n"
      "gsd: 0.25\n"
      "10 10 40 10 40 30 10 30 plane 2\n"
      "50 10 80 10 80 30 50 30 ship 0\n",
      DotaOptions());
  ASSERT_EQ(result.objects.size(), 2U);
  EXPECT_EQ(result.image_source, "GoogleEarth");
  EXPECT_DOUBLE_EQ(*result.ground_sample_distance, 0.25);
  EXPECT_EQ(result.objects[0].target.category, 0U);
  EXPECT_TRUE(result.objects[0].target.difficult);
  EXPECT_EQ(result.objects[0].difficulty, 2);
  ASSERT_TRUE(result.objects[1].target.oriented_box);
  EXPECT_DOUBLE_EQ((*result.objects[1].target.oriented_box)[2].x, 80);
  EXPECT_EQ(result.source_sha256.size(), 64U);
}

TEST(ExtendedImageAnnotationTest,
     DotaRejectsMalformedRowsGeometryAndUnknownClass) {
  const auto options = DotaOptions();
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane\n", options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 0 10 10 10 plane 0\n", options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 unknown 0\n", options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane -1\n", options),
      jyppx::liuding::SerializationError);
}

TextAnnotationReadOptions KittiOptions() {
  TextAnnotationReadOptions options;
  options.image_path = "training/image_000000.png";
  options.width = 1242;
  options.height = 375;
  options.categories = {{0, "Car"}, {1, "Pedestrian"}, {2, "DontCare"}};
  return options;
}

TEST(ExtendedImageAnnotationTest, KittiPreserves2D3DUnknownsAndScore) {
  const std::string row =
      "Car 0.10 1 -0.20 10 20 110 220 1.50 1.60 3.90 1.0 2.0 15.0 1.57\n";
  const auto result = parse_kitti_object_annotations(
      row, KittiObjectDialect::ground_truth, KittiOptions());
  ASSERT_EQ(result.objects.size(), 1U);
  const auto& object = result.objects.front();
  EXPECT_EQ(object.target.category, 0U);
  EXPECT_EQ(object.occlusion, 1);
  EXPECT_TRUE(object.alpha_known);
  EXPECT_TRUE(object.box_3d.dimensions_known);
  EXPECT_DOUBLE_EQ(object.box_3d.dimensions_hwl[2], 3.90);
  EXPECT_FALSE(object.score.has_value());

  const auto detection = parse_kitti_object_annotations(
      row.substr(0, row.size() - 1) + " 0.73\n",
      KittiObjectDialect::detection_result, KittiOptions());
  ASSERT_TRUE(detection.objects[0].score);
  EXPECT_DOUBLE_EQ(*detection.objects[0].score, 0.73);
}

TEST(ExtendedImageAnnotationTest, KittiKeepsDontCareAndSentinel3DValues) {
  const auto result = parse_kitti_object_annotations(
      "DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n",
      KittiObjectDialect::ground_truth, KittiOptions());
  const auto& object = result.objects.front();
  EXPECT_TRUE(object.target.ignore);
  EXPECT_FALSE(object.box_3d.dimensions_known);
  EXPECT_FALSE(object.box_3d.location_known);
  EXPECT_FALSE(object.box_3d.rotation_known);
  EXPECT_FALSE(object.alpha_known);
}

TEST(ExtendedImageAnnotationTest,
     KittiRejectsBadColumnsRangesAndPartialSentinels) {
  const auto options = KittiOptions();
  EXPECT_THROW(
      parse_kitti_object_annotations("Car 0 0 0 1 2 3 4\n",
                                     KittiObjectDialect::ground_truth, options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_kitti_object_annotations("Car 2 0 0 1 2 3 4 1 1 1 0 0 0 0\n",
                                     KittiObjectDialect::ground_truth, options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_kitti_object_annotations("Car 0 0 0 1 2 3 4 1 1 1 -1000 0 0 0\n",
                                     KittiObjectDialect::ground_truth, options),
      jyppx::liuding::SerializationError);
}

MotChallengeReadOptions MotOptions(MotChallengeDialect dialect) {
  MotChallengeReadOptions options;
  options.dialect = dialect;
  options.text.image_path = "MOT17-02";
  options.text.width = 1920;
  options.text.height = 1080;
  options.text.categories = {{1, "pedestrian"}, {2, "car"}, {99, "unknown"}};
  if (dialect != MotChallengeDialect::ground_truth_9)
    options.unclassified_category = 0;
  return options;
}

TEST(ExtendedImageAnnotationTest,
     MotGroundTruthGroupsFramesAndRetainsVisibility) {
  const auto result = parse_mot_challenge_annotations(
      "1,7,11,21,31,41,1,1,0.75\n"
      "1,8,51,61,20,30,0,2,0.00\n"
      "2,7,12,22,31,41,1,1,0.80\n",
      MotOptions(MotChallengeDialect::ground_truth_9));
  ASSERT_EQ(result.frames.size(), 2U);
  ASSERT_EQ(result.frames[0].objects.size(), 2U);
  EXPECT_EQ(result.frames[0].frame, 1);
  EXPECT_EQ(result.frames[0].objects[0].track_id, 7);
  EXPECT_DOUBLE_EQ(*result.frames[0].objects[0].visibility, 0.75);
  EXPECT_TRUE(result.frames[0].objects[1].target.ignore);
  EXPECT_DOUBLE_EQ(result.frames[1].objects[0].target.box->x_min, 11);
}

TEST(ExtendedImageAnnotationTest,
     MotDetectionSupportsMinusOneIdsAndWorldColumns) {
  auto options = MotOptions(MotChallengeDialect::detection_10);
  const auto detection =
      parse_mot_challenge_annotations("4,-1,1,2,30,40,0.90,0,0,0\n", options);
  EXPECT_EQ(detection.frames[0].objects[0].track_id, -1);
  EXPECT_DOUBLE_EQ(detection.frames[0].objects[0].confidence, 0.90);
  EXPECT_EQ(detection.frames[0].objects[0].target.category, 0U);
  ASSERT_TRUE(detection.frames[0].objects[0].world_xyz);
  EXPECT_DOUBLE_EQ((*detection.frames[0].objects[0].world_xyz)[0], 0);

  options.dialect = MotChallengeDialect::ground_truth_world_10;
  const auto world = parse_mot_challenge_annotations(
      "4,12,1,2,30,40,1,10.5,11.5,-1\n", options);
  ASSERT_TRUE(world.frames[0].objects[0].world_xyz);
  EXPECT_DOUBLE_EQ((*world.frames[0].objects[0].world_xyz)[0], 10.5);
}

TEST(ExtendedImageAnnotationTest,
     MotRejectsDuplicateIdentityClassAndVisibility) {
  auto options = MotOptions(MotChallengeDialect::ground_truth_9);
  EXPECT_THROW(parse_mot_challenge_annotations(
                   "1,7,1,1,10,10,1,1,1\n1,7,2,2,10,10,1,1,1\n", options),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_mot_challenge_annotations("1,7,1,1,10,10,1,77,1\n", options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_mot_challenge_annotations("1,7,1,1,10,10,1,1,2\n", options),
      jyppx::liuding::SerializationError);
}

TEST(ExtendedImageAnnotationTest,
     ExtendedReadersEnforceCancellationAndBudgets) {
  auto options = DotaOptions();
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane 0\n",
                                         options, stop.get_token()),
               jyppx::liuding::Cancelled);
  options.max_line_bytes = 4;
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane 0\n", options),
      jyppx::liuding::ResourceExhausted);
  options = DotaOptions();
  options.limits.max_objects = 0;
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane 0\n", options),
      jyppx::liuding::ResourceExhausted);
}

TEST(ExtendedImageAnnotationTest, DotaRetainsNullGsdAndEitherPolygonWinding) {
  const auto result = parse_dota_1x_annotations(
      "gsd: null\n10 10 10 30 40 30 40 10 plane 0\r\n", DotaOptions());
  EXPECT_TRUE(result.has_ground_sample_distance_header);
  EXPECT_FALSE(result.ground_sample_distance);
  EXPECT_FALSE(result.image_source);
  ASSERT_TRUE(result.objects[0].target.oriented_box);
  EXPECT_DOUBLE_EQ((*result.objects[0].target.oriented_box)[1].y, 30);
  EXPECT_THROW(parse_dota_1x_annotations("gsd: 1\ngsd: null\n", DotaOptions()),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_dota_1x_annotations(
                   "10 10 10 30 40 30 40 10 plane 0\ngsd: 1\n", DotaOptions()),
               jyppx::liuding::SerializationError);
}

TEST(ExtendedImageAnnotationTest,
     ExplicitBoundsPolicyKeepsOutsideGeometryWithoutClipping) {
  auto options = DotaOptions();
  const std::string row = "-5 10 10 10 10 20 -5 20 plane 0\n";
  EXPECT_THROW(parse_dota_1x_annotations(row, options),
               jyppx::liuding::SerializationError);
  options.bounds = AnnotationBounds::preserve_outside;
  const auto result = parse_dota_1x_annotations(row, options);
  ASSERT_TRUE(result.objects[0].target.oriented_box);
  EXPECT_DOUBLE_EQ((*result.objects[0].target.oriented_box)[0].x, -5);

  auto mot = MotOptions(MotChallengeDialect::detection_10);
  mot.text.bounds = AnnotationBounds::preserve_outside;
  const auto outside =
      parse_mot_challenge_annotations("1,-1,-5,1,10,10,.5,-1,-1,-1", mot);
  EXPECT_DOUBLE_EQ(outside.frames[0].objects[0].target.box->x_min, -6);
}

TEST(ExtendedImageAnnotationTest,
     MotDoesNotGuessUnclassifiedCategoryOrCoordinatePolicy) {
  auto options = MotOptions(MotChallengeDialect::detection_10);
  options.unclassified_category.reset();
  EXPECT_THROW(parse_mot_challenge_annotations("", options),
               jyppx::liuding::InvalidArgument);
  options.unclassified_category = 2;
  options.coordinates = MotCoordinates::zero_based;
  const auto result =
      parse_mot_challenge_annotations("1,-1,0,0,10,10,.5,-1,-1,-1", options);
  EXPECT_EQ(result.frames[0].objects[0].target.category, 2U);
  EXPECT_DOUBLE_EQ(result.frames[0].objects[0].target.box->x_min, 0);
  EXPECT_DOUBLE_EQ((*result.frames[0].objects[0].world_xyz)[2], -1);
  options.coordinates = MotCoordinates::one_based;
  EXPECT_THROW(
      parse_mot_challenge_annotations("1,-1,0,0,10,10,.5,-1,-1,-1", options),
      jyppx::liuding::SerializationError);
}

TEST(ExtendedImageAnnotationTest,
     MotDetectionIdsAndFrameOrderRemainSourceDefined) {
  auto options = MotOptions(MotChallengeDialect::detection_10);
  const auto result = parse_mot_challenge_annotations(
      "5,-1,1,1,10,10,.5,-1,-1,-1\n"
      "2,-1,1,1,10,10,.5,-1,-1,-1\n"
      "5,-1,2,2,10,10,.6,-1,-1,-1\n",
      options);
  ASSERT_EQ(result.frames.size(), 2U);
  EXPECT_EQ(result.frames[0].frame, 5);
  EXPECT_EQ(result.frames[1].frame, 2);
  EXPECT_EQ(result.frames[0].objects.size(), 2U);
  options.dialect = MotChallengeDialect::tracking_result_10;
  EXPECT_THROW(
      parse_mot_challenge_annotations("1,-1,1,1,10,10,.5,-1,-1,-1", options),
      jyppx::liuding::SerializationError);
  options.dialect = MotChallengeDialect::detection_10;
  options.frame_count = 3;
  EXPECT_THROW(
      parse_mot_challenge_annotations("4,-1,1,1,10,10,.5,-1,-1,-1", options),
      jyppx::liuding::SerializationError);
}

TEST(ExtendedImageAnnotationTest,
     ExtendedReadersRejectNonFiniteMalformedUtf8AndPrivateDiagnostics) {
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 nan 0 10 10 0 10 plane 0", DotaOptions()),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_kitti_object_annotations(
                   "Car 0 0 0 0 0 10 10 1 1 1 0 inf 0 0",
                   KittiObjectDialect::ground_truth, KittiOptions()),
               jyppx::liuding::SerializationError);
  auto mot = MotOptions(MotChallengeDialect::ground_truth_9);
  EXPECT_THROW(parse_mot_challenge_annotations("1,7,1,1,10,10,1,1,nan", mot),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(
      parse_dota_1x_annotations(std::string("\xc0\xaf", 2), DotaOptions()),
      jyppx::liuding::SerializationError);
  try {
    parse_dota_1x_annotations("0 0 10 0 10 10 0 10 private_name 0",
                              DotaOptions());
    FAIL();
  } catch (const jyppx::liuding::SerializationError& error) {
    EXPECT_EQ(std::string(error.what()).find("private_name"),
              std::string::npos);
    EXPECT_EQ(error.details().component, "data.image.annotations");
  }
}

TEST(ExtendedImageAnnotationTest,
     ExtendedLimitsCoverFieldsPointsLinesFramesAndInput) {
  auto options = DotaOptions();
  options.limits.max_nodes = 9;
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane 0", options),
      jyppx::liuding::ResourceExhausted);
  options = DotaOptions();
  options.limits.max_points = 3;
  EXPECT_THROW(
      parse_dota_1x_annotations("0 0 10 0 10 10 0 10 plane 0", options),
      jyppx::liuding::ResourceExhausted);
  options = DotaOptions();
  options.max_lines = 1;
  EXPECT_THROW(parse_dota_1x_annotations("\n\n", options),
               jyppx::liuding::ResourceExhausted);
  options = DotaOptions();
  options.limits.max_input_bytes = 1;
  EXPECT_THROW(parse_dota_1x_annotations("gsd: null", options),
               jyppx::liuding::ResourceExhausted);
  auto mot = MotOptions(MotChallengeDialect::detection_10);
  mot.text.limits.max_images = 1;
  EXPECT_THROW(
      parse_mot_challenge_annotations(
          "1,-1,1,1,10,10,.5,-1,-1,-1\n2,-1,1,1,10,10,.5,-1,-1,-1", mot),
      jyppx::liuding::ResourceExhausted);
}

TEST(ExtendedImageAnnotationTest, EmptyDocumentsRetainContextAndStableDigest) {
  const auto dota = parse_dota_1x_annotations("", DotaOptions());
  const auto kitti = parse_kitti_object_annotations(
      "", KittiObjectDialect::ground_truth, KittiOptions());
  const auto mot = parse_mot_challenge_annotations(
      "", MotOptions(MotChallengeDialect::ground_truth_9));
  EXPECT_TRUE(dota.objects.empty());
  EXPECT_TRUE(kitti.objects.empty());
  EXPECT_TRUE(mot.frames.empty());
  EXPECT_EQ(dota.source_sha256, kitti.source_sha256);
  EXPECT_EQ(kitti.source_sha256, mot.source_sha256);
  EXPECT_EQ(dota.source_sha256,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST(ExtendedImageAnnotationTest,
     CityscapesPreservesReviewAndInstanceMetadata) {
  auto options = DotaOptions();
  options.image_path = "leftImg8bit/train/city_000000_000000_leftImg8bit.png";
  options.width = 100;
  options.height = 80;
  options.categories = {{24, "car"}, {25, "person"}};
  const auto result = parse_cityscapes_polygon_annotations(
      R"({"imgWidth":100,"imgHeight":80,"objects":[
        {"label":"car","polygon":[[10,10],[40,10],[40,30]],"id":17,
         "instanceId":24001,"deleted":false,"verified":true,
         "user":"annotator","date":"2026-01-01"},
        {"label":"person","polygon":[],"id":18,"deleted":true,"verified":false}
      ]})",
      options);
  ASSERT_EQ(result.objects.size(), 2U);
  EXPECT_EQ(result.objects[0].object_id, 17);
  EXPECT_EQ(result.objects[0].instance_id, 24001);
  EXPECT_TRUE(result.objects[0].verified);
  EXPECT_EQ(result.objects[0].user, "annotator");
  EXPECT_EQ(result.objects[0].target.polygons[0].size(), 3U);
  EXPECT_TRUE(result.objects[1].deleted);
  EXPECT_TRUE(result.objects[1].target.ignore);
  EXPECT_TRUE(result.objects[1].target.polygons.empty());
}

TEST(ExtendedImageAnnotationTest,
     CityscapesRejectsUnknownLabelsDimensionsAndBadPolygons) {
  auto options = DotaOptions();
  options.image_path = "city.png";
  options.width = 100;
  options.height = 80;
  options.categories = {{24, "car"}};
  EXPECT_THROW(parse_cityscapes_polygon_annotations(
                   R"({"imgWidth":100,"imgHeight":80,"objects":[
                     {"label":"truck","polygon":[[1,1],[2,1],[2,2]]}]})",
                   options),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_cityscapes_polygon_annotations(
                   R"({"imgWidth":101,"imgHeight":80,"objects":[]})", options),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_cityscapes_polygon_annotations(
                   R"({"imgWidth":100,"imgHeight":80,"objects":[
                     {"label":"car","polygon":[[1,1],[2,2]]}]})",
                   options),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_cityscapes_polygon_annotations(
                   R"({"imgWidth":100,"imgHeight":80,"objects":[
                     {"label":"car","polygon":[[1,1],[2,2],[1,1]]}]})",
                   options),
               jyppx::liuding::SerializationError);
}

TEST(ExtendedImageAnnotationTest,
     CityscapesDeletedObjectMayRetainEmptyPolygonButStillHonorsLimits) {
  auto options = DotaOptions();
  options.image_path = "city.png";
  options.categories = {{24, "car"}};
  options.limits.max_objects = 1;
  const auto result = parse_cityscapes_polygon_annotations(
      R"({"imgWidth":100,"imgHeight":80,"objects":[
        {"label":"car","polygon":[],"deleted":true}]})",
      options);
  EXPECT_EQ(result.objects.size(), 1U);
  options.limits.max_objects = 0;
  EXPECT_THROW(parse_cityscapes_polygon_annotations(
                   R"({"imgWidth":100,"imgHeight":80,"objects":[
                     {"label":"car","polygon":[],"deleted":true}]})",
                   options),
               jyppx::liuding::ResourceExhausted);
}

TEST(ExtendedImageAnnotationTest,
     CityscapesRejectsDuplicateKeysIdsAndMalformedJson) {
  auto options = DotaOptions();
  options.categories = {{24, "car"}};
  EXPECT_THROW(
      parse_cityscapes_polygon_annotations(
          R"({"imgWidth":100,"imgWidth":100,"imgHeight":80,"objects":[]})",
          options),
      jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_cityscapes_polygon_annotations(
                   R"({"imgWidth":100,"imgHeight":80,"objects":[
                     {"label":"car","id":1,"deleted":true,"polygon":[]},
                     {"label":"car","id":1,"deleted":true,"polygon":[]}]})",
                   options),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(parse_cityscapes_polygon_annotations("{bad json", options),
               jyppx::liuding::SerializationError);
}

}  // namespace
