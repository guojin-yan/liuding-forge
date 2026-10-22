// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/data_image/annotation_files.hpp>
#include <jyppx/data_image/annotations.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <string>

namespace {
using namespace jyppx::data::image;

const char* kCoco = R"({
  "images":[{"id":17,"file_name":"train/猫.jpg","width":100,"height":50},
            {"id":91,"file_name":"empty.png","width":20,"height":20}],
  "categories":[{"id":90,"name":"猫"},{"id":3,"name":"dog"}],
  "annotations":[{"id":22,"image_id":17,"category_id":3,
    "bbox":[10,5,20,15],"area":280,"iscrowd":1,
    "segmentation":[[10,5,30,5,30,20,10,20]]}]
})";

AnnotationReadOptions Yolo() {
  AnnotationReadOptions options;
  options.width = 100;
  options.height = 50;
  options.image_path = "images/a.jpg";
  options.categories = {{19, "cat"}, {91, "dog"}};
  return options;
}

std::string Replace(std::string text, const std::string& from,
                    const std::string& to) {
  const auto position = text.find(from);
  if (position == std::string::npos) return {};
  text.replace(position, from.size(), to);
  return text;
}

TEST(ImageAnnotationTest, CocoPreservesSparseIdsCategoriesCrowdAndEmptyImages) {
  const auto index = parse_annotations(kCoco, AnnotationFormat::coco_instances);
  ASSERT_EQ(index.images.size(), 2U);
  ASSERT_EQ(index.categories.size(), 2U);
  EXPECT_EQ(index.categories[0].source_id, 90);
  EXPECT_EQ(index.categories[1].source_id, 3);
  EXPECT_EQ(index.images[0].image_path, "train/猫.jpg");
  EXPECT_TRUE(index.images[1].objects.empty());
  ASSERT_EQ(index.images[0].objects.size(), 1U);
  const auto& object = index.images[0].objects.front();
  EXPECT_EQ(object.category, 1U);
  EXPECT_EQ(object.source_id, 22);
  EXPECT_TRUE(object.crowd);
  ASSERT_TRUE(object.box);
  EXPECT_EQ(object.box->x_max, 30);
  EXPECT_EQ(object.box->y_max, 20);
  EXPECT_EQ(object.area, 280);
  ASSERT_EQ(object.polygons.size(), 1U);
  EXPECT_EQ(object.polygons.front().size(), 4U);
  EXPECT_EQ(index.source_sha256.size(), 64U);
  EXPECT_EQ(
      index.source_sha256,
      parse_annotations(kCoco, AnnotationFormat::coco_instances).source_sha256);
}

TEST(ImageAnnotationTest, CocoRejectsDuplicateAndDanglingIdentities) {
  EXPECT_THROW(parse_annotations(
                   Replace(kCoco, "\"category_id\":3", "\"category_id\":7"),
                   AnnotationFormat::coco_instances),
               jyppx::SerializationError);
  EXPECT_THROW(
      parse_annotations(Replace(kCoco, "\"image_id\":17", "\"image_id\":99"),
                        AnnotationFormat::coco_instances),
      jyppx::SerializationError);
  EXPECT_THROW(parse_annotations(Replace(kCoco, "\"id\":91", "\"id\":17"),
                                 AnnotationFormat::coco_instances),
               jyppx::SerializationError);
  EXPECT_THROW(parse_annotations(Replace(kCoco, "\"width\":100",
                                         "\"width\":100,\"width\":100"),
                                 AnnotationFormat::coco_instances),
               jyppx::SerializationError);
}

TEST(ImageAnnotationTest, CocoRejectsTraversalAndInvalidGeometry) {
  EXPECT_THROW(parse_annotations(Replace(kCoco, "train/猫.jpg", "../a.jpg"),
                                 AnnotationFormat::coco_instances),
               jyppx::SerializationError);
  EXPECT_THROW(parse_annotations(Replace(kCoco, "[10,5,20,15]", "[10,5,-1,15]"),
                                 AnnotationFormat::coco_instances),
               jyppx::SerializationError);
  EXPECT_THROW(parse_annotations(Replace(kCoco, "[10,5,20,15]", "[90,5,20,15]"),
                                 AnnotationFormat::coco_instances),
               jyppx::SerializationError);
}

TEST(ImageAnnotationTest,
     CocoDecodesCompressedRleSignedDeltasWithoutChangingMask) {
  const std::string prefix =
      R"({"images":[{"id":1,"file_name":"x.png","width":3,"height":2}],
    "categories":[{"id":5,"name":"object"}],"annotations":[
    {"id":1,"image_id":1,"category_id":5,"iscrowd":1,
     "segmentation":{"size":[2,3],"counts":)";
  const auto a = parse_annotations(prefix + "[0,3,1,1,1]}}]}",
                                   AnnotationFormat::coco_instances);
  const auto b = parse_annotations(prefix + "\"031N0\"}}]}",
                                   AnnotationFormat::coco_instances);
  ASSERT_TRUE(a.images[0].objects[0].rle);
  ASSERT_TRUE(b.images[0].objects[0].rle);
  EXPECT_EQ(a.images[0].objects[0].rle->counts,
            b.images[0].objects[0].rle->counts);
  EXPECT_EQ(b.images[0].objects[0].rle->counts,
            (std::vector<std::uint64_t>{0, 3, 1, 1, 1}));
  EXPECT_THROW(
      parse_annotations(prefix + "[0,7]}}]}", AnnotationFormat::coco_instances),
      jyppx::SerializationError);
  EXPECT_THROW(
      parse_annotations(prefix + "\"o\"}}]}", AnnotationFormat::coco_instances),
      jyppx::SerializationError);
}

TEST(ImageAnnotationTest, CocoKeypointVisibilityAndSkeletonAreNotDropped) {
  const auto index = parse_annotations(
      R"({"images":[{"id":1,"file_name":"pose.png","width":10,"height":20}],
    "categories":[{"id":17,"name":"person","keypoints":["left","right"],"skeleton":[[1,2]]}],
    "annotations":[{"id":2,"image_id":1,"category_id":17,
      "keypoints":[3,4,1,0,0,0],"num_keypoints":1}]})",
      AnnotationFormat::coco_keypoints);
  EXPECT_EQ(index.categories[0].skeleton[0][0], 0U);
  EXPECT_EQ(index.categories[0].skeleton[0][1], 1U);
  EXPECT_EQ(index.images[0].objects[0].keypoints[0].visibility, 1U);
  EXPECT_EQ(index.images[0].objects[0].keypoints[1].visibility, 0U);
}

TEST(ImageAnnotationTest, CocoPanopticPreservesSegmentIdAndMaskReference) {
  const auto index = parse_annotations(
      R"({"images":[{"id":4,"file_name":"image.jpg","width":10,"height":5}],
    "categories":[{"id":9,"name":"road","isthing":0}],
    "annotations":[{"image_id":4,"file_name":"maps/image.png","segments_info":[
      {"id":16777215,"category_id":9,"area":50,"iscrowd":0,"bbox":[0,0,10,5]}]}]})",
      AnnotationFormat::coco_panoptic);
  EXPECT_FALSE(index.categories[0].is_thing);
  EXPECT_EQ(index.images[0].panoptic_mask_path, "maps/image.png");
  EXPECT_EQ(index.images[0].objects[0].source_id, 16777215);
}

TEST(ImageAnnotationTest, YoloDetectionMapsNormalizedBoxesAndKeepsClassTable) {
  const auto index = parse_annotations(
      "1 0.5 0.5 0.2 0.4\r\n\n", AnnotationFormat::yolo_detection, Yolo());
  const auto& object = index.images[0].objects[0];
  EXPECT_EQ(object.category, 1U);
  EXPECT_EQ(index.categories[1].source_id, 91);
  ASSERT_TRUE(object.box);
  EXPECT_DOUBLE_EQ(object.box->x_min, 40);
  EXPECT_DOUBLE_EQ(object.box->x_max, 60);
  EXPECT_DOUBLE_EQ(object.box->y_min, 15);
  EXPECT_DOUBLE_EQ(object.box->y_max, 35);
  EXPECT_TRUE(parse_annotations("", AnnotationFormat::yolo_detection, Yolo())
                  .images[0]
                  .objects.empty());
}

TEST(ImageAnnotationTest, YoloRejectsWrongDialectNonFiniteAndBadClasses) {
  for (const auto* text : {"2 0.5 0.5 1 1", "0 nan 0.5 1 1", "0 0.5 0.5 0 1",
                           "0.0 0.5 0.5 1 1", "0 0.5 0.5 1 1 0.2", "0 0 0 1 1"})
    EXPECT_THROW(
        parse_annotations(text, AnnotationFormat::yolo_detection, Yolo()),
        jyppx::SerializationError);
  EXPECT_THROW(parse_annotations("", AnnotationFormat::yolo_detection),
               jyppx::InvalidArgument);
}

TEST(ImageAnnotationTest, YoloSegmentationAndObbRetainDistinctGeometry) {
  const std::string row = "0 .1 .2 .9 .2 .9 .8 .1 .8";
  const auto segment =
      parse_annotations(row, AnnotationFormat::yolo_segmentation, Yolo());
  EXPECT_FALSE(segment.images[0].objects[0].oriented_box);
  EXPECT_EQ(segment.images[0].objects[0].polygons[0].size(), 4U);
  const auto obb = parse_annotations(row, AnnotationFormat::yolo_obb, Yolo());
  EXPECT_TRUE(obb.images[0].objects[0].polygons.empty());
  ASSERT_TRUE(obb.images[0].objects[0].oriented_box);
  EXPECT_DOUBLE_EQ((*obb.images[0].objects[0].oriented_box)[2].x, 90);
  EXPECT_THROW(parse_annotations("0 .1 .2 .9 .8 .9 .2 .1 .8",
                                 AnnotationFormat::yolo_obb, Yolo()),
               jyppx::SerializationError);
}

TEST(ImageAnnotationTest, YoloPoseShapeAndVisibilityAreExplicit) {
  auto options = Yolo();
  options.keypoint_count = 2;
  const auto index = parse_annotations("0 .5 .5 1 1 .2 .4 1 0 0 0",
                                       AnnotationFormat::yolo_pose, options);
  EXPECT_DOUBLE_EQ(index.images[0].objects[0].keypoints[0].point.x, 20);
  EXPECT_DOUBLE_EQ(index.images[0].objects[0].keypoints[0].point.y, 20);
  EXPECT_EQ(index.images[0].objects[0].keypoints[0].visibility, 1U);
  EXPECT_EQ(index.images[0].objects[0].keypoints[1].visibility, 0U);
  EXPECT_THROW(parse_annotations("0 .5 .5 1 1 .2 .4 3 0 0 0",
                                 AnnotationFormat::yolo_pose, options),
               jyppx::SerializationError);
}

const char* kVoc =
    R"(<annotation><filename>img.jpg</filename><size><width>20</width>
  <height>10</height></size><object><name>cat</name><difficult>1</difficult>
  <truncated>1</truncated><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>20</xmax>
  <ymax>10</ymax></bndbox></object></annotation>)";

TEST(ImageAnnotationTest, VocMakesInclusiveCoordinatesAndDifficultExplicit) {
  const auto index = parse_annotations(kVoc, AnnotationFormat::voc);
  EXPECT_EQ(index.categories[0].name, "cat");
  const auto& object = index.images[0].objects[0];
  ASSERT_TRUE(object.box);
  EXPECT_DOUBLE_EQ(object.box->x_min, 0);
  EXPECT_DOUBLE_EQ(object.box->x_max, 20);
  EXPECT_TRUE(object.difficult);
  EXPECT_TRUE(object.truncated);
  auto options = Yolo();
  options.width = 20;
  options.height = 10;
  options.voc_coordinates = VocCoordinates::zero_based_boundary;
  const auto zero = parse_annotations(kVoc, AnnotationFormat::voc, options);
  EXPECT_DOUBLE_EQ(zero.images[0].objects[0].box->x_min, 1);
}

TEST(ImageAnnotationTest, VocRejectsExternalEntitiesAndDuplicateFields) {
  EXPECT_THROW(
      parse_annotations(
          "<!DOCTYPE annotation [<!ENTITY e SYSTEM 'file:///etc/passwd'>]>" +
              std::string(kVoc),
          AnnotationFormat::voc),
      jyppx::SerializationError);
  EXPECT_THROW(parse_annotations(Replace(kVoc, "<width>20</width>",
                                         "<width>20</width><width>20</width>"),
                                 AnnotationFormat::voc),
               jyppx::SerializationError);
}

TEST(ImageAnnotationTest, VocJoinsAllTextSegmentsAndRejectsExtraRootElements) {
  const auto bytes =
      Replace(Replace(kVoc, "<name>cat</name>",
                      "<name>ca<!--comment--><![CDATA[t]]></name>"),
              "<xmax>20</xmax>", "<xmax>2<![CDATA[0]]></xmax>");
  const auto index = parse_annotations(bytes, AnnotationFormat::voc);
  EXPECT_EQ(index.categories[0].name, "cat");
  EXPECT_EQ(index.images[0].objects[0].box->x_max, 20);
  EXPECT_THROW(
      parse_annotations(std::string(kVoc) + "<other/>", AnnotationFormat::voc),
      jyppx::SerializationError);
}

TEST(ImageAnnotationTest, PolygonRejectsDegenerateCrossingAndOverlappingEdges) {
  for (const auto* row : {"0 .1 .1 .2 .2 .3 .3", "0 0 0 1 1 1 0 0 1",
                          "0 0 0 1 0 .5 0 1 1 0 1", "0 0 0 1 0 1 0 0 1"})
    EXPECT_THROW(
        parse_annotations(row, AnnotationFormat::yolo_segmentation, Yolo()),
        jyppx::SerializationError);
  const auto closed = parse_annotations(
      "0 0 0 1 0 1 1 0 1 0 0", AnnotationFormat::yolo_segmentation, Yolo());
  EXPECT_EQ(closed.images[0].objects[0].polygons[0].size(), 5U);
  auto options = Yolo();
  options.limits.max_geometry_checks = 0;
  EXPECT_THROW(parse_annotations("0 0 0 1 0 1 1 0 1",
                                 AnnotationFormat::yolo_segmentation, options),
               jyppx::ResourceExhausted);
}

TEST(ImageAnnotationTest, YoloNodeBudgetCountsAllRows) {
  auto options = Yolo();
  options.limits.max_nodes = 9;
  EXPECT_THROW(parse_annotations("0 .5 .5 1 1\n0 .5 .5 1 1",
                                 AnnotationFormat::yolo_detection, options),
               jyppx::ResourceExhausted);
}

TEST(ImageAnnotationTest, MetadataUsesSharedPointAndXmlNodeBudgets) {
  auto options = Yolo();
  options.categories[0].keypoint_names = {"a", "b"};
  options.categories[1].keypoint_names = {"c", "d"};
  options.limits.max_points = 3;
  EXPECT_THROW(parse_annotations("", AnnotationFormat::yolo_detection, options),
               jyppx::ResourceExhausted);
  options = {};
  options.limits.max_string_bytes = 16;
  EXPECT_THROW(parse_annotations("<!--" + std::string(17, 'x') + "-->" + kVoc,
                                 AnnotationFormat::voc, options),
               jyppx::ResourceExhausted);
  options.limits.max_string_bytes = 16384;
  options.limits.max_nodes = 20;
  std::string comments;
  for (int i = 0; i < 21; ++i) comments += "<!--x-->";
  EXPECT_THROW(
      parse_annotations(comments + kVoc, AnnotationFormat::voc, options),
      jyppx::ResourceExhausted);
}

class AnnotationFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<unsigned> counter{0};
    root_ = std::filesystem::temp_directory_path() /
            ("jyppx-annotations-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "-" + std::to_string(counter.fetch_add(1)));
    ASSERT_TRUE(std::filesystem::create_directory(root_));
  }
  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(root_, error);
  }
  std::filesystem::path Write(const std::filesystem::path& name,
                              const std::string& bytes) {
    const auto path = root_ / name;
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    EXPECT_TRUE(output.good());
    return path;
  }
  std::filesystem::path root_;
};

TEST_F(AnnotationFileTest, ReadsUnicodeFileAndMatchesMemoryParserDigest) {
  const auto path =
      Write(std::filesystem::path(std::u8string(u8"标注.json")), kCoco);
  const auto result =
      read_annotation_file(path, AnnotationFormat::coco_instances);
  EXPECT_EQ(
      result.source_sha256,
      parse_annotations(kCoco, AnnotationFormat::coco_instances).source_sha256);
  EXPECT_EQ(result.images[0].objects[0].category, 1U);
  const auto yaml = Write("dataset.yaml", "names: [cat, dog]");
  EXPECT_EQ(read_yolo_dataset_yaml_file(yaml).categories.size(), 2U);
  EXPECT_TRUE(read_annotation_file(Write("empty.txt", ""),
                                   AnnotationFormat::yolo_detection, Yolo())
                  .images[0]
                  .objects.empty());
}

TEST_F(AnnotationFileTest, EnforcesByteLimitsAndCancelsBeforeOpening) {
  const auto path = Write("annotations.json", kCoco);
  auto options = Yolo();
  options.limits.max_input_bytes = 8;
  EXPECT_THROW(
      read_annotation_file(path, AnnotationFormat::coco_instances, options),
      jyppx::ResourceExhausted);
  EXPECT_THROW(read_annotation_file(root_, AnnotationFormat::coco_instances),
               jyppx::IoError);
  EXPECT_THROW(read_annotation_file(root_ / "missing", AnnotationFormat::voc),
               jyppx::IoError);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(read_annotation_file(root_ / "missing", AnnotationFormat::voc,
                                    {}, stop.get_token()),
               jyppx::Cancelled);
}

TEST(ImageAnnotationTest, LabelMeHandlesRectanglePolygonPointAndGrouping) {
  const auto index = parse_annotations(
      R"({"imagePath":"a.png","imageWidth":20,"imageHeight":10,
    "imageData":null,"shapes":[{"label":"cat","shape_type":"rectangle","points":[[10,8],[2,1]],"group_id":9},
      {"label":"cat","shape_type":"polygon","points":[[0,0],[20,0],[20,10]]},
      {"label":"point","shape_type":"point","points":[[5,5]],"flags":{"ignore":true}}]})",
      AnnotationFormat::labelme);
  ASSERT_EQ(index.images[0].objects.size(), 3U);
  EXPECT_EQ(index.images[0].objects[0].box->x_min, 2);
  EXPECT_EQ(index.images[0].objects[0].group_id, 9);
  EXPECT_EQ(index.images[0].objects[1].polygons[0].size(), 3U);
  EXPECT_EQ(index.images[0].objects[2].keypoints[0].visibility, 2U);
  EXPECT_TRUE(index.images[0].objects[2].ignore);
}

TEST(ImageAnnotationTest,
     LimitsCancellationAndStructuredDiagnosticsApplyToAllReaders) {
  auto options = Yolo();
  options.limits.max_objects = 0;
  EXPECT_THROW(parse_annotations("0 .5 .5 1 1",
                                 AnnotationFormat::yolo_detection, options),
               jyppx::ResourceExhausted);
  options.limits.max_input_bytes = 1;
  EXPECT_THROW(
      parse_annotations(kCoco, AnnotationFormat::coco_instances, options),
      jyppx::ResourceExhausted);
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(parse_annotations(kCoco, AnnotationFormat::coco_instances, {},
                                 stop.get_token()),
               jyppx::Cancelled);
  try {
    parse_annotations("{\"private_label\":1}",
                      AnnotationFormat::coco_instances);
    FAIL();
  } catch (const jyppx::SerializationError& error) {
    EXPECT_EQ(error.details().component, "data.image.annotations");
    EXPECT_EQ(std::string(error.what()).find("private_label"),
              std::string::npos);
  }
}

TEST(ImageAnnotationTest, IndependentConcurrentCallsKeepOwnedResults) {
  std::vector<std::future<AnnotationIndex>> calls;
  for (int i = 0; i < 16; ++i)
    calls.push_back(std::async(std::launch::async, [] {
      return parse_annotations(kCoco, AnnotationFormat::coco_instances);
    }));
  for (auto& call : calls) {
    const auto result = call.get();
    EXPECT_EQ(result.images[0].objects[0].category, 1U);
    EXPECT_EQ(result.categories[0].name, "猫");
  }
}

TEST(ImageAnnotationTest,
     YoloYamlParsesNamesSplitsAndPoseWithoutExecutingDownloads) {
  const auto spec = parse_yolo_dataset_yaml(R"(path: ../datasets/test
train: [images/train, images/extra]
val: images/val
names: {0: person, 1: 猫}
nc: 2
kpt_shape: [2, 3]
flip_idx: [1, 0]
download: 'never execute this command'
)");
  EXPECT_EQ(spec.root, "../datasets/test");
  EXPECT_EQ(spec.categories[1].name, "猫");
  EXPECT_EQ(spec.train.size(), 2U);
  EXPECT_EQ(spec.validation, (std::vector<std::string>{"images/val"}));
  EXPECT_EQ(spec.keypoint_count, 2U);
  EXPECT_EQ(spec.flip_indices, (std::vector<std::size_t>{1, 0}));
  EXPECT_TRUE(spec.has_download_instruction);
  EXPECT_EQ(spec.source_sha256.size(), 64U);
}

TEST(ImageAnnotationTest,
     YoloYamlRejectsAliasesTagsDuplicatesAndSparseClassIndices) {
  for (const auto* text :
       {"names: &x [cat]\ntrain: *x", "names: !!python/object cat",
        "names: [cat]\nnames: [dog]", "names: {1: cat}",
        "names: [cat]\n---\nnames: [dog]", "names: [cat]\nnc: 2",
        "names: [cat]\nkpt_shape: [2,3]\nflip_idx: [0,0]"})
    EXPECT_THROW(parse_yolo_dataset_yaml(text), jyppx::SerializationError);
  EXPECT_EQ(parse_yolo_dataset_yaml("names: [cat, dog]").categories.size(), 2U);
}

}  // namespace
