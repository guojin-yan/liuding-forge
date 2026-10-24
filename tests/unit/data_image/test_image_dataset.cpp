// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_dataset.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

namespace {
using namespace jyppx::liuding::data::image;
namespace fs = std::filesystem;

class TempDirectory final {
 public:
  TempDirectory() {
    static std::atomic<unsigned> counter{0U};
    const auto base = fs::temp_directory_path();
    for (unsigned attempt = 0; attempt < 64U; ++attempt) {
      const auto candidate =
          base /
          ("jyppx-image-dataset-" +
           std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()) +
           "-" + std::to_string(++counter) + "-" +
           std::to_string(std::random_device{}()));
      std::error_code error;
      if (fs::create_directory(candidate, error)) {
        path_ = candidate;
        return;
      }
    }
    throw std::runtime_error(
        "could not create image dataset fixture directory");
  }

  ~TempDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

  const fs::path& path() const noexcept { return path_; }

 private:
  fs::path path_;
};

std::vector<char> FixtureBytes(std::string_view name) {
  const auto path =
      fs::path(JYPPX_LIUDING_TEST_FIXTURE_ROOT) / "data_image" / name;
  std::ifstream stream(path, std::ios::binary);
  EXPECT_TRUE(stream.good()) << path.string();
  return {std::istreambuf_iterator<char>(stream),
          std::istreambuf_iterator<char>()};
}

void WriteFixture(const fs::path& path, std::string_view fixture_name) {
  fs::create_directories(path.parent_path());
  const auto bytes = FixtureBytes(fixture_name);
  std::ofstream stream(path, std::ios::binary);
  ASSERT_TRUE(stream.good());
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(stream.good());
}

ManifestReadOptions ClassificationOptions() {
  ManifestReadOptions options;
  options.categories = {{41, "cat"}, {99, "dog"}};
  return options;
}

ManifestIndex ClassificationIndex(std::string_view document) {
  return parse_manifest(document, ManifestFormat::csv,
                        ManifestTask::classification, ClassificationOptions());
}

TEST(ImageDatasetTest, LazilyDecodesManifestRowsIntoVisionSamples) {
  TempDirectory directory;
  WriteFixture(directory.path() / "cat" / "rgb.png", "rgb.png");
  WriteFixture(directory.path() / "dog.png", "rgb.jpg");
  const auto index =
      ClassificationIndex("path,labels\ncat/rgb.png,cat\ndog.png,dog\n");
  ImageDataset dataset(directory.path(), index);

  ASSERT_EQ(dataset.size(), 2U);
  const auto first = dataset.get(0U);
  EXPECT_EQ(first.image_path, "cat/rgb.png");
  EXPECT_EQ(first.labels, (std::vector<std::size_t>{0U}));
  ASSERT_NE(first.label_categories, nullptr);
  EXPECT_EQ(first.label_categories->at(0U).name, "cat");
  EXPECT_EQ(first.width, 2U);
  EXPECT_EQ(first.height, 1U);
  EXPECT_EQ(first.image.scalar_type(), torch::kUInt8);
  EXPECT_EQ(first.pixels.color, PixelColorModel::rgb);
  EXPECT_NO_THROW(validate_pixels(first.image, first.pixels));
  ASSERT_TRUE(first.geometry);
  EXPECT_TRUE(first.geometry->retains_full_source());
  EXPECT_EQ(first.geometry->source_extent().width, first.width);
  EXPECT_EQ(first.geometry->source_extent().height, first.height);
  const auto point = first.geometry->inverse_point({1, 0.5});
  ASSERT_TRUE(point);
  EXPECT_DOUBLE_EQ(point->x, 1);
  EXPECT_DOUBLE_EQ(point->y, 0.5);
  EXPECT_FALSE(first.paired_target.has_value());
  EXPECT_FALSE(first.paired_target_pixels.has_value());
  EXPECT_EQ(dataset.index().source_sha256(), index.source_sha256());

  const auto batch = dataset.get_batch({1U, 0U, 1U});
  ASSERT_EQ(batch.size(), 3U);
  EXPECT_EQ(batch[0].image_path, "dog.png");
  EXPECT_EQ(batch[1].image_path, "cat/rgb.png");
  EXPECT_EQ(batch[2].image_path, "dog.png");
  EXPECT_EQ(batch[0].labels, (std::vector<std::size_t>{1U}));
}

TEST(ImageDatasetTest, DecodesPairedTargetAndCanSelectIndependentPolicies) {
  TempDirectory directory;
  WriteFixture(directory.path() / "images" / "input.png", "rgb.png");
  WriteFixture(directory.path() / "masks" / "target.png", "rgb.png");
  ManifestReadOptions manifest_options;
  const auto index = parse_manifest(
      "input,target\nimages/input.png,masks/target.png\n", ManifestFormat::csv,
      ManifestTask::paired, manifest_options);
  ImageDatasetOptions options;
  options.paired_target.channels = ImageChannels::gray;
  options.paired_target.dtype = ImageDtype::float32_normalized;
  const ImageDataset dataset(directory.path(), index, options);

  const auto sample = dataset.get(0U);
  ASSERT_TRUE(sample.paired_target.has_value());
  EXPECT_EQ(sample.target_path, "masks/target.png");
  EXPECT_EQ(sample.paired_target->size(0), 1);
  EXPECT_EQ(sample.paired_target->scalar_type(), torch::kFloat32);
  ASSERT_TRUE(sample.paired_target_pixels);
  EXPECT_EQ(sample.paired_target_pixels->color, PixelColorModel::gray);
  EXPECT_EQ(sample.paired_target_pixels->range, PixelValueRange::unit_interval);
  EXPECT_NO_THROW(
      validate_pixels(*sample.paired_target, *sample.paired_target_pixels));
  EXPECT_EQ(sample.pixels.color, PixelColorModel::rgb);
  EXPECT_EQ(sample.pixels.dtype, torch::kUInt8);
  EXPECT_EQ(sample.paired_target->size(1), sample.image.size(1));
  EXPECT_EQ(sample.paired_target->size(2), sample.image.size(2));
}

TEST(ImageDatasetTest,
     ExplicitMaskTargetPreservesIdsAndHasNoPhotoPixelSemantics) {
  TempDirectory directory;
  WriteFixture(directory.path() / "input.png", "rgb.png");
  WriteFixture(directory.path() / "target.png", "masks/pair16.png");
  const auto index = parse_manifest("input,target\ninput.png,target.png\n",
                                    ManifestFormat::csv, ManifestTask::paired);
  ImageDatasetOptions options;
  options.image.dtype = ImageDtype::float32_normalized;
  options.paired_target.dtype = ImageDtype::float32_normalized;
  options.paired_target.channels = ImageChannels::rgb;
  options.mask_target.emplace();
  options.mask_target->ignore_id = 65535;
  const ImageDataset dataset(directory.path(), index, options);
  const auto sample = dataset.get(0);
  EXPECT_EQ(sample.image.scalar_type(), torch::kFloat32);
  ASSERT_TRUE(sample.mask_target);
  EXPECT_EQ(sample.mask_target->ids.scalar_type(), torch::kInt64);
  EXPECT_EQ(sample.mask_target->ids.sizes(), torch::IntArrayRef({1, 2}));
  EXPECT_EQ(sample.mask_target->ids[0][0].item<std::int64_t>(), 1024);
  EXPECT_EQ(sample.mask_target->ids[0][1].item<std::int64_t>(), 65535);
  EXPECT_EQ(sample.mask_target->ignore_id, 65535);
  EXPECT_FALSE(sample.paired_target);
  EXPECT_FALSE(sample.paired_target_pixels);
  EXPECT_EQ(sample.target_path, "target.png");
  ASSERT_TRUE(sample.geometry);
  EXPECT_TRUE(sample.geometry->retains_full_source());
}

TEST(ImageDatasetTest,
     MaskTargetRejectsMisalignmentEvenWithPairedImageSizeCheckDisabled) {
  TempDirectory directory;
  WriteFixture(directory.path() / "input.png", "rgb.png");
  WriteFixture(directory.path() / "target.png", "masks/gray16.png");
  const auto index = parse_manifest("input,target\ninput.png,target.png\n",
                                    ManifestFormat::csv, ManifestTask::paired);
  ImageDatasetOptions options;
  options.mask_target.emplace();
  options.require_matching_target_size = false;
  const ImageDataset dataset(directory.path(), index, options);
  EXPECT_THROW(dataset.get(0), jyppx::liuding::SerializationError);
}

TEST(ImageDatasetTest,
     RejectsMaskPolicyOnClassificationAndNeverFallsBackToPhotoDecode) {
  TempDirectory directory;
  ImageDatasetOptions options;
  options.mask_target.emplace();
  const auto classification =
      ClassificationIndex("path,labels\ninput.png,cat\n");
  EXPECT_THROW(static_cast<void>(
                   ImageDataset(directory.path(), classification, options)),
               jyppx::liuding::InvalidArgument);
  WriteFixture(directory.path() / "input.png", "rgb.png");
  WriteFixture(directory.path() / "target.png", "rgb.jpg");
  const auto paired = parse_manifest("input,target\ninput.png,target.png\n",
                                     ManifestFormat::csv, ManifestTask::paired);
  const ImageDataset dataset(directory.path(), paired, options);
  EXPECT_THROW(dataset.get(0), jyppx::liuding::SerializationError);
}

TEST(ImageDatasetTest, JoinsImmutableObjectAnnotationsByExactImagePath) {
  TempDirectory directory;
  WriteFixture(directory.path() / "cat" / "rgb.png", "rgb.png");
  auto manifest = ClassificationIndex("path,labels\ncat/rgb.png,cat\n");
  AnnotationIndex annotations;
  annotations.categories = {{41, "cat"}};
  VisionAnnotationRecord record;
  record.source_image_id = 12;
  record.image_path = "cat/rgb.png";
  record.width = 2;
  record.height = 1;
  VisionObject object;
  object.source_id = 8;
  object.category = 0;
  object.box = ImageBox{0, 0, 1, 1};
  record.objects.push_back(object);
  annotations.images.push_back(record);

  const ImageDataset dataset(directory.path(), std::move(manifest),
                             std::move(annotations));
  const auto sample = dataset.get(0U);
  ASSERT_NE(sample.annotation, nullptr);
  ASSERT_NE(sample.categories, nullptr);
  EXPECT_EQ(sample.categories->front().source_id, 41);
  EXPECT_EQ(sample.annotation->source_image_id, 12);
  ASSERT_EQ(sample.annotation->objects.size(), 1U);
  EXPECT_DOUBLE_EQ(sample.annotation->objects.front().box->x_max, 1.0);
}

TEST(ImageDatasetTest, RejectsAnnotationSizeMismatchWhenSampleIsRead) {
  TempDirectory directory;
  WriteFixture(directory.path() / "cat" / "rgb.png", "rgb.png");
  auto manifest = ClassificationIndex("path,labels\ncat/rgb.png,cat\n");
  AnnotationIndex annotations;
  annotations.categories = {{41, "cat"}};
  VisionAnnotationRecord record;
  record.image_path = "cat/rgb.png";
  record.width = 20;
  record.height = 10;
  annotations.images.push_back(record);
  const ImageDataset dataset(directory.path(), manifest, annotations);

  EXPECT_THROW(static_cast<void>(dataset.get(0U)),
               jyppx::liuding::SerializationError);
}

TEST(ImageDatasetTest, RejectsMissingLinksAndOutOfRangeSamples) {
  TempDirectory directory;
  const auto index = ClassificationIndex("path,labels\nmissing.png,cat\n");
  const ImageDataset dataset(directory.path(), index);
  EXPECT_THROW(static_cast<void>(dataset.get(0U)), jyppx::liuding::IoError);
  EXPECT_THROW(static_cast<void>(dataset.get(1U)),
               jyppx::liuding::InvalidArgument);
}

TEST(ImageDatasetTest, ObservesCancellationBeforeFilesystemAccess) {
  TempDirectory directory;
  const auto index = ClassificationIndex("path,labels\nmissing.png,cat\n");
  const ImageDataset dataset(directory.path(), index);
  std::stop_source cancellation;
  cancellation.request_stop();
  EXPECT_THROW(static_cast<void>(dataset.get(0U, cancellation.get_token())),
               jyppx::liuding::Cancelled);
}

TEST(ImageDatasetTest, RejectsSelectedRootThatDoesNotExist) {
  TempDirectory directory;
  const auto index = ClassificationIndex("path,labels\nimage.png,cat\n");
  EXPECT_THROW(
      static_cast<void>(ImageDataset(directory.path() / "absent", index)),
      jyppx::liuding::IoError);
}

TEST(ImageDatasetTest, RejectsAnnotationsOutsideTheSelectedManifest) {
  TempDirectory directory;
  WriteFixture(directory.path() / "cat" / "rgb.png", "rgb.png");
  auto manifest = ClassificationIndex("path,labels\ncat/rgb.png,cat\n");
  AnnotationIndex annotations;
  annotations.categories = {{41, "cat"}};
  VisionAnnotationRecord record;
  record.image_path = "cat/unindexed.png";
  annotations.images.push_back(record);

  EXPECT_THROW(
      static_cast<void>(ImageDataset(directory.path(), manifest, annotations)),
      jyppx::liuding::SerializationError);
}

TEST(ImageDatasetTest, RejectsSymbolicLinkInSamplePathWhenAvailable) {
  TempDirectory directory;
  const auto external =
      directory.path().parent_path() /
      (directory.path().filename().string() + "-external.png");
  WriteFixture(external, "rgb.png");
  const auto link = directory.path() / "linked.png";
  std::error_code error;
  fs::create_symlink(external, link, error);
  if (error) {
    GTEST_SKIP() << "symbolic links are not available: " << error.message();
  }
  const ImageDataset dataset(
      directory.path(), ClassificationIndex("path,labels\nlinked.png,cat\n"));
  EXPECT_THROW(static_cast<void>(dataset.get(0U)), jyppx::liuding::IoError);
  fs::remove(external, error);
}

AnnotationIndex PanopticIndex() {
  AnnotationIndex index;
  index.format = AnnotationFormat::coco_panoptic;
  index.categories = {{17, "part"}};
  VisionAnnotationRecord record;
  record.image_path = "photo.png";
  record.width = 2;
  record.height = 1;
  record.panoptic_mask_path = "labels.png";
  record.objects.push_back(
      {.source_id = 255, .box = ImageBox{0, 0, 1, 1}, .area = 1});
  record.objects.push_back(
      {.source_id = 65280, .box = ImageBox{1, 0, 2, 1}, .area = 1});
  index.images.push_back(std::move(record));
  return index;
}

ImageDatasetOptions PanopticOptions() {
  ImageDatasetOptions options;
  options.panoptic_target.emplace();
  options.panoptic_target->encoding = MaskEncoding::panoptic_rgb;
  options.panoptic_target->void_id = 0;
  return options;
}

TEST(ImageDatasetTest, ExplicitPanopticReadBindsSourcePathAndSegmentTable) {
  TempDirectory directory;
  WriteFixture(directory.path() / "photo.png", "rgb.png");
  WriteFixture(directory.path() / "labels.png", "rgb.png");
  const auto manifest = ClassificationIndex("path,labels\nphoto.png,cat\n");
  auto annotations = PanopticIndex();
  ImageDataset dataset(directory.path(), manifest, annotations,
                       PanopticOptions());
  const auto sample = dataset.get(0);
  ASSERT_TRUE(sample.mask_target);
  EXPECT_FALSE(sample.paired_target);
  EXPECT_EQ(sample.target_path, "labels.png");
  EXPECT_TRUE(sample.annotation->panoptic_mask_materialized);
  EXPECT_EQ(sample.mask_target->ids[0][0].item<int>(), 255);
  EXPECT_EQ(sample.mask_target->ids[0][1].item<int>(), 65280);
  EXPECT_EQ(sample.annotation->objects[0].mask_pixel_area, 1U);
  EXPECT_FALSE(annotations.images[0].panoptic_mask_materialized);
  ImageDataset metadata_only(directory.path(), manifest, annotations);
  EXPECT_FALSE(metadata_only.get(0).mask_target);
  annotations.images[0].objects[0].area = 2;
  ImageDataset mismatch(directory.path(), manifest, annotations,
                        PanopticOptions());
  EXPECT_THROW(mismatch.get(0), jyppx::liuding::SerializationError);
}

TEST(ImageDatasetTest, PanopticReadRequiresCompleteExplicitAssociations) {
  TempDirectory directory;
  WriteFixture(directory.path() / "photo.png", "rgb.png");
  WriteFixture(directory.path() / "labels.png", "rgb.png");
  const auto manifest = ClassificationIndex("path,labels\nphoto.png,cat\n");
  auto options = PanopticOptions();
  EXPECT_THROW(ImageDataset(directory.path(), manifest, options),
               jyppx::liuding::InvalidArgument);
  auto index = PanopticIndex();
  options.panoptic_target->void_id.reset();
  EXPECT_THROW(ImageDataset(directory.path(), manifest, index, options),
               jyppx::liuding::InvalidArgument);
  options = PanopticOptions();
  options.panoptic_target->encoding = MaskEncoding::rgb_mapping;
  EXPECT_THROW(ImageDataset(directory.path(), manifest, index, options),
               jyppx::liuding::InvalidArgument);
  options = PanopticOptions();
  index.images.clear();
  EXPECT_THROW(ImageDataset(directory.path(), manifest, index, options),
               jyppx::liuding::InvalidArgument);
  index = PanopticIndex();
  index.images[0].panoptic_mask_path = "../escape.png";
  EXPECT_THROW(ImageDataset(directory.path(), manifest, index, options),
               jyppx::liuding::SerializationError);
}

}  // namespace
