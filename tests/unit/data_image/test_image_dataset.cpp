// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/image_dataset.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <array>
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

TEST(TensorImageDatasetTest, SharesNoncontiguousCpuStorageAndSurvivesDataset) {
  const auto hwc = torch::ones({2, 3, 3}, torch::kUInt8);
  const auto chw = hwc.permute({2, 0, 1});
  ASSERT_FALSE(chw.is_contiguous());
  VisionSample retained;
  {
    TensorImageDataset dataset(
        {TensorImageRecord{chw, PixelSpec{}, "memory:sample-0", {0}}},
        {{41, "cat"}});
    EXPECT_EQ(dataset.size(), 1U);
    const auto sample = dataset.get(0);
    EXPECT_EQ(sample.image.data_ptr(), chw.data_ptr());
    EXPECT_EQ(sample.image_path, "memory:sample-0");
    EXPECT_EQ(sample.width, 3U);
    EXPECT_EQ(sample.height, 2U);
    EXPECT_FALSE(sample.image_resource_leases);
    ASSERT_TRUE(sample.geometry);
    EXPECT_EQ(sample.geometry->source_extent().width, 3U);
    ASSERT_TRUE(sample.label_categories);
    EXPECT_EQ(sample.label_categories->at(0).name, "cat");
    const jyppx::liuding::data::Dataset<VisionSample>& source = dataset;
    const auto repeated = source.get_batch({0, 0});
    ASSERT_EQ(repeated.size(), 2U);
    EXPECT_EQ(repeated[0].image.data_ptr(), repeated[1].image.data_ptr());
    const std::array keys{ImageBatchKey{"memory:sample-0", 0, 0},
                          ImageBatchKey{"memory:sample-0", 1, 0}};
    const auto batch = collate_images(repeated, keys);
    EXPECT_EQ(batch.images.size(0), 2);
    EXPECT_EQ(batch.items[0].image_path, "memory:sample-0");
    retained = sample;
  }
  EXPECT_EQ(retained.image.data_ptr(), chw.data_ptr());
  EXPECT_EQ(retained.image[0][0][0].item<std::uint8_t>(), 1U);
}

TEST(TensorImageDatasetTest, RejectsInvalidMetadataAndLogicalLimits) {
  const auto image = torch::zeros({3, 2, 3}, torch::kUInt8);
  const auto record = TensorImageRecord{image, PixelSpec{}, "sample", {0}};
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  EXPECT_THROW(
      TensorImageDataset({TensorImageRecord{image, PixelSpec{}, "", {}}}),
      jyppx::liuding::InvalidArgument);
  auto wrong_dtype = record;
  wrong_dtype.pixels.dtype = torch::kFloat32;
  EXPECT_THROW(TensorImageDataset({wrong_dtype}, {{41, "cat"}}),
               jyppx::liuding::InvalidArgument);
  auto wrong_channels = record;
  wrong_channels.pixels.color = PixelColorModel::gray;
  EXPECT_THROW(TensorImageDataset({wrong_channels}, {{41, "cat"}}),
               jyppx::liuding::InvalidArgument);
  auto limits = TensorImageDatasetLimits{};
  limits.max_dimension = 2;
  EXPECT_THROW(TensorImageDataset({record}, {{41, "cat"}}, limits),
               jyppx::liuding::ResourceExhausted);
  limits.max_dimension = 3;
  limits.max_sample_bytes = 17;
  EXPECT_THROW(TensorImageDataset({record}, {{41, "cat"}}, limits),
               jyppx::liuding::ResourceExhausted);
  limits.max_sample_bytes = 18;
  EXPECT_NO_THROW(TensorImageDataset({record}, {{41, "cat"}}, limits));
  limits.max_records = 0;
  EXPECT_THROW(TensorImageDataset({record}, {{41, "cat"}}, limits),
               jyppx::liuding::InvalidArgument);
}

TEST(TensorImageDatasetTest, ChecksIndexAndCancellationBeforePublication) {
  const auto image = torch::zeros({3, 2, 3}, torch::kUInt8);
  TensorImageDataset dataset(
      {TensorImageRecord{image, PixelSpec{}, "sample", {}}});
  EXPECT_THROW(dataset.get(1), jyppx::liuding::InvalidArgument);
  std::stop_source cancellation;
  cancellation.request_stop();
  EXPECT_THROW(dataset.get(0, cancellation.get_token()),
               jyppx::liuding::Cancelled);
}

TEST(TensorImageDatasetTest, SharesPairedPixelsThroughCollationAndGeometry) {
  const auto photo = torch::zeros({3, 2, 3}, torch::kUInt8);
  const auto paired = torch::arange(6, torch::kFloat32).reshape({2, 3, 1}) / 5;
  PixelSpec paired_spec;
  paired_spec.dtype = torch::kFloat32;
  paired_spec.layout = ImageLayout::hwc;
  paired_spec.color = PixelColorModel::gray;
  paired_spec.range = PixelValueRange::unit_interval;
  TensorImageRecord record{photo, PixelSpec{}, "source", {}};
  record.paired_target = paired;
  record.paired_target_pixels = paired_spec;
  record.target_id = "target";
  TensorImageDataset dataset({record});

  const auto sample = dataset.get(0);
  ASSERT_TRUE(sample.paired_target);
  EXPECT_EQ(sample.paired_target->data_ptr(), paired.data_ptr());
  EXPECT_EQ(sample.target_path, "target");
  const std::array samples{sample};
  const std::array keys{ImageBatchKey{"source", 0, 0}};
  const auto batch = collate_images(samples, keys);
  ASSERT_TRUE(batch.paired_images);
  EXPECT_EQ(batch.items[0].target_path, "target");
  EXPECT_FLOAT_EQ((*batch.paired_images)[0][0][0][2].item<float>(), 0.4f);

  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  const auto flipped = transform_sample(sample, steps);
  ASSERT_TRUE(flipped.paired_target);
  EXPECT_FLOAT_EQ((*flipped.paired_target)[0][0][0].item<float>(), 0.4f);
  EXPECT_EQ(sample.paired_target->data_ptr(), paired.data_ptr());
}

TEST(TensorImageDatasetTest, KeepsLargeDiscreteIdsThroughCpuTransform) {
  constexpr std::int64_t large_id = 9007199254740993LL;
  const auto photo = torch::zeros({3, 2, 3}, torch::kUInt8);
  const auto ids = torch::full({3, 2}, large_id, torch::kInt64).transpose(0, 1);
  ids[0][0] = -9;
  ASSERT_FALSE(ids.is_contiguous());
  DecodedMask mask;
  mask.ids = ids;
  mask.width = 3;
  mask.height = 2;
  mask.ignore_id = -9;
  TensorImageRecord record{photo, PixelSpec{}, "photo", {}};
  record.mask_target = mask;
  record.target_id = "labels";
  TensorImageDataset dataset({record});

  const auto sample = dataset.get(0);
  ASSERT_TRUE(sample.mask_target);
  EXPECT_EQ(sample.mask_target->ids.data_ptr(), ids.data_ptr());
  const std::array steps{
      ImageGeometryStep{.kind = ImageGeometryKind::horizontal_flip}};
  const auto flipped = transform_sample(sample, steps);
  ASSERT_TRUE(flipped.mask_target);
  EXPECT_EQ(flipped.mask_target->ids[0][2].item<std::int64_t>(), -9);
  EXPECT_EQ(flipped.mask_target->ids[1][0].item<std::int64_t>(), large_id);
  const std::array samples{sample};
  const std::array keys{ImageBatchKey{"photo", 0, 0}};
  const auto batch = collate_images(samples, keys);
  ASSERT_TRUE(batch.masks);
  EXPECT_EQ((*batch.masks)[0][0][1].item<std::int64_t>(), large_id);
}

TEST(TensorImageDatasetTest, RetainsTargetsAfterDatasetDestruction) {
  VisionSample paired_sample;
  VisionSample mask_sample;
  {
    TensorImageRecord paired{
        torch::zeros({3, 1, 2}, torch::kUInt8), PixelSpec{}, "photo", {}};
    paired.paired_target = torch::full({3, 1, 2}, 7, torch::kUInt8);
    paired.paired_target_pixels = PixelSpec{};
    paired.target_id = "paired";

    DecodedMask mask;
    mask.ids = torch::full({1, 2}, 9007199254740993LL, torch::kInt64);
    mask.width = 2;
    mask.height = 1;
    TensorImageRecord labeled{
        torch::zeros({3, 1, 2}, torch::kUInt8), PixelSpec{}, "labels", {}};
    labeled.mask_target = std::move(mask);
    labeled.target_id = "mask";

    TensorImageDataset dataset({std::move(paired), std::move(labeled)});
    paired_sample = dataset.get(0);
    mask_sample = dataset.get(1);
  }
  ASSERT_TRUE(paired_sample.paired_target);
  EXPECT_EQ(paired_sample.paired_target->index({0, 0, 0}).item<std::uint8_t>(),
            7U);
  EXPECT_EQ(paired_sample.target_path, "paired");
  ASSERT_TRUE(mask_sample.mask_target);
  EXPECT_EQ(mask_sample.mask_target->ids[0][1].item<std::int64_t>(),
            9007199254740993LL);
  EXPECT_EQ(mask_sample.target_path, "mask");
}

TEST(TensorImageDatasetTest, RejectsMisalignedOrAmbiguousTargets) {
  const auto photo = torch::zeros({3, 2, 3}, torch::kUInt8);
  const auto pair = torch::zeros({3, 2, 3}, torch::kUInt8);
  TensorImageRecord record{photo, PixelSpec{}, "photo", {}};
  record.paired_target = pair;
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  record.paired_target_pixels = PixelSpec{};
  record.target_id = "pair";
  EXPECT_NO_THROW(TensorImageDataset({record}));
  record.paired_target = torch::zeros({3, 3, 2}, torch::kUInt8);
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  record.paired_target = pair;
  DecodedMask mask;
  mask.ids = torch::zeros({2, 3}, torch::kInt64);
  mask.width = 3;
  mask.height = 2;
  record.mask_target = mask;
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  record.paired_target.reset();
  record.paired_target_pixels.reset();
  EXPECT_NO_THROW(TensorImageDataset({record}));
  record.mask_target->ids = torch::zeros({2, 3}, torch::kFloat32);
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  record.mask_target->ids = torch::zeros({2, 3}, torch::kInt64);
  record.mask_target->width = 2;
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  record.mask_target->width = 3;
  auto limits = TensorImageDatasetLimits{};
  limits.max_sample_bytes = 65;
  EXPECT_THROW(TensorImageDataset({record}, {}, limits),
               jyppx::liuding::ResourceExhausted);
  limits.max_sample_bytes = 66;
  EXPECT_NO_THROW(TensorImageDataset({record}, {}, limits));
  record.mask_target.reset();
  record.paired_target = pair;
  record.paired_target_pixels = PixelSpec{};
  limits.max_sample_bytes = 35;
  EXPECT_THROW(TensorImageDataset({record}, {}, limits),
               jyppx::liuding::ResourceExhausted);
  limits.max_sample_bytes = 36;
  EXPECT_NO_THROW(TensorImageDataset({record}, {}, limits));
}

TEST(TensorImageDatasetTest, RetainsCudaStorageWithoutDeviceTransfer) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is unavailable";
  const auto image = torch::zeros({3, 2, 3}, torch::kUInt8).to(torch::kCUDA);
  TensorImageDataset dataset(
      {TensorImageRecord{image, PixelSpec{}, "cuda:sample", {}}});
  const auto sample = dataset.get(0);
  EXPECT_TRUE(sample.image.is_cuda());
  EXPECT_EQ(sample.image.data_ptr(), image.data_ptr());
  EXPECT_EQ(sample.width, 3U);
  EXPECT_EQ(sample.height, 2U);
}

TEST(TensorImageDatasetTest, RequiresSameDeviceForCudaPairedPixels) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA is unavailable";
  const auto cpu = torch::zeros({3, 2, 3}, torch::kUInt8);
  const auto cuda = cpu.to(torch::kCUDA);
  TensorImageRecord record{cuda, PixelSpec{}, "cuda:photo", {}};
  record.paired_target = cpu;
  record.paired_target_pixels = PixelSpec{};
  record.target_id = "cuda:paired";
  EXPECT_THROW(TensorImageDataset({record}), jyppx::liuding::InvalidArgument);
  record.paired_target = cuda.clone();
  TensorImageDataset dataset({record});
  const auto sample = dataset.get(0);
  ASSERT_TRUE(sample.paired_target);
  EXPECT_TRUE(sample.paired_target->is_cuda());
  EXPECT_EQ(sample.paired_target->data_ptr(), record.paired_target->data_ptr());
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

TEST(ImageDatasetTest, HoldsSourceLeaseAcrossCollationAndReleasesOnFailure) {
  TempDirectory directory;
  WriteFixture(directory.path() / "input.png", "rgb.png");
  const auto classification =
      ClassificationIndex("path,labels\ninput.png,cat\n");
  auto budget = std::make_shared<ImageResourceBudget>();
  ImageDatasetOptions options;
  options.resource_budget = budget;
  const ImageDataset dataset(directory.path(), classification, options);
  {
    auto sample = dataset.get(0);
    ASSERT_NE(sample.image_resource_leases, nullptr);
    EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 6U);
    ImageCollateOptions collate_options;
    collate_options.resource_budget = budget;
    const std::array samples{sample};
    const std::array keys{ImageBatchKey{"input", 0, 0}};
    auto batch = collate_images(samples, keys, collate_options);
    EXPECT_GT(budget->usage(ImageResourceKind::decoded_bytes).current, 6U);
    EXPECT_EQ(batch.items.size(), 1U);
  }
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);

  WriteFixture(directory.path() / "target.png", "rgb.png");
  const auto paired = parse_manifest("input,target\ninput.png,target.png\n",
                                     ManifestFormat::csv, ManifestTask::paired);
  ImageResourceBudgetLimits limits;
  limits.max_decoded_bytes = 6;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  const ImageDataset paired_dataset(directory.path(), paired, options);
  EXPECT_THROW(paired_dataset.get(0), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);
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
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  const ImageDataset dataset(directory.path(), index, options);
  const auto sample = dataset.get(0);
  EXPECT_EQ(sample.image.scalar_type(), torch::kFloat32);
  ASSERT_TRUE(sample.mask_target);
  ASSERT_NE(sample.mask_target->resource_leases, nullptr);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 40U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 4U);
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

TEST(ImageDatasetTest, PanopticBindingBudgetFollowsSharedAnnotation) {
  TempDirectory directory;
  WriteFixture(directory.path() / "photo.png", "rgb.png");
  WriteFixture(directory.path() / "labels.png", "rgb.png");
  const auto manifest = ClassificationIndex("path,labels\nphoto.png,cat\n");
  const auto annotations = PanopticIndex();
  auto options = PanopticOptions();
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  ImageDataset dataset(directory.path(), manifest, annotations, options);

  auto sample = dataset.get(0);
  const auto live = budget->usage(ImageResourceKind::annotation_bytes);
  EXPECT_GT(live.current, 0U);
  EXPECT_GT(live.peak, live.current);
  auto retained = sample;
  sample = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current,
            live.current);
  retained = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);

  ImageResourceBudgetLimits limits;
  limits.max_annotation_bytes = 255;
  auto tight = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = tight;
  ImageDataset rejected(directory.path(), manifest, annotations, options);
  EXPECT_THROW(rejected.get(0), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).peak, 0U);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).current, 0U);

  limits.max_annotation_bytes = 256;
  tight = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = tight;
  ImageDataset output_rejected(directory.path(), manifest, annotations,
                               options);
  EXPECT_THROW(output_rejected.get(0), jyppx::liuding::ResourceExhausted);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).peak, 256U);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).current, 0U);
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
