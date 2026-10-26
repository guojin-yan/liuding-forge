// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data/data_pipeline.hpp>
#include <jyppx/liuding/data/samplers/sequential_sampler.hpp>
#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/data_image/batch_identity.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <thread>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::data;
using namespace jyppx::liuding::data::image;

VisionSample Photo(std::uint32_t w = 3, std::uint32_t h = 2, int channels = 3) {
  VisionSample s;
  s.width = w;
  s.height = h;
  s.image_path = "fixture.png";
  s.pixels.color = channels == 1 ? PixelColorModel::gray : PixelColorModel::rgb;
  s.image =
      torch::arange(static_cast<std::int64_t>(channels) * w * h, torch::kInt64)
          .reshape({channels, h, w})
          .to(torch::kUInt8);
  s.geometry.emplace(ImageExtent{w, h});
  return s;
}
std::vector<ImageBatchKey> Keys(std::size_t n) {
  std::vector<ImageBatchKey> keys;
  for (std::size_t i = 0; i < n; ++i)
    keys.push_back({"sample-" + std::to_string(i), i, i + 1});
  return keys;
}
ImageBatch Batch(const std::vector<VisionSample>& samples,
                 ImageCollateOptions options = {}) {
  return collate_images(samples, Keys(samples.size()), options);
}
void Mask(VisionSample& s) {
  s.mask_target.emplace();
  s.mask_target->ids =
      torch::arange(static_cast<std::int64_t>(s.width) * s.height,
                    torch::kInt64)
          .reshape({s.height, s.width}) +
      9007199254740993LL;
  s.mask_target->width = s.width;
  s.mask_target->height = s.height;
  s.mask_target->ignore_id = -7;
  s.mask_target->void_id = -9;
}
std::shared_ptr<VisionAnnotationRecord> Record(VisionSample& s) {
  s.categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{
          {.source_id = 71, .name = "object", .keypoint_names = {"one"}}});
  auto r = std::make_shared<VisionAnnotationRecord>();
  r->source_image_id = std::numeric_limits<std::int64_t>::min();
  r->image_path = s.image_path;
  r->width = s.width;
  r->height = s.height;
  s.annotation = r;
  return r;
}
ImageMaskRle Encode(const std::vector<int>& row_major, std::uint32_t w,
                    std::uint32_t h) {
  ImageMaskRle r{w, h, {0}};
  int previous = 0;
  for (std::uint32_t x = 0; x < w; ++x)
    for (std::uint32_t y = 0; y < h; ++y) {
      const int value = row_major[y * w + x];
      if (previous != value) {
        r.counts.push_back(0);
        previous = value;
      }
      ++r.counts.back();
    }
  return r;
}
std::vector<int> Decode(const ImageMaskRle& r) {
  std::vector<int> row_major(static_cast<std::size_t>(r.width) * r.height);
  std::uint64_t index = 0;
  bool value = false;
  for (const auto count : r.counts) {
    for (std::uint64_t i = 0; i < count; ++i, ++index)
      row_major[(index % r.height) * r.width + index / r.height] =
          value ? 1 : 0;
    value = !value;
  }
  return row_major;
}

TEST(ImageBatchTest, StackAcceptsMixedStridedLayoutsAndOwnsFinalStorage) {
  auto a = Photo();
  auto b = a;
  b.image = torch::zeros({2, 6, 3}, torch::kUInt8).slice(1, 0, 6, 2);
  b.image.copy_(a.image.permute({1, 2, 0}));
  b.pixels.layout = ImageLayout::hwc;
  ASSERT_FALSE(b.image.is_contiguous());
  const auto batch = Batch({a, b});
  EXPECT_EQ(batch.images.sizes(), torch::IntArrayRef({2, 3, 2, 3}));
  EXPECT_TRUE(batch.images.is_contiguous());
  EXPECT_FALSE(batch.images.is_pinned());
  EXPECT_TRUE(torch::equal(batch.images[0], batch.images[1]));
  EXPECT_EQ(batch.valid.sum().item<int>(), 12);
  a.image.zero_();
  b.image.zero_();
  EXPECT_EQ(batch.images[0][2][1][2].item<int>(), 17);
  EXPECT_EQ(batch.items[1].key.id, "sample-1");
  EXPECT_EQ(batch.items[1].key.occurrence, 1U);
  EXPECT_EQ(batch.items[1].key.view, 2U);
}

TEST(ImageBatchResourceBudgetTest, BatchStagesRetainAggregateLeases) {
  auto budget = std::make_shared<ImageResourceBudget>();
  ImageCollateOptions collate;
  collate.resource_budget = budget;
  auto batch = Batch({Photo(), Photo()}, collate);

  const auto staged = budget->usage(ImageResourceKind::staged_batches);
  const auto decoded = budget->usage(ImageResourceKind::decoded_bytes);
  const auto pixels = budget->usage(ImageResourceKind::pixels);
  const auto annotations = budget->usage(ImageResourceKind::annotation_bytes);
  const auto indexes = budget->usage(ImageResourceKind::index_bytes);
  EXPECT_EQ(staged.current, 1U);
  EXPECT_EQ(staged.peak, 1U);
  EXPECT_EQ(decoded.current, 48U);  // image bytes plus valid masks
  EXPECT_EQ(pixels.current, 12U);   // two 3x2 image canvases
  EXPECT_GT(annotations.current, 0U);
  EXPECT_EQ(indexes.current, 24U);  // three int64 label offsets

  ImageBatchTransformOptions numeric;
  numeric.image.dtype = torch::kFloat32;
  numeric.image.scale = BatchPixelScale::preserve;
  numeric.resource_budget = budget;
  auto transformed = transform_image_batch(batch, numeric);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 2U);
  EXPECT_GT(budget->usage(ImageResourceKind::decoded_bytes).current,
            decoded.current);

  transformed = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 1U);
  batch = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
}

TEST(ImageBatchResourceBudgetTest, CollateChargesEveryPairedCanvas) {
  auto a = Photo();
  a.paired_target = a.image.clone();
  a.paired_target_pixels = a.pixels;
  auto b = Photo();
  b.paired_target = b.image.clone();
  b.paired_target_pixels = b.pixels;
  ImageCollateOptions options;
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  auto batch = Batch({a, b}, options);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 24U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 84U);
  batch = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);

  ImageResourceBudgetLimits limits;
  limits.max_pixels = 23;
  auto tight_budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = tight_budget;
  EXPECT_THROW(Batch({a, b}, options), ResourceExhausted);
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::staged_batches).current, 0U);
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(tight_budget->usage(ImageResourceKind::pixels).current, 0U);
}

TEST(ImageBatchResourceBudgetTest, AssignmentReleasesStorageBeforeLease) {
  auto budget = std::make_shared<ImageResourceBudget>();
  std::uint64_t charged_at_storage_release = 0;
  auto* bytes = new std::byte[4];
  ImageBatch batch;
  batch.resource_lease_sets.push_back(
      std::make_shared<ImageResourceLeaseSet>());
  batch.resource_lease_sets.back()->push_back(
      budget->reserve(ImageResourceKind::decoded_bytes, 4));
  batch.images = torch::from_blob(
      bytes, {4},
      [budget, &charged_at_storage_release](void* storage) {
        charged_at_storage_release =
            budget->usage(ImageResourceKind::decoded_bytes).current;
        delete[] static_cast<std::byte*>(storage);
      },
      torch::TensorOptions().dtype(torch::kUInt8));
  batch = {};
  EXPECT_EQ(charged_at_storage_release, 4U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
}

TEST(ImageBatchResourceBudgetTest,
     NumericReservesOutputAndWorkingStorageBeforeConversion) {
  const auto batch = Batch({Photo()});
  ImageBatchTransformOptions options;
  options.image.scale = BatchPixelScale::preserve;
  ImageResourceBudgetLimits limits;
  limits.max_decoded_bytes = 77;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch(batch, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).peak, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);

  limits.max_decoded_bytes = 221;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch(batch, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).peak, 78U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);

  limits.max_decoded_bytes = 222;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto result = transform_image_batch(batch, options);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).peak, 222U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 78U);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 1U);
  result = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);
}

TEST(ImageBatchResourceBudgetTest,
     NumericChargesUnchangedPairedPhotosInRaggedOutput) {
  auto a = Photo();
  auto b = Photo(2, 1);
  for (auto* sample : {&a, &b}) {
    sample->paired_target = sample->image.clone();
    sample->paired_target_pixels = sample->pixels;
  }
  const auto batch = Batch({a, b}, {.layout = ImageBatchLayout::ragged});
  ImageBatchTransformOptions options;
  options.image.scale = BatchPixelScale::preserve;
  ImageResourceBudgetLimits limits;
  limits.max_pixels = 15;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch(batch, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);

  limits.max_pixels = 16;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto result = transform_image_batch(batch, options);
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 16U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 128U);
  EXPECT_EQ(result.paired_image_list[0].data_ptr(),
            batch.paired_image_list[0].data_ptr());
  result = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::pixels).current, 0U);
}

TEST(ImageBatchResourceBudgetTest, GeometryAndMemoryStagesCanBeBound) {
  auto budget = std::make_shared<ImageResourceBudget>();
  ImageCollateOptions collate;
  collate.resource_budget = budget;
  const auto batch = Batch({Photo()}, collate);

  ImageBatchGeometryOptions geometry;
  geometry.resource_budget = budget;
  const auto transformed = transform_image_batch_geometry(
      batch, std::vector<std::vector<ImageGeometryStep>>(1), geometry);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 2U);

  const auto prepared =
      prepare_image_batch(transformed, BatchMemoryOptions{}, {}, budget);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 3U);
  EXPECT_GT(prepared.resource_lease_sets.size(), 0U);
}

TEST(ImageBatchResourceBudgetTest, FailedReservationRollsBackEarlierLeases) {
  ImageResourceBudgetLimits limits;
  limits.max_decoded_bytes = 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  ImageCollateOptions collate;
  collate.resource_budget = budget;
  EXPECT_THROW(Batch({Photo()}, collate), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
}

TEST(ImageBatchResourceBudgetTest,
     WorkerPipelineRetainsImageLeaseUntilEnvelopeRelease) {
  auto budget = std::make_shared<ImageResourceBudget>();
  DataPipelineOptions options;
  options.pipeline_id = "image.batch.resource-worker";
  options.dataset_version = "image-batch-fixture/v1";
  options.dataset_size = 2;
  options.batch_factory_id = "image-batch-fixture/resource/v1";
  options.batch_size = 2;
  options.num_workers = 2;
  options.prefetch_batches = 1;
  const auto template_batch = Batch({Photo(), Photo()});
  options.schema = image_batch_schema(template_batch, torch::kCPU);
  options.resource_lease_factory =
      make_image_batch_resource_lease_factory(budget);

  DataPipeline pipeline(
      std::make_shared<samplers::SequentialSampler>(2), options,
      [](std::span<const std::size_t> indices,
         const DataWorkerContext& context, std::stop_token stop) {
        std::vector<VisionSample> samples;
        std::vector<ImageBatchKey> keys;
        for (const auto index : indices) {
          auto sample = Photo();
          sample.image.fill_(static_cast<int>(index));
          samples.push_back(std::move(sample));
          keys.push_back({std::to_string(index), context.epoch, 0});
        }
        return image_batch_value(collate_images(samples, keys, {}, stop));
      });
  pipeline.set_epoch(0);
  auto envelope = pipeline.next();
  ASSERT_TRUE(envelope);
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 1U);
  EXPECT_GT(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
  envelope.reset();
  EXPECT_EQ(budget->usage(ImageResourceKind::staged_batches).current, 0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::decoded_bytes).current, 0U);
}

TEST(ImageBatchResourceBudgetTest,
     AdmissionBoundsWorkerProductionUntilConsumerReleasesBatch) {
  ImageResourceBudgetLimits limits;
  limits.max_pipeline_inflight_batches = 1;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  DataPipelineOptions options;
  options.pipeline_id = "image.batch.admission-worker";
  options.dataset_version = "image-batch-fixture/v1";
  options.dataset_size = 3;
  options.batch_factory_id = "image-batch-fixture/admission/v1";
  options.batch_size = 1;
  options.num_workers = 2;
  options.prefetch_batches = 2;
  options.batch_timeout = std::chrono::seconds(2);
  options.schema = image_batch_schema(Batch({Photo()}), torch::kCPU);
  options.admission_factory = make_image_batch_admission_factory(budget);
  options.resource_lease_factory =
      make_image_batch_resource_lease_factory(budget);
  std::atomic<int> factory_calls = 0;
  std::atomic<int> first_sequence = -1;
  DataPipeline pipeline(
      std::make_shared<samplers::SequentialSampler>(3), options,
      [&](std::span<const std::size_t> indices,
          const DataWorkerContext& context, std::stop_token stop) {
        int expected = -1;
        first_sequence.compare_exchange_strong(
            expected, static_cast<int>(context.sequence));
        ++factory_calls;
        const std::array<VisionSample, 1> samples{Photo()};
        const std::array<ImageBatchKey, 1> keys{
            ImageBatchKey{std::to_string(indices.front()), context.epoch, 0}};
        return image_batch_value(collate_images(samples, keys, {}, stop));
      });
  pipeline.set_epoch(0);
  auto first = pipeline.next();
  ASSERT_TRUE(first);
  EXPECT_EQ(first_sequence.load(), 0);
  EXPECT_EQ(factory_calls.load(), 1);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            1U);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_EQ(factory_calls.load(), 1);
  first.reset();
  auto second = pipeline.next();
  ASSERT_TRUE(second);
  EXPECT_EQ(second->sequence, 1U);
  EXPECT_EQ(factory_calls.load(), 2);
  pipeline.close();
  EXPECT_EQ(factory_calls.load(), 2);
  second.reset();
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).peak,
            1U);
}

TEST(ImageBatchResourceBudgetTest,
     ByteAdmissionBoundsWorkersEvenWithAvailableBatchSlots) {
  ImageResourceBudgetLimits resource_limits;
  resource_limits.max_pipeline_inflight_batches = 3;
  resource_limits.max_pipeline_inflight_bytes = 64;
  auto budget = std::make_shared<ImageResourceBudget>(resource_limits);
  ImageBatchLimits batch_limits;
  batch_limits.max_output_bytes = 64;
  ImageBatchLimits invalid_limits = batch_limits;
  invalid_limits.max_output_bytes = 0;
  EXPECT_THROW(make_image_batch_byte_admission_factory(budget, invalid_limits),
               InvalidArgument);
  DataPipelineOptions options;
  options.pipeline_id = "image.batch.byte-admission-worker";
  options.dataset_version = "image-batch-fixture/v1";
  options.dataset_size = 3;
  options.batch_factory_id = "image-batch-fixture/byte-admission/v1";
  options.batch_size = 1;
  options.num_workers = 2;
  options.prefetch_batches = 2;
  options.batch_timeout = std::chrono::seconds(2);
  options.schema = image_batch_schema(Batch({Photo()}), torch::kCPU);
  options.admission_factory =
      make_image_batch_byte_admission_factory(budget, batch_limits);
  options.resource_lease_factory =
      make_image_batch_resource_lease_factory(budget, batch_limits);
  std::atomic<int> factory_calls = 0;
  DataPipeline pipeline(
      std::make_shared<samplers::SequentialSampler>(3), options,
      [&](std::span<const std::size_t> indices,
          const DataWorkerContext& context, std::stop_token stop) {
        ++factory_calls;
        const std::array<VisionSample, 1> samples{Photo()};
        const std::array<ImageBatchKey, 1> keys{
            ImageBatchKey{std::to_string(indices.front()), context.epoch, 0}};
        return image_batch_value(
            collate_images(samples, keys, {.limits = batch_limits}, stop));
      });
  pipeline.set_epoch(0);
  auto first = pipeline.next();
  ASSERT_TRUE(first);
  EXPECT_EQ(factory_calls.load(), 1);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_bytes).current,
            64U);
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  EXPECT_EQ(factory_calls.load(), 1);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_batches).current,
            1U);
  first.reset();
  auto second = pipeline.next();
  ASSERT_TRUE(second);
  EXPECT_EQ(factory_calls.load(), 2);
  pipeline.close();
  second.reset();
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_bytes).current,
            0U);
  EXPECT_EQ(budget->usage(ImageResourceKind::pipeline_inflight_bytes).peak,
            64U);
}

TEST(ImageBatchTest,
     ClassificationOffsetsAndMissingVersusEmptyAnnotationsRoundTrip) {
  auto a = Photo(), b = Photo(), c = Photo();
  const auto classes = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{3, "cat"}, {9, "dog"}});
  a.label_categories = b.label_categories = c.label_categories = classes;
  a.labels = {1, 0};
  c.labels = {0};
  Record(b);
  auto record = Record(c);
  record->objects.push_back({.source_id = 811,
                             .box = ImageBox{0, 0, 2, 1},
                             .keypoints = {{{1, 1}, 2}},
                             .area = 42,
                             .group_id = -11,
                             .crowd = true,
                             .ignore = true,
                             .difficult = true,
                             .truncated = true});
  auto keys = Keys(3);
  keys[1].occurrence = std::numeric_limits<std::uint64_t>::max();
  keys[2].view = 1ULL << 63U;
  const auto batch = collate_images(std::array{a, b, c}, keys);
  const auto value = image_batch_value(batch);
  validate_batch_value(value, image_batch_schema(batch, torch::kCPU));
  const auto result = image_batch_from_value(value);
  EXPECT_TRUE(
      torch::equal(result.labels, torch::tensor({1, 0, 0}, torch::kInt64)));
  EXPECT_EQ(result.label_offsets, (std::vector<std::int64_t>{0, 2, 2, 3}));
  EXPECT_FALSE(result.items[0].annotation);
  ASSERT_TRUE(result.items[1].annotation);
  EXPECT_TRUE(result.items[1].annotation->objects.empty());
  ASSERT_EQ(result.items[2].annotation->objects.size(), 1U);
  const auto& object = result.items[2].annotation->objects[0];
  EXPECT_EQ(object.source_id, 811);
  EXPECT_EQ(object.group_id, -11);
  EXPECT_TRUE(object.crowd && object.ignore && object.difficult &&
              object.truncated);
  EXPECT_EQ(object.area, 42);
  EXPECT_EQ(object.keypoints[0].visibility, 2);
  EXPECT_EQ(result.items[1].key.occurrence,
            std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(result.items[2].key.view, 1ULL << 63U);
  EXPECT_EQ(result.items[2].annotation->source_image_id,
            std::numeric_limits<std::int64_t>::min());
  EXPECT_EQ((*result.items[2].label_categories)[1].source_id, 9);
}

TEST(ImageBatchTest, PadUsesExplicitFillValidExtentsAndExistingContentTrace) {
  auto a = Photo(2, 2), b = Photo(3, 1);
  Mask(a);
  Mask(b);
  a.geometry = GeometryTrace({1, 2}).padded(0, 0, 1, 0);
  const auto original = a.mask_target->ids.clone();
  ImageCollateOptions options;
  options.layout = ImageBatchLayout::pad;
  options.size_divisor = 4;
  options.image_padding = 77;
  options.mask_padding_id = -7;
  const auto batch = Batch({a, b}, options);
  EXPECT_EQ(batch.images.sizes(), torch::IntArrayRef({2, 3, 4, 4}));
  EXPECT_EQ(batch.valid.sum().item<int>(), 7);
  EXPECT_TRUE(batch.valid[0][0][1]
                  .item<bool>());  // preexisting content padding remains valid
  EXPECT_FALSE(batch.items[0].geometry->inverse_point({1.5, 1}));
  EXPECT_FALSE(batch.valid[0][0][2].item<bool>());
  EXPECT_EQ(batch.images[1][0][3][3].item<int>(), 77);
  EXPECT_EQ((*batch.masks)[0][3][3].item<std::int64_t>(), -7);
  EXPECT_EQ((*batch.masks)[0][0][0].item<std::int64_t>(), 9007199254740993LL);
  EXPECT_TRUE(torch::equal(original, a.mask_target->ids));
  EXPECT_EQ(batch.items[0].valid_extent.width, 2U);
  EXPECT_EQ(batch.items[0].geometry->output_extent().width, 4U);
}

TEST(ImageBatchTest, RaggedOwnsDifferentSizesAndSeparatePairedPixelSemantics) {
  auto a = Photo(), b = Photo(1, 4);
  for (auto* s : {&a, &b}) {
    s->paired_target = s->image.to(torch::kFloat64).div(255);
    s->paired_target_pixels = s->pixels;
    s->paired_target_pixels->dtype = torch::kFloat64;
    s->paired_target_pixels->range = PixelValueRange::unit_interval;
    s->target_path = "paired.png";
  }
  const auto batch = Batch({a, b}, {.layout = ImageBatchLayout::ragged});
  EXPECT_FALSE(batch.images.defined());
  EXPECT_FALSE(batch.paired_images);
  ASSERT_EQ(batch.image_list.size(), 2U);
  ASSERT_EQ(batch.paired_image_list.size(), 2U);
  EXPECT_EQ(batch.valid_list[1].sizes(), torch::IntArrayRef({4, 1}));
  const auto result = image_batch_from_value(image_batch_value(batch));
  EXPECT_TRUE(torch::equal(result.paired_image_list[1], *b.paired_target));
  b.paired_target->zero_();
  EXPECT_GT(result.paired_image_list[1].sum().item<double>(), 0);
}

TEST(ImageBatchTest, PadRleMatchesIndependentPixelsForAllSmallBinaryMasks) {
  for (unsigned bits = 0; bits < 512; ++bits) {
    auto s = Photo(3, 3, 1);
    auto record = Record(s);
    std::vector<int> pixels(9);
    for (unsigned i = 0; i < 9; ++i) pixels[i] = (bits >> i) & 1U;
    auto runs = Encode(pixels, 3, 3);
    // Additional zero runs retain parity but must not lose or invent pixels.
    runs.counts.insert(runs.counts.begin(), {0, 0});
    record->objects.push_back(
        {.source_id = 17,
         .rle = runs,
         .area = 42,
         .mask_pixel_area = static_cast<std::uint64_t>(std::popcount(bits)),
         .crowd = true});
    const auto result = Batch({s}, {.layout = ImageBatchLayout::pad,
                                    .padded_extent = ImageExtent{5, 4}});
    const auto& object = result.items[0].annotation->objects[0];
    const auto actual = Decode(*object.rle);
    for (unsigned y = 0; y < 4; ++y)
      for (unsigned x = 0; x < 5; ++x)
        ASSERT_EQ(actual[y * 5 + x], x < 3 && y < 3 ? pixels[y * 3 + x] : 0)
            << bits;
    EXPECT_EQ(object.area, 42);
    EXPECT_TRUE(object.crowd);
    EXPECT_EQ(object.mask_pixel_area, std::popcount(bits));
    EXPECT_EQ(record->width, 3U);
    EXPECT_EQ(record->objects[0].rle->height, 3U);
  }
}

TEST(ImageBatchTest, PadRleHasHandDerivedColumnMajorRuns) {
  auto s = Photo(2, 2);
  auto r = Record(s);
  r->objects.push_back(
      {.rle = ImageMaskRle{2, 2, {1, 2, 1}}, .mask_pixel_area = 2});
  const auto batch = Batch({s}, {.layout = ImageBatchLayout::pad,
                                 .padded_extent = ImageExtent{3, 3}});
  EXPECT_EQ(batch.items[0].annotation->objects[0].rle->counts,
            (std::vector<std::uint64_t>{1, 1, 1, 1, 5}));
}

TEST(ImageBatchTest,
     AllRichGeometryAndConversionFlagsSurvivePaddingAndValueRoundTrip) {
  auto s = Photo(3, 3);
  auto r = Record(s);
  r->objects.push_back({.source_id = 8,
                        .polygons = {{{0, 0}, {2, 0}, {1, 2}}},
                        .oriented_box_to_polygon = true,
                        .polygon_partitioned = true});
  r->objects.push_back({.source_id = 9,
                        .oriented_box = std::array<ImagePoint, 4>{
                            {{0, 0}, {2, 0}, {2, 1}, {0, 1}}}});
  const auto batch = Batch({s}, {.layout = ImageBatchLayout::pad,
                                 .padded_extent = ImageExtent{4, 5}});
  const auto result = image_batch_from_value(image_batch_value(batch));
  const auto& record = *result.items[0].annotation;
  EXPECT_EQ(record.width, 4U);
  EXPECT_EQ(record.height, 5U);
  ASSERT_EQ(record.objects.size(), 2U);
  EXPECT_TRUE(record.objects[0].polygon_partitioned &&
              record.objects[0].oriented_box_to_polygon);
  EXPECT_DOUBLE_EQ(record.objects[0].polygons[0][2].y, 2);
  EXPECT_DOUBLE_EQ((*record.objects[1].oriented_box)[2].x, 2);
}

VisionSample Panoptic() {
  auto s = Photo(2, 2);
  Mask(s);
  auto r = Record(s);
  s.mask_target->ids = torch::tensor({{4, 4}, {0, -7}}, torch::kInt64);
  s.mask_target->encoding = MaskEncoding::panoptic_rgb;
  s.mask_target->void_id = 0;
  r->panoptic_mask_path = "segments.png";
  r->objects.push_back(
      {.source_id = 4, .box = ImageBox{0, 0, 2, 1}, .area = 2, .crowd = true});
  return s;
}
TEST(ImageBatchTest, PanopticBindsOnCpuAndPaddingMustUseVoid) {
  auto s = Panoptic();
  ImageCollateOptions options{.layout = ImageBatchLayout::pad,
                              .padded_extent = ImageExtent{3, 3},
                              .mask_padding_id = 0};
  const auto batch = Batch({s}, options);
  const auto result = image_batch_from_value(image_batch_value(batch));
  EXPECT_EQ((*result.masks)[0][0][1].item<std::int64_t>(), 4);
  EXPECT_EQ((*result.masks)[0][1][1].item<std::int64_t>(), -7);
  EXPECT_EQ((*result.masks)[0][2][2].item<std::int64_t>(), 0);
  const auto& record = *result.items[0].annotation;
  EXPECT_TRUE(record.panoptic_mask_materialized);
  EXPECT_EQ(record.objects[0].mask_pixel_area, 2);
  EXPECT_EQ(record.objects[0].area, 2);
  EXPECT_DOUBLE_EQ(record.objects[0].box->y_max, 1);
  EXPECT_FALSE(s.annotation->panoptic_mask_materialized);
  options.mask_padding_id = -7;
  EXPECT_THROW(Batch({s}, options), InvalidArgument);
  s.mask_target->ids[0][0] = 999;
  options.mask_padding_id = 0;
  EXPECT_THROW(Batch({s}, options), SerializationError);
}

TEST(ImageBatchResourceBudgetTest, PanopticCopiesTransferToFinalBatchBudget) {
  const auto sample = Panoptic();
  ImageCollateOptions options{.layout = ImageBatchLayout::pad,
                              .padded_extent = ImageExtent{3, 3},
                              .mask_padding_id = 0};
  auto budget = std::make_shared<ImageResourceBudget>();
  options.resource_budget = budget;
  auto batch = Batch({sample}, options);
  const auto live = budget->usage(ImageResourceKind::annotation_bytes);
  EXPECT_GT(live.current, 0U);
  EXPECT_GT(live.peak, live.current);
  EXPECT_TRUE(batch.items[0].annotation->panoptic_mask_materialized);
  batch = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::annotation_bytes).current, 0U);

  ImageResourceBudgetLimits limits;
  limits.max_annotation_bytes = 127;
  auto tight = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = tight;
  EXPECT_THROW(Batch({sample}, options), ResourceExhausted);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).peak, 0U);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).current, 0U);

  limits.max_annotation_bytes = live.peak - 1U;
  tight = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = tight;
  EXPECT_THROW(Batch({sample}, options), ResourceExhausted);
  EXPECT_EQ(tight->usage(ImageResourceKind::annotation_bytes).current, 0U);
}

TEST(ImageBatchTest, RejectsAmbiguousShapesTargetsFillsLabelsAndCancellation) {
  auto a = Photo(), b = Photo(2, 3);
  EXPECT_THROW(Batch({a, b}), InvalidArgument);
  EXPECT_THROW(Batch({a}, {.padded_extent = ImageExtent{4, 4}}),
               InvalidArgument);
  EXPECT_THROW(Batch({a}, {.layout = ImageBatchLayout::pad,
                           .padded_extent = ImageExtent{2, 2}}),
               InvalidArgument);
  EXPECT_THROW(Batch({a}, {.image_padding = 0.5}), InvalidArgument);
  EXPECT_THROW(
      Batch({a}, {.image_padding = std::numeric_limits<double>::infinity()}),
      InvalidArgument);
  Mask(a);
  EXPECT_THROW(Batch({a, b}, {.layout = ImageBatchLayout::ragged}),
               InvalidArgument);
  EXPECT_THROW(Batch({a}, {.layout = ImageBatchLayout::pad, .size_divisor = 4}),
               InvalidArgument);
  a = Photo();
  a.labels = {0};
  EXPECT_THROW(Batch({a}), InvalidArgument);
  a = Photo();
  a.image = a.image.to(torch::kFloat32).requires_grad_(true);
  a.pixels.dtype = torch::kFloat32;
  a.pixels.range = PixelValueRange::unbounded;
  EXPECT_THROW(Batch({a}), InvalidArgument);
  std::stop_source cancel;
  cancel.request_stop();
  EXPECT_THROW(
      collate_images(std::array{Photo()}, Keys(1), {}, cancel.get_token()),
      Cancelled);
  EXPECT_THROW(collate_images({}, {}), InvalidArgument);
}

TEST(ImageBatchTest,
     RejectsMetadataTensorPixelAndWorkingBudgetBeforePublication) {
  const auto s = Photo();
  for (const int field : {0, 1, 2, 3, 4, 5}) {
    ImageCollateOptions options;
    if (field == 0) options.limits.max_output_bytes = 18;
    if (field == 1) options.limits.max_pixels = 5;
    if (field == 2) options.limits.max_dimension = 2;
    if (field == 3) options.limits.max_metadata_nodes = 32;
    if (field == 4) options.limits.max_metadata_bytes = 100;
    if (field == 5) options.limits.max_working_bytes = 100;
    EXPECT_THROW(Batch({s}, options), ResourceExhausted) << field;
  }
  auto malformed = s;
  auto r = Record(malformed);
  r->objects.push_back({.rle = ImageMaskRle{3, 2, {5}}});
  EXPECT_THROW(Batch({malformed}), InvalidArgument);
  r->objects[0].rle =
      ImageMaskRle{3, 2, {std::numeric_limits<std::uint64_t>::max()}};
  EXPECT_THROW(Batch({malformed}), InvalidArgument);
}

TEST(ImageBatchTest,
     ValueRejectsWrongVersionFieldsArityTensorTypeAndSemantics) {
  const auto batch = Batch({Photo()});
  const auto mutate = [&](const char* key, c10::IValue replacement) {
    auto value = image_batch_value(batch);
    auto dictionary = value.toGenericDict();
    dictionary.insert_or_assign(key, std::move(replacement));
    EXPECT_THROW(image_batch_from_value(value), InvalidArgument) << key;
  };
  mutate("format", 2);
  mutate("layout", 77);
  mutate("extra", 1);
  mutate("images", batch.images.to(torch::kFloat32));
  mutate("valid", torch::ones({1, 2, 3}, torch::kInt64));
  mutate("valid", torch::ones({0, 2, 3}, torch::kBool));
  mutate("pixels", c10::ivalue::Tuple::create({1, 2}));
  mutate("label_offsets", c10::List<std::int64_t>({0, 1}));
  auto bad = batch;
  bad.items[0].valid_extent.width = 4;
  EXPECT_THROW(image_batch_value(bad), InvalidArgument);
  ImageBatchLimits limits;
  limits.max_metadata_nodes = 30;
  EXPECT_THROW(image_batch_from_value(image_batch_value(batch), limits),
               ResourceExhausted);
  limits = {};
  limits.max_output_bytes = 1;
  EXPECT_THROW(image_batch_from_value(image_batch_value(batch), limits),
               ResourceExhausted);
}

TEST(ImageBatchTransformTest,
     PackedNormalizationUsesIndependentAnalyticValues) {
  auto a = Photo(), b = Photo();
  Mask(a);
  Mask(b);
  const auto batch = Batch({a, b});
  ImageBatchTransformOptions options;
  options.image.dtype = torch::kFloat64;
  options.image.swap_red_blue = true;
  options.image.mean = {0.1, 0.2, 0.3};
  options.image.std = {0.25, 0.5, 2};
  const auto result = transform_image_batch(batch, options);
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x) {
          const double encoded = (2 - c) * 6 + y * 3 + x;
          EXPECT_NEAR(
              result.images[n][c][y][x].item<double>(),
              (encoded / 255 - options.image.mean[c]) / options.image.std[c],
              1e-14);
        }
  EXPECT_EQ(result.pixels.color, PixelColorModel::bgr);
  EXPECT_EQ(result.pixels.range, PixelValueRange::unbounded);
  EXPECT_TRUE(torch::equal(*result.masks, *batch.masks));
  EXPECT_EQ(result.masks->data_ptr(), batch.masks->data_ptr());
  EXPECT_EQ(batch.images.scalar_type(), torch::kUInt8);
  EXPECT_EQ(batch.images[0][2][1][2].item<int>(), 17);
  EXPECT_NE(result.images.data_ptr(), batch.images.data_ptr());
}

TEST(ImageBatchTransformTest, Uint16ScaleAndRaggedPairedPoliciesAreExplicit) {
  auto a = Photo(2, 1, 1), b = Photo(1, 3, 1);
  for (auto* s : {&a, &b}) {
    s->image = torch::full({1, s->height, s->width}, 65535, torch::kUInt16);
    s->pixels.dtype = torch::kUInt16;
    s->paired_target =
        torch::full({1, s->height, s->width}, 127, torch::kUInt8);
    s->paired_target_pixels = s->pixels;
    s->paired_target_pixels->dtype = torch::kUInt8;
  }
  const auto batch = Batch({a, b}, {.layout = ImageBatchLayout::ragged});
  ImageBatchTransformOptions options;
  options.paired = BatchPixelTransform{.dtype = torch::kFloat64,
                                       .scale = BatchPixelScale::preserve};
  const auto result = transform_image_batch(batch, options);
  EXPECT_EQ(result.image_list[0][0][0][0].item<float>(), 1);
  EXPECT_EQ(result.paired_image_list[1][0][2][0].item<double>(), 127);
  EXPECT_EQ(result.paired_pixels->range, PixelValueRange::unbounded);
  EXPECT_EQ(result.pixels.range, PixelValueRange::unit_interval);
  const auto main_only = transform_image_batch(batch);
  EXPECT_EQ(main_only.paired_image_list[0].data_ptr(),
            batch.paired_image_list[0].data_ptr());
}

TEST(ImageBatchTransformTest,
     LowPrecisionCastsUseFloatComputeAndUnitInputIsNotRescaled) {
  auto s = Photo(2, 1, 1);
  s.image = torch::tensor({0.25, 0.75}, torch::kFloat32).reshape({1, 1, 2});
  s.pixels.dtype = torch::kFloat32;
  s.pixels.range = PixelValueRange::unit_interval;
  const auto batch = Batch({s});
  for (const auto dtype :
       {torch::kFloat16, torch::kBFloat16, torch::kFloat32, torch::kFloat64}) {
    ImageBatchTransformOptions options;
    options.image.dtype = dtype;
    options.image.mean = {0.5};
    options.image.std = {0.25};
    const auto result = transform_image_batch(batch, options);
    EXPECT_EQ(result.images.scalar_type(), dtype);
    EXPECT_EQ(result.images[0][0][0][0].item<double>(), -1);
    EXPECT_EQ(result.images[0][0][0][1].item<double>(), 1);
  }
}

TEST(ImageBatchTransformTest,
     AlphaAssociationIsPreservedAndNormalizationIsExplicitlyRejected) {
  auto s = Photo(1, 1);
  s.image = torch::tensor({16, 32, 64, 128}, torch::kUInt8).reshape({4, 1, 1});
  s.pixels.color = PixelColorModel::rgba;
  s.pixels.alpha = PixelAlphaMode::premultiplied;
  const auto batch = Batch({s});
  ImageBatchTransformOptions o;
  o.image.dtype = torch::kFloat64;
  o.image.swap_red_blue = true;
  const auto result = transform_image_batch(batch, o);
  EXPECT_EQ(result.pixels.color, PixelColorModel::bgra);
  EXPECT_EQ(result.pixels.alpha, PixelAlphaMode::premultiplied);
  EXPECT_DOUBLE_EQ(result.images[0][0][0][0].item<double>(), 64.0 / 255);
  EXPECT_DOUBLE_EQ(result.images[0][3][0][0].item<double>(), 128.0 / 255);
  o.image.mean = {0, 0, 0, 0};
  o.image.std = {1, 1, 1, 1};
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
}

TEST(ImageBatchTransformTest, PackedPairedPhotosUseIndependentNumericPolicies) {
  auto s = Photo(2, 1, 1);
  s.paired_target = torch::full({1, 1, 2}, 0.75, torch::kFloat64);
  s.paired_target_pixels = s.pixels;
  s.paired_target_pixels->dtype = torch::kFloat64;
  s.paired_target_pixels->range = PixelValueRange::unit_interval;
  const auto batch = Batch({s, s});
  ImageBatchTransformOptions o;
  o.paired = BatchPixelTransform{
      .dtype = torch::kFloat64, .mean = {0.5}, .std = {0.25}};
  const auto result = transform_image_batch(batch, o);
  EXPECT_DOUBLE_EQ((*result.paired_images)[1][0][0][1].item<double>(), 1);
  EXPECT_DOUBLE_EQ((*batch.paired_images)[1][0][0][1].item<double>(), 0.75);
  EXPECT_NE(result.paired_images->data_ptr(), batch.paired_images->data_ptr());
  EXPECT_EQ(result.pixels.range, PixelValueRange::unit_interval);
  EXPECT_EQ(result.paired_pixels->range, PixelValueRange::unbounded);
}

TEST(ImageBatchTransformTest,
     RejectsInvalidParametersAndBudgetsWithoutMutatingInput) {
  const auto batch = Batch({Photo()});
  ImageBatchTransformOptions o;
  o.image.mean = {0};
  o.image.std = {1};
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o.image.mean = {0, 0, 0};
  o.image.std = {1, 0, 1};
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o = {};
  o.image.dtype = torch::kInt64;
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o = {};
  o.paired.emplace();
  EXPECT_THROW(transform_image_batch(batch, o), InvalidArgument);
  o = {};
  o.limits.max_output_bytes = 24;
  EXPECT_THROW(transform_image_batch(batch, o), ResourceExhausted);
  o = {};
  o.limits.max_working_bytes = 1200;
  EXPECT_THROW(transform_image_batch(batch, o), ResourceExhausted);
  auto normalized = transform_image_batch(
      batch, {.image = {.scale = BatchPixelScale::preserve}});
  EXPECT_THROW(transform_image_batch(normalized), InvalidArgument);
  std::stop_source cancel;
  cancel.request_stop();
  EXPECT_THROW(transform_image_batch(batch, {}, cancel.get_token()), Cancelled);
  EXPECT_EQ(batch.images[0][2][1][2].item<int>(), 17);
}

TEST(ImageBatchTest, IdentityBindsAllPoliciesAndSchemaBindsPixelMeaning) {
  ImageCollateOptions c;
  const auto original = image_batch_identity(c);
  // Independently encoded with Python hashlib/struct, without C++ helpers.
  EXPECT_EQ(original,
            "a101f266d838ae1d48e5b6a4048c0336d8b1d7880a5dd292ba6fd1c03a044ffc");
  c.image_padding = -0.0;
  EXPECT_EQ(image_batch_identity(c), original);
  c.limits.max_geometry_checks--;
  EXPECT_NE(image_batch_identity(c), original);
  c = {};
  c.mask_padding_id = 0;
  EXPECT_NE(image_batch_identity(c), original);
  const auto numeric = image_batch_identity({}, ImageBatchTransformOptions{});
  EXPECT_NE(original, numeric);
  ImageBatchTransformOptions n;
  n.image.mean = {0, 0, 0};
  n.image.std = {1, 1, 1};
  const auto norm = image_batch_identity({}, n);
  n.image.std[2] = 2;
  EXPECT_NE(image_batch_identity({}, n), norm);
  n = {};
  n.limits.max_metadata_nodes--;
  EXPECT_NE(image_batch_identity({}, n), numeric);
  auto a = Batch({Photo()}), b = a;
  b.pixels.color = PixelColorModel::bgr;
  EXPECT_NE(canonical_batch_schema(image_batch_schema(a, torch::kCPU)),
            canonical_batch_schema(image_batch_schema(b, torch::kCPU)));
  b = a;
  b.layout = ImageBatchLayout::pad;
  EXPECT_NE(canonical_batch_schema(image_batch_schema(a, torch::kCPU)),
            canonical_batch_schema(image_batch_schema(b, torch::kCPU)));
}

TEST(ImageBatchContentIdentityTest, BindsTensorValuesAndBatchSemantics) {
  const auto photo = Batch({Photo()});
  const auto identity = image_batch_content_identity(photo);
  EXPECT_EQ(identity,
            "8f5a8e68d386f924bd2b5558f762fd814dd177b87773e8b90e6e63082427efa2");
  EXPECT_EQ(identity, image_batch_content_identity(photo));

  auto changed_photo = photo;
  changed_photo.images = photo.images.clone();
  changed_photo.images[0][0][0][0] = 99;
  EXPECT_NE(image_batch_content_identity(changed_photo),
            image_batch_content_identity(photo));

  auto mask_sample = Photo();
  Mask(mask_sample);
  const auto with_mask = Batch({mask_sample});
  auto changed_mask = with_mask;
  changed_mask.masks = with_mask.masks->clone();
  changed_mask.masks->index_put_({0, 0, 0}, 123);
  EXPECT_NE(image_batch_content_identity(changed_mask),
            image_batch_content_identity(with_mask));

  auto metadata_sample = Photo();
  auto categories = std::make_shared<const std::vector<VisionCategory>>(
      std::vector<VisionCategory>{{.source_id = 1, .name = "object"}});
  metadata_sample.categories = categories;
  auto annotation = std::make_shared<VisionAnnotationRecord>();
  annotation->image_path = metadata_sample.image_path;
  annotation->width = metadata_sample.width;
  annotation->height = metadata_sample.height;
  annotation->objects.push_back(
      {.source_id = 7, .category = 0, .box = ImageBox{0, 0, 2, 1}});
  metadata_sample.annotation = annotation;
  const auto annotated = Batch({metadata_sample});
  auto changed_annotation = annotated;
  auto changed_record = std::make_shared<VisionAnnotationRecord>(
      *changed_annotation.items[0].annotation);
  changed_record->objects[0].box->x_max = 1;
  changed_annotation.items[0].annotation = changed_record;
  EXPECT_NE(image_batch_content_identity(changed_annotation),
            image_batch_content_identity(annotated));

  ImageCollateOptions padded;
  padded.layout = ImageBatchLayout::pad;
  padded.padded_extent = ImageExtent{3, 2};
  const auto padded_batch = Batch({Photo()}, padded);
  EXPECT_NE(image_batch_content_identity(photo),
            image_batch_content_identity(padded_batch));
}

TEST(ImageBatchContentIdentityTest, ValidatesLimitsAndCancellation) {
  const auto batch = Batch({Photo()});
  std::stop_source stop;
  stop.request_stop();
  EXPECT_THROW(image_batch_content_identity(batch, {}, stop.get_token()),
               Cancelled);

  ImageBatchLimits limits;
  limits.max_output_bytes = 1;
  EXPECT_THROW(image_batch_content_identity(batch, limits), ResourceExhausted);
}

TEST(ImageBatchCUDATest, FinalBatchPinTransferNumericAndRaggedMetadata) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto layout : {ImageBatchLayout::stack, ImageBatchLayout::pad,
                            ImageBatchLayout::ragged}) {
    auto a = Photo(), b = Photo();
    Mask(a);
    Mask(b);
    ImageCollateOptions collate;
    collate.layout = layout;
    if (layout == ImageBatchLayout::pad) {
      collate.size_divisor = 4;
      collate.mask_padding_id = -7;
    }
    const auto batch = Batch({a, b}, collate);
    BatchMemoryOptions memory{.pin_memory = true,
                              .target_device = Device(torch::kCUDA, 0),
                              .non_blocking = true};
    const auto prepared = prepare_image_batch(batch, memory);
    const auto& pinned = layout == ImageBatchLayout::ragged
                             ? prepared.image_list[0]
                             : prepared.images;
    EXPECT_TRUE(pinned.device().is_cpu());
    EXPECT_TRUE(pinned.is_pinned());
    EXPECT_EQ(prepared.memory.pinned_tensor_count,
              layout == ImageBatchLayout::ragged ? 6U : 3U);
    const auto gpu = transfer_image_batch(prepared, memory);
    ImageBatchTransformOptions numeric;
    numeric.image.mean = {0.1, 0.2, 0.3};
    numeric.image.std = {0.25, 0.5, 2};
    const auto converted = transform_image_batch(gpu, numeric);
    EXPECT_FALSE(converted.memory.staged);
    const auto round_trip =
        image_batch_from_value(image_batch_value(converted));
    const auto cpu = transform_image_batch(batch, numeric);
    const auto& actual = layout == ImageBatchLayout::ragged
                             ? round_trip.image_list[0]
                             : round_trip.images;
    const auto& expected =
        layout == ImageBatchLayout::ragged ? cpu.image_list[0] : cpu.images;
    EXPECT_TRUE(actual.device().is_cuda());
    EXPECT_TRUE(torch::allclose(actual.cpu(), expected, 1e-6, 1e-7));
    const auto& mask =
        layout == ImageBatchLayout::ragged ? gpu.mask_list[0] : *gpu.masks;
    const auto& original =
        layout == ImageBatchLayout::ragged ? batch.mask_list[0] : *batch.masks;
    EXPECT_TRUE(torch::equal(mask.cpu(), original));
    EXPECT_EQ(round_trip.items[0].key.id, "sample-0");
  }
}

TEST(ImageBatchCUDATest, ContentIdentityRejectsDeviceBatchWithoutTransfer) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto batch = Batch({Photo()});
  BatchMemoryOptions memory{.pin_memory = true,
                            .target_device = Device(torch::kCUDA, 0),
                            .non_blocking = true};
  const auto device_batch =
      transfer_image_batch(prepare_image_batch(batch, memory), memory);
  EXPECT_THROW(image_batch_content_identity(device_batch), InvalidArgument);
}

TEST(ImageBatchCUDATest, NumericReservesGpuOutputAndWorkingStorage) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto batch = Batch({Photo()});
  BatchMemoryOptions memory{.pin_memory = true,
                            .target_device = Device(torch::kCUDA, 0),
                            .non_blocking = true};
  const auto device_batch =
      transfer_image_batch(prepare_image_batch(batch, memory), memory);
  ImageBatchTransformOptions options;
  options.image.scale = BatchPixelScale::preserve;
  ImageResourceBudgetLimits limits;
  limits.max_gpu_bytes = 221;
  auto budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  EXPECT_THROW(transform_image_batch(device_batch, options), ResourceExhausted);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 78U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);

  limits.max_gpu_bytes = 222;
  budget = std::make_shared<ImageResourceBudget>(limits);
  options.resource_budget = budget;
  auto result = transform_image_batch(device_batch, options);
  EXPECT_TRUE(result.images.device().is_cuda());
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).peak, 222U);
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 78U);
  result = {};
  EXPECT_EQ(budget->usage(ImageResourceKind::gpu_bytes).current, 0U);
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchCUDATest, NumericAndTransferUseCallerCurrentNondefaultStream) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto batch = Batch({Photo(), Photo()});
  BatchMemoryOptions memory{.pin_memory = true,
                            .target_device = Device(torch::kCUDA, 0),
                            .non_blocking = true};
  const auto pinned = prepare_image_batch(batch, memory);
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  auto transferred = transfer_image_batch(pinned, memory);
  transferred.images.fill_(255);  // Ordered producer operation on this stream.
  ImageBatchTransformOptions numeric;
  numeric.image.mean = {0.5, 0.5, 0.5};
  numeric.image.std = {0.25, 0.25, 0.25};
  const auto result = transform_image_batch(transferred, numeric);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
  EXPECT_EQ(result.images.sum().item<float>(), 72);  // synchronize only in test
}
#endif

std::unique_ptr<DataPipeline> Pipeline(std::size_t workers, bool fail = false) {
  DataPipelineOptions options;
  options.dataset_size = 5;
  options.dataset_version = "image-batch-fixture/v1";
  options.batch_factory_id = "image-batch-fixture/stack-v1";
  options.batch_size = 2;
  options.num_workers = workers;
  options.schema = image_batch_schema(Batch({Photo()}), torch::kCPU);
  return std::make_unique<DataPipeline>(
      std::make_shared<samplers::SequentialSampler>(5), options,
      [fail](std::span<const std::size_t> indices,
             const DataWorkerContext& context, std::stop_token stop) {
        if (fail && context.sequence == 1)
          throw InvalidArgument("injected image collator failure");
        std::vector<VisionSample> samples;
        std::vector<ImageBatchKey> keys;
        for (const auto index : indices) {
          samples.push_back(Photo());
          samples.back().image.fill_(static_cast<int>(index));
          keys.push_back({std::to_string(index), context.epoch, 0});
        }
        return image_batch_value(collate_images(samples, keys, {}, stop));
      });
}

class SavedState final : public StateReader, public StateWriter {
 public:
  explicit SavedState(StateManifest manifest)
      : manifest_(std::move(manifest)) {}
  const StateManifest& manifest() const noexcept override { return manifest_; }
  bool contains(std::string_view name) const override {
    return blobs_.contains(std::string(name));
  }
  std::span<const std::byte> read_blob(std::string_view name,
                                       std::uint64_t maximum) const override {
    const auto& b = blobs_.at(std::string(name));
    if (b.size() > maximum) throw std::runtime_error("state size");
    return b;
  }
  void write_blob(std::string_view name,
                  std::span<const std::byte> bytes) override {
    blobs_[std::string(name)] = {bytes.begin(), bytes.end()};
  }

 private:
  StateManifest manifest_;
  std::map<std::string, std::vector<std::byte>> blobs_;
};

TEST(ImageBatchPipelineTest, FullSchemaStableKeysTailAndSameTopologyRestore) {
  auto inline_pipeline = Pipeline(0), workers = Pipeline(2);
  inline_pipeline->set_epoch(8);
  workers->set_epoch(8);
  ASSERT_TRUE(workers->next());
  ASSERT_TRUE(inline_pipeline->next());
  SavedState state(workers->describe_state());
  workers->save_state(state);
  auto resumed = Pipeline(2);
  resumed->load_state(state, RestorePolicy::exact_only);
  while (const auto expected = inline_pipeline->next()) {
    const auto actual = resumed->next();
    ASSERT_TRUE(actual);
    const auto a = image_batch_from_value(actual->value),
               b = image_batch_from_value(expected->value);
    EXPECT_TRUE(torch::equal(a.images, b.images));
    ASSERT_EQ(a.items.size(), b.items.size());
    EXPECT_EQ(a.items[0].key.id, b.items[0].key.id);
    EXPECT_EQ(a.items[0].key.occurrence, 8U);
  }
  EXPECT_FALSE(resumed->next());
  EXPECT_EQ(resumed->state().next_cursor, 5U);
  EXPECT_THROW(inline_pipeline->load_state(state, RestorePolicy::exact_only),
               CompatibilityError);
}

TEST(ImageBatchPipelineTest,
     FactoryFailurePreservesCommittedCursorAndCanReplay) {
  for (const auto workers : {0U, 2U}) {
    auto pipeline = Pipeline(workers, true);
    pipeline->set_epoch(8);
    ASSERT_TRUE(pipeline->next());
    EXPECT_THROW(pipeline->next(), InvalidArgument);
    EXPECT_EQ(pipeline->state().next_cursor, 2U);
    SavedState state(pipeline->describe_state());
    pipeline->save_state(state);
    auto resumed = Pipeline(workers);
    resumed->load_state(state, RestorePolicy::exact_only);
    const auto next = resumed->next();
    ASSERT_TRUE(next);
    EXPECT_EQ(image_batch_from_value(next->value).items[0].key.id, "2");
  }
}

std::unique_ptr<DataPipeline> NumericPipeline(std::size_t workers,
                                              Device device,
                                              std::shared_ptr<bool> fail) {
  DataPipelineOptions config;
  config.pipeline_id = "image.numeric.precommit";
  config.dataset_version = "image-batch-fixture/v1";
  config.dataset_size = 4;
  config.batch_factory_id = "image-batch-fixture/with-mask/v1";
  config.batch_size = 2;
  config.num_workers = workers;
  auto photo = Photo();
  Mask(photo);
  const auto input = Batch({photo, photo});
  ImageBatchTransformOptions numeric;
  numeric.image.mean = {0.5, 0.5, 0.5};
  numeric.image.std = {0.25, 0.25, 0.25};
  config.schema =
      image_batch_schema(transform_image_batch(input, numeric), device);
  if (device.is_cuda()) {
    config.memory = {
        .pin_memory = true, .target_device = device, .non_blocking = true};
  }
  config.consumer_transform = BatchConsumerTransform{
      .id = "image-numeric/" + image_batch_identity({}, numeric),
      .input_schema = image_batch_schema(input, device),
      .apply = [numeric, fail](const c10::IValue& v,
                               const BatchConsumerContext& c,
                               std::stop_token stop) {
        if (*fail && c.sequence == 1)
          throw InvalidArgument("injected numeric precommit failure");
        return image_batch_value(
            transform_image_batch(image_batch_from_value(v), numeric, stop));
      }};
  return std::make_unique<DataPipeline>(
      std::make_shared<samplers::SequentialSampler>(4), config,
      [](std::span<const std::size_t> ids, const DataWorkerContext& c,
         std::stop_token stop) {
        std::vector<VisionSample> samples;
        std::vector<ImageBatchKey> keys;
        for (auto id : ids) {
          auto sample = Photo();
          sample.image.fill_(255);
          Mask(sample);
          samples.push_back(std::move(sample));
          keys.push_back({std::to_string(id), c.epoch, 0});
        }
        return image_batch_value(collate_images(samples, keys, {}, stop));
      });
}

TEST(ImageBatchPipelineTest,
     NumericPrecommitFailureRestoresWithoutLosingMaskIds) {
  for (const auto workers : {0U, 2U}) {
    auto fail = std::make_shared<bool>(true);
    auto p = NumericPipeline(workers, torch::kCPU, fail);
    p->set_epoch(7);
    const auto first = p->next();
    ASSERT_TRUE(first);
    EXPECT_TRUE(torch::equal(image_batch_from_value(first->value).images,
                             torch::full({2, 3, 2, 3}, 2.0F)));
    EXPECT_THROW(p->next(), InvalidArgument);
    EXPECT_EQ(p->state().next_cursor, 2U);
    EXPECT_EQ(p->outstanding_batches(), 0U);
    SavedState saved(p->describe_state());
    p->save_state(saved);
    *fail = false;
    p->load_state(saved, RestorePolicy::exact_only);
    const auto next = p->next();
    ASSERT_TRUE(next);
    const auto decoded = image_batch_from_value(next->value);
    EXPECT_EQ(next->sequence, 1U);
    EXPECT_EQ(decoded.items[0].key.id, "2");
    EXPECT_EQ(decoded.items[0].key.occurrence, 7U);
    EXPECT_EQ((*decoded.masks)[0][0][0].item<std::int64_t>(),
              9007199254740993LL);
    ASSERT_TRUE(next->transformed_memory);
    EXPECT_EQ(next->transformed_memory->tensor_bytes,
              2U * 3U * 2U * 3U * 4U + 2U * 2U * 3U * 8U + 2U * 2U * 3U);
  }
}

#if defined(JYPPX_LIUDING_IMAGE_TEST_CUDA)
TEST(ImageBatchCUDATest,
     PipelinePrecommitRunsAfterH2dOnNondefaultCallerStream) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  for (const auto workers : {0U, 2U}) {
    auto p = NumericPipeline(workers, Device(torch::kCUDA, 0),
                             std::make_shared<bool>(false));
    p->set_epoch(7);
    const auto stream = c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard guard(stream);
    const auto e = p->next();
    ASSERT_TRUE(e);
    const auto decoded = image_batch_from_value(e->value);
    EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
    EXPECT_TRUE(decoded.images.device().is_cuda());
    EXPECT_EQ(decoded.images.sum().item<float>(), 72.0F);
    EXPECT_EQ((*decoded.masks)[0][0][0].item<std::int64_t>(),
              9007199254740993LL);
    EXPECT_TRUE(e->memory.staged);
    EXPECT_EQ(e->memory.transferred_tensor_count, 4U);  // includes empty labels
    EXPECT_EQ(e->memory.pinned_tensor_count, 3U);
    EXPECT_EQ(e->memory.tensor_bytes, 36U + 96U + 12U);
    ASSERT_TRUE(e->transformed_memory);
    EXPECT_EQ(e->transformed_memory->tensor_bytes, 144U + 96U + 12U);
    EXPECT_EQ(e->transformed_memory->tensor_count, 4U);
  }
}
#endif
}  // namespace
