// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "preprocessing_reference.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/models/registry.hpp>
#include <jyppx/liuding/models/weights.hpp>

#include <algorithm>
#include <array>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

#if defined(JYPPX_LIUDING_PREPROCESS_IMAGE_TESTS)
#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#endif
#if defined(JYPPX_LIUDING_PREPROCESS_CUDA_TESTS)
#include <c10/cuda/CUDAGuard.h>
#endif

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;

WeightInputSpec InputSpec() {
  WeightInputSpec spec;
  spec.source_revision = "0123456789abcdef";
  spec.state_mapping_version = "fixture-map-v1";
  spec.reference_uri = "fixture://recipe/v1";
  spec.reference_revision = "fedcba9876543210";
  spec.image.input_size = {3, 4};
  spec.image.resize = PreprocessResize::shortest_edge_center_crop;
  spec.image.resize_shorter_side = 4;
  spec.image.interpolation = PreprocessInterpolation::bicubic;
  spec.image.color = PreprocessColor::bgr;
  spec.image.mean = {0.1, 0.2, 0.3};
  spec.image.std = {0.5, 0.25, 0.125};
  spec.categories =
      std::vector<ClassificationCategory>{{"cat", "猫"}, {"dog", "狗"}};
  return spec;
}

WeightRecord Record() {
  return {.id = "fixture-v1",
          .model_name = "fixture",
          .source_uri = "fixture://weights/v1",
          .license_spdx = "Apache-2.0",
          .license_uri = "fixture://license/v1",
          .availability = WeightAvailability::pinned_external,
          .artifact = ArtifactRef{.digest = {.hex = std::string(64, '0')},
                                  .size_bytes = 123,
                                  .logical_name = "fixture.safetensors",
                                  .media_type = "application/x-safetensors",
                                  .producer = "independent-fixture"},
          .status_reason = "Metadata fixture; not actual provisioned weights.",
          .input_spec = InputSpec()};
}

TEST(ModelsPreprocessing, IdentityMatchesIndependentCanonicalByteReference) {
  const auto record = Record();
  EXPECT_EQ(weight_input_identity(record),
            preprocessing_reference::kInputIdentity);
  EXPECT_EQ(&weight_input_spec(record), &*record.input_spec);
  auto changed = record;
  changed.status_reason = "Diagnostic wording does not change interpretation.";
  EXPECT_EQ(weight_input_identity(record), weight_input_identity(changed));
  changed.input_spec->image.mean[0] = 0.0;
  const auto zero = weight_input_identity(changed);
  changed.input_spec->image.mean[0] = -0.0;
  EXPECT_EQ(weight_input_identity(changed), zero);
}

TEST(ModelsPreprocessing, IdentityBindsWeightRecipeProvenanceAndCategoryOrder) {
  const auto original = Record();
  const auto expected = weight_input_identity(original);
  const std::vector<std::function<void(WeightRecord&)>> mutations{
      [](auto& r) { r.id += "2"; },
      [](auto& r) { r.model_name += "2"; },
      [](auto& r) { r.source_uri += "2"; },
      [](auto& r) { r.license_spdx = "MIT"; },
      [](auto& r) { r.license_uri += "2"; },
      [](auto& r) { r.artifact->digest.hex[0] = '1'; },
      [](auto& r) { ++r.artifact->size_bytes; },
      [](auto& r) { r.artifact->logical_name += "2"; },
      [](auto& r) { r.artifact->producer += "2"; },
      [](auto& r) { r.input_spec->source_revision += "2"; },
      [](auto& r) { r.input_spec->state_mapping_version += "2"; },
      [](auto& r) { r.input_spec->reference_uri += "2"; },
      [](auto& r) { r.input_spec->reference_revision += "2"; },
      [](auto& r) { r.input_spec->image.input_size[0] = 2; },
      [](auto& r) { r.input_spec->image.resize_shorter_side = 5; },
      [](auto& r) {
        r.input_spec->image.interpolation = PreprocessInterpolation::bilinear;
      },
      [](auto& r) { r.input_spec->image.color = PreprocessColor::rgb; },
      [](auto& r) { r.input_spec->image.mean[0] += 0.1; },
      [](auto& r) { r.input_spec->image.std[0] += 0.1; },
      [](auto& r) {
        std::reverse(r.input_spec->categories->begin(),
                     r.input_spec->categories->end());
      },
      [](auto& r) { (*r.input_spec->categories)[0].id += "2"; },
      [](auto& r) { (*r.input_spec->categories)[0].name += "2"; },
      [](auto& r) { r.input_spec->categories.reset(); },
      [](auto& r) {
        r.input_spec->image.resize = PreprocessResize::stretch;
        r.input_spec->image.resize_shorter_side = 0;
      },
  };
  for (std::size_t i = 0; i < mutations.size(); ++i) {
    auto changed = original;
    mutations[i](changed);
    EXPECT_NE(weight_input_identity(changed), expected) << i;
  }
}

TEST(ModelsPreprocessing, MissingRecipesStayExplicitForEveryBuiltinCandidate) {
  std::size_t visual = 0;
  for (const auto& info : list_models()) {
    const auto candidates = available_weights(info.name);
    for (const auto& record : candidates) {
      EXPECT_FALSE(record.input_spec) << info.name;
      EXPECT_EQ(record.availability, WeightAvailability::not_provisioned);
      EXPECT_FALSE(record.artifact);
      EXPECT_EQ(record.license_spdx, "NOASSERTION");
      EXPECT_THROW(weight_input_spec(record), NotReady);
      EXPECT_THROW(weight_input_identity(record), NotReady);
    }
    if (!info.features.empty()) ++visual;
  }
  EXPECT_EQ(visual, 33U);
  auto record = Record();
  record.input_spec.reset();
  EXPECT_NO_THROW(validate_weight_record(record));
  EXPECT_THROW(weight_input_spec(record), NotReady);
  record.input_spec = InputSpec();
  record.availability = WeightAvailability::not_provisioned;
  record.artifact.reset();
  record.license_spdx = "NOASSERTION";
  EXPECT_NO_THROW(weight_input_spec(record));
  EXPECT_THROW(weight_input_identity(record), NotReady);
}

TEST(ModelsPreprocessing, GeometryResolvesPortraitLandscapeAndNearestEvenCrop) {
  auto spec = InputSpec().image;
  const auto landscape = resolve_image_preprocess_geometry(spec, {2, 3});
  EXPECT_EQ(landscape.resized_size, (std::array<std::uint32_t, 2>{4, 6}));
  EXPECT_EQ(landscape.output_size, spec.input_size);
  EXPECT_EQ(landscape.crop_top, 0U);
  EXPECT_EQ(landscape.crop_left, 1U);
  const auto portrait = resolve_image_preprocess_geometry(spec, {3, 2});
  EXPECT_EQ(portrait.resized_size, (std::array<std::uint32_t, 2>{6, 4}));
  EXPECT_EQ(portrait.crop_top, 2U);
  EXPECT_EQ(portrait.crop_left, 0U);
  spec.input_size = {2, 2};
  spec.resize_shorter_side = 2;
  const auto odd = resolve_image_preprocess_geometry(spec, {2, 5});
  EXPECT_EQ(odd.crop_left, 2U);
  spec.resize = PreprocessResize::stretch;
  spec.resize_shorter_side = 0;
  const auto stretch = resolve_image_preprocess_geometry(spec, {7, 11});
  EXPECT_EQ(stretch.resized_size, spec.input_size);
  EXPECT_EQ(stretch.crop_top, 0U);
  EXPECT_EQ(stretch.crop_left, 0U);
}

TEST(ModelsPreprocessing, GeometryRejectsOverflowZeroSourceAndOversizedCrop) {
  auto spec = InputSpec().image;
  EXPECT_THROW(resolve_image_preprocess_geometry(spec, {0, 3}),
               InvalidArgument);
  EXPECT_THROW(resolve_image_preprocess_geometry(spec, {3, 0}),
               InvalidArgument);
  EXPECT_THROW(resolve_image_preprocess_geometry(
                   spec, {1, std::numeric_limits<std::uint32_t>::max()}),
               InvalidArgument);
  spec.input_size = {3, 9};
  EXPECT_THROW(resolve_image_preprocess_geometry(spec, {2, 3}),
               InvalidArgument);
}

TEST(ModelsPreprocessing, MalformedRecipesAreRejectedThroughWeightBoundary) {
  const auto original = Record();
  const std::vector<std::function<void(WeightInputSpec&)>> mutations{
      [](auto& s) { s.source_revision.clear(); },
      [](auto& s) { s.state_mapping_version.clear(); },
      [](auto& s) { s.reference_uri += '\n'; },
      [](auto& s) { s.reference_revision = std::string(4097, 'a'); },
      [](auto& s) { s.image.input_size[1] = 0; },
      [](auto& s) { s.image.resize = static_cast<PreprocessResize>(255); },
      [](auto& s) { s.image.resize_shorter_side = 0; },
      [](auto& s) { s.image.resize = PreprocessResize::stretch; },
      [](auto& s) {
        s.image.interpolation = static_cast<PreprocessInterpolation>(255);
      },
      [](auto& s) { s.image.color = static_cast<PreprocessColor>(255); },
      [](auto& s) { s.image.mean = {0}; },
      [](auto& s) { s.image.std.pop_back(); },
      [](auto& s) {
        s.image.mean[0] = std::numeric_limits<double>::infinity();
      },
      [](auto& s) {
        s.image.std[0] = std::numeric_limits<double>::quiet_NaN();
      },
      [](auto& s) { s.image.std[0] = 0; },
      [](auto& s) { s.image.std[0] = -1; },
      [](auto& s) { s.categories->clear(); },
      [](auto& s) { (*s.categories)[0].id.clear(); },
      [](auto& s) { (*s.categories)[0].name += '\0'; },
      [](auto& s) { (*s.categories)[1].id = (*s.categories)[0].id; },
  };
  for (std::size_t i = 0; i < mutations.size(); ++i) {
    auto record = original;
    mutations[i](*record.input_spec);
    EXPECT_THROW(validate_weight_record(record), InvalidArgument) << i;
    EXPECT_THROW(weight_input_identity(record), InvalidArgument) << i;
  }
  auto gray = original;
  gray.input_spec->image.color = PreprocessColor::gray;
  gray.input_spec->image.mean = {0};
  gray.input_spec->image.std = {1};
  gray.input_spec->categories.reset();
  EXPECT_NO_THROW(validate_weight_record(gray));
}

TEST(ModelsPreprocessing, CategoryMetadataHasExplicitCountAndTextBounds) {
  auto spec = InputSpec();
  spec.categories->resize(100001);
  EXPECT_THROW(validate_weight_input_spec(spec), InvalidArgument);
  spec = InputSpec();
  spec.categories->clear();
  for (std::size_t i = 0; i < 4096; ++i)
    spec.categories->push_back({std::to_string(i), std::string(4096, 'x')});
  EXPECT_THROW(validate_weight_input_spec(spec), InvalidArgument);
}

#if defined(JYPPX_LIUDING_PREPROCESS_IMAGE_TESTS)
void CheckImageComposition(bool cuda) {
  namespace image = jyppx::liuding::data::image;
  torch::set_num_threads(1);
  const auto record = Record();
  const auto& spec = weight_input_spec(record).image;
  image::VisionSample sample;
  sample.width = 3;
  sample.height = 2;
  sample.image_path = "fixture://rgb";
  sample.pixels.dtype = torch::kFloat64;
  sample.pixels.range = image::PixelValueRange::unit_interval;
  sample.image =
      torch::tensor(std::vector<double>{0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
                                        0.25, 0.5, 0.75, 1, 0, 0.25},
                    torch::kFloat64)
          .reshape({3, 2, 3});
  sample.geometry.emplace(image::ImageExtent{3, 2});
  auto batch = image::collate_images(std::array{sample},
                                     std::array{image::ImageBatchKey{"input"}});
  if (cuda) {
    batch.images = batch.images.cuda();
    batch.valid = batch.valid.cuda();
    batch.labels = batch.labels.cuda();
  }
  const auto geometry =
      resolve_image_preprocess_geometry(spec, {sample.height, sample.width});
  const std::vector<std::vector<image::ImageGeometryStep>> steps{{
      {.kind = image::ImageGeometryKind::resize,
       .extent = {geometry.resized_size[1], geometry.resized_size[0]}},
      {.kind = image::ImageGeometryKind::crop,
       .extent = {geometry.output_size[1], geometry.output_size[0]},
       .left = geometry.crop_left,
       .top = geometry.crop_top},
  }};
  image::ImageBatchGeometryOptions options;
  options.joint.interpolation = image::ImageInterpolation::bicubic;
  const auto resized =
      image::transform_image_batch_geometry(batch, steps, options);
  image::ImageBatchTransformOptions numeric;
  numeric.image.swap_red_blue = spec.color == PreprocessColor::bgr;
  numeric.image.mean = spec.mean;
  numeric.image.std = spec.std;
  const auto output = image::transform_image_batch(resized, numeric);
  const auto expected =
      torch::tensor(
          std::vector<double>(preprocessing_reference::kExpected.begin(),
                              preprocessing_reference::kExpected.end()),
          torch::kFloat64)
          .reshape({1, 3, 3, 4});
  EXPECT_TRUE(torch::allclose(output.images.cpu().to(torch::kFloat64), expected,
                              2e-6, 2e-6));
  EXPECT_EQ(output.images.device().is_cuda(), cuda);
  EXPECT_EQ(output.pixels.color, image::PixelColorModel::bgr);
  EXPECT_EQ(output.pixels.range, image::PixelValueRange::unbounded);
  EXPECT_EQ(output.items[0].geometry->source_extent().width, 3U);
  EXPECT_EQ(output.items[0].geometry->output_extent().width, 4U);
  EXPECT_EQ(weight_input_identity(record),
            preprocessing_reference::kInputIdentity);
}

TEST(ModelsPreprocessing, CpuRecipeComposesExistingGeometryAndNormalization) {
  CheckImageComposition(false);
}
#if defined(JYPPX_LIUDING_PREPROCESS_CUDA_TESTS)
TEST(ModelsPreprocessing,
     CudaRecipeComposesOnCallerStreamWithoutPixelReadback) {
  if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA runtime unavailable";
  const auto stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard guard(stream);
  CheckImageComposition(true);
  EXPECT_EQ(c10::cuda::getCurrentCUDAStream(0), stream);
}
#endif
#endif
}  // namespace
