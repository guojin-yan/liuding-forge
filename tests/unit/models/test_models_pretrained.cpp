// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "reference_file.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/models.hpp>

#include <torch/optim.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#if defined(JYPPX_LIUDING_PRETRAINED_IMAGE_TESTS)
#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#endif

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;

constexpr const char* kReferenceDigest =
    "4736a2ef8f98bfd773700c6d06fe2614a0e5fb8d6d8c02c36b9f0c643d6250a6";

class IeeePrecision final {
 public:
  IeeePrecision()
      : convolution_(at::globalContext().allowTF32CuDNN()),
        matmul_(at::globalContext().allowTF32CuBLAS()) {
    at::globalContext().setAllowTF32CuDNN(false);
    at::globalContext().setAllowTF32CuBLAS(false);
  }
  ~IeeePrecision() {
    at::globalContext().setAllowTF32CuDNN(convolution_);
    at::globalContext().setAllowTF32CuBLAS(matmul_);
  }

 private:
  bool convolution_, matmul_;
};

Tensor ReferenceTensor(std::span<const std::byte> encoded,
                       const std::string& name) {
  const interop::SafeTensorsView view(encoded);
  const auto& entries = view.directory().tensors;
  const auto entry =
      std::find_if(entries.begin(), entries.end(),
                   [&](const auto& t) { return t.name == name; });
  if (entry == entries.end() || entry->dtype != interop::SafeTensorDtype::f32)
    throw std::runtime_error("missing pretrained F32 reference");
  const auto bytes = view.tensor_bytes(name);
  std::vector<std::int64_t> shape(entry->shape.begin(), entry->shape.end());
  auto result = torch::empty(shape, torch::kFloat32);
  std::memcpy(result.data_ptr(), bytes.data(), bytes.size());
  return result;
}

WeightRecord ExternalRecord(bool dino) {
  const std::string revision = dino
                                   ? "ed25f3a31f01632728cabb09d1542f84ab7b0056"
                                   : "767268603ca0cb0bfe326fa87277f19c419566ef";
  const std::string repo =
      dino ? "facebook/dinov2-small" : "timm/resnet50.a1_in1k";
  WeightRecord record{
      .id =
          dino ? "facebook.dinov2-small.unmasked.v1" : "timm.resnet50.a1_in1k",
      .model_name = dino ? "vit_small_patch14_dinov2" : "resnet50",
      .source_uri = "https://huggingface.co/" + repo + "/resolve/" + revision +
                    "/model.safetensors",
      .license_spdx = "Apache-2.0",
      .license_uri =
          "https://huggingface.co/" + repo + "/blob/" + revision + "/README.md",
      .availability = WeightAvailability::pinned_external,
      .artifact =
          ArtifactRef{
              .digest = {.hex = dino ? "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50"
                                       "ca68b29db8a68c8c84c9810e5"
                                     : "773525d5821de224f8f30c33377b7a795d7863e"
                                       "08522698200d3217d3f2a41bb"},
              .size_bytes = dino ? 88240512U : 102469840U,
              .logical_name = dino ? "dinov2-small.liuding.safetensors"
                                   : "resnet50.a1_in1k.safetensors",
              .media_type = "application/x-safetensors",
              .producer = dino ? "liuding-dinov2-hf-small-unmasked-v1"
                               : "timm/resnet50.a1_in1k"},
      .status_reason =
          "Explicit external pretrained bytes; synthetic execution gate only, "
          "no real-dataset accuracy or complete fine-tuning qualification."};
  record.input_spec = WeightInputSpec{
      .source_revision = revision,
      .state_mapping_version = dino
                                   ? "jyppx.liuding.dinov2-hf-small.unmasked.v1"
                                   : "timm.resnet50.a1_in1k.identity.v1",
      .reference_uri = "https://huggingface.co/" + repo + "/resolve/" +
                       revision +
                       (dino ? "/preprocessor_config.json" : "/config.json"),
      .reference_revision = revision,
      .image = {.input_size = {224, 224},
                .resize = PreprocessResize::shortest_edge_center_crop,
                .resize_shorter_side = dino ? 256U : 235U,
                .interpolation = PreprocessInterpolation::bicubic,
                .color = PreprocessColor::rgb,
                .mean = {0.485, 0.456, 0.406},
                .std = {0.229, 0.224, 0.225}},
      .categories = std::nullopt};
  return record;
}

void Compare(const Tensor& actual, const Tensor& expected,
             const std::string& label, double tolerance = 3e-4) {
  ASSERT_EQ(actual.sizes(), expected.sizes()) << label;
  const auto cpu = actual.detach().cpu();
  ASSERT_TRUE(torch::isfinite(cpu).all().item<bool>()) << label;
  const auto error = (cpu - expected).abs().max().item<double>();
  std::cout << label << " max_abs_error=" << error << '\n';
  EXPECT_TRUE(torch::allclose(cpu, expected, tolerance, tolerance)) << label;
}

std::string BackboneDigest(nn::ModelAdapter& model, const std::string& head) {
  Sha256 digest;
  const auto consume = [&](const auto& entries) {
    for (const auto& entry : entries) {
      if (entry.key().starts_with(head + '.')) continue;
      const auto value = entry.value().detach().cpu().contiguous();
      digest.update(
          std::as_bytes(std::span(entry.key().data(), entry.key().size())));
      digest.update({static_cast<const std::byte*>(value.const_data_ptr()),
                     value.nbytes()});
    }
  };
  consume(model.state().named_parameters());
  consume(model.state().named_buffers());
  return sha256_digest_hex(digest.finish());
}

void FrozenHeadUpdate(nn::ModelAdapter& model, const Tensor& input) {
  const auto head = classifier_info(model).parameter_prefix;
  const auto before_backbone = BackboneDigest(model, head);
  for (const auto& p : model.state().parameters()) p.set_requires_grad(false);
  reset_classifier(model, 3);
  std::vector<Tensor> parameters;
  for (const auto& entry : model.state().named_parameters())
    if (entry.key().starts_with(head + '.'))
      parameters.push_back(entry.value());
  ASSERT_EQ(parameters.size(), 2U);
  const auto before_head = parameters.back().detach().clone();
  torch::optim::SGD optimizer(parameters, torch::optim::SGDOptions(0.001));
  const auto images = torch::cat({input, input.flip({3})}, 0);
  const auto labels = torch::tensor({0, 1}, torch::kInt64).to(input.device());
  const auto output = model.invoke({.positional = {images}}).value.toTensor();
  const auto loss = torch::nn::functional::cross_entropy(output, labels);
  ASSERT_TRUE(torch::isfinite(loss).item<bool>());
  loss.backward();
  for (const auto& p : parameters) {
    ASSERT_TRUE(p.grad().defined());
    EXPECT_TRUE(torch::isfinite(p.grad()).all().item<bool>());
    EXPECT_GT(p.grad().abs().sum().item<double>(), 0);
  }
  optimizer.step();
  EXPECT_FALSE(torch::equal(before_head, parameters.back()));
  EXPECT_EQ(BackboneDigest(model, head), before_backbone);
  for (const auto& entry : model.state().named_parameters())
    if (!entry.key().starts_with(head + '.'))
      EXPECT_FALSE(entry.value().grad().defined());
  std::cout << "frozen-head-update finite_loss=" << loss.item<double>()
            << " backbone_sha256=" << before_backbone << '\n';
}

#if defined(JYPPX_LIUDING_PRETRAINED_IMAGE_TESTS)
Tensor DirectDevicePreprocess(const Tensor& image, bool dino) {
  // Independently compose torchvision's declared tensor recipe on this backend.
  // CUDA coordinate arithmetic need not be bitwise equal to the CPU oracle.
  const std::int64_t height = dino ? 256 : 235;
  const std::int64_t width = image.size(2) * height / image.size(1);
  const auto top =
      static_cast<std::int64_t>(std::nearbyint((height - 224) / 2.0));
  const auto left =
      static_cast<std::int64_t>(std::nearbyint((width - 224) / 2.0));
  auto value = torch::nn::functional::interpolate(
      image.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions()
                              .size(std::vector<std::int64_t>{height, width})
                              .mode(torch::kBicubic)
                              .align_corners(false)
                              .antialias(true));
  value = value.slice(2, top, top + 224).slice(3, left, left + 224);
  const auto mean = torch::tensor({0.485F, 0.456F, 0.406F}, image.options())
                        .view({1, 3, 1, 1});
  const auto std = torch::tensor({0.229F, 0.224F, 0.225F}, image.options())
                       .view({1, 3, 1, 1});
  return (value - mean) / std;
}

Tensor Preprocess(const Tensor& image, const WeightRecord& record, bool cuda) {
  namespace img = jyppx::liuding::data::image;
  const auto& spec = weight_input_spec(record).image;
  img::VisionSample sample;
  sample.width = image.size(2);
  sample.height = image.size(1);
  sample.image_path = "fixture://deterministic-rgb";
  sample.pixels.dtype = torch::kFloat32;
  sample.pixels.range = img::PixelValueRange::unit_interval;
  sample.image = image;
  sample.geometry.emplace(img::ImageExtent{sample.width, sample.height});
  auto batch = img::collate_images(std::array{sample},
                                   std::array{img::ImageBatchKey{"input"}});
  if (cuda) {
    batch.images = batch.images.cuda();
    batch.valid = batch.valid.cuda();
    batch.labels = batch.labels.cuda();
  }
  const auto g =
      resolve_image_preprocess_geometry(spec, {sample.height, sample.width});
  const std::vector<std::vector<img::ImageGeometryStep>> steps{{
      {.kind = img::ImageGeometryKind::resize,
       .extent = {g.resized_size[1], g.resized_size[0]}},
      {.kind = img::ImageGeometryKind::crop,
       .extent = {g.output_size[1], g.output_size[0]},
       .left = g.crop_left,
       .top = g.crop_top},
  }};
  img::ImageBatchGeometryOptions geometry;
  geometry.joint.interpolation = img::ImageInterpolation::bicubic;
  auto resized = img::transform_image_batch_geometry(batch, steps, geometry);
  img::ImageBatchTransformOptions numeric;
  numeric.image.mean = spec.mean;
  numeric.image.std = spec.std;
  auto result = img::transform_image_batch(resized, numeric);
  EXPECT_EQ(result.items[0].geometry->source_extent().width, sample.width);
  EXPECT_EQ(result.items[0].geometry->output_extent().height, 224U);
  return result.images;
}
#endif

void CheckPretrained(bool dino, bool cuda) {
  const auto* directory = std::getenv("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR");
  if (directory == nullptr || *directory == '\0')
    GTEST_SKIP() << "Explicit pinned external weights/reference directory "
                    "required; no network acquisition";
  if (cuda && !torch::cuda::is_available())
    GTEST_SKIP() << "CUDA runtime unavailable";
  torch::set_num_threads(2);
  IeeePrecision precision;
  const auto root = std::filesystem::path(directory);
  const test::MappedReference reference(
      root / "reference/pretrained-reference.safetensors", 9884068U);
  ASSERT_EQ(sha256_hex(reference), kReferenceDigest);
  const auto record = ExternalRecord(dino);
  const test::MappedReference weights(root / record.artifact->logical_name,
                                      record.artifact->size_bytes);
  ModelOptions options;
  options.dinov2.position_antialias = false;
  auto model = create_model(record.model_name, options);
  const auto device = torch::Device(cuda ? torch::kCUDA : torch::kCPU);
  model.state().to(device);
  const auto imported = load_weights(model, record, weights);
  ASSERT_TRUE(imported.applied);
  EXPECT_EQ(imported.applied_tensors, dino ? 174U : 320U);
  EXPECT_EQ(imported.plan.tensor_coverage, 1.0);
  EXPECT_EQ(imported.plan.element_coverage, 1.0);
  EXPECT_TRUE(imported.plan.missing.empty());
  EXPECT_TRUE(imported.plan.unexpected.empty());
  model.state().eval();
  const std::string prefix = dino ? "dinov2" : "resnet50";
  const auto input = ReferenceTensor(reference, prefix + "_input").to(device);
  {
    torch::NoGradGuard guard;
#if defined(JYPPX_LIUDING_PRETRAINED_IMAGE_TESTS)
    const auto prepared =
        Preprocess(ReferenceTensor(reference, "image"), record, cuda);
    const auto direct = DirectDevicePreprocess(
        ReferenceTensor(reference, "image").to(device), dino);
    Compare(prepared, direct.cpu(), prefix + " tensor-preprocess-device", 2e-5);
    if (!cuda)
      Compare(prepared, input.cpu(), prefix + " tensor-preprocess-cpu", 2e-5);
    std::cout << prefix << " cpu-cuda-preprocess-drift="
              << (prepared.cpu() - input.cpu()).abs().max().item<double>()
              << '\n';
    Compare(model.invoke({.positional = {prepared}}).value.toTensor(),
            ReferenceTensor(reference, prefix + "_output"),
            prefix + " pipeline-output");
#endif
    Compare(model.invoke({.positional = {input}}).value.toTensor(),
            ReferenceTensor(reference, prefix + "_output"), prefix + " output");
    if (dino) {
      const auto concrete =
          std::dynamic_pointer_cast<DinoV2Impl>(model.state().shared_module());
      ASSERT_NE(concrete, nullptr);
      ASSERT_FALSE(concrete->options().position_antialias);
      for (const std::string tag :
           {"dinov2", "dinov2_dynamic", "dinov2_native"}) {
        const auto value =
            ReferenceTensor(reference, tag + "_input").to(device);
        Compare(concrete->forward_features(value),
                ReferenceTensor(reference, tag + "_tokens"), tag + " tokens");
      }
    }
  }
  FrozenHeadUpdate(model, input);
  ::testing::Test::RecordProperty("artifact_sha256",
                                  record.artifact->digest.hex);
  ::testing::Test::RecordProperty("weight_input_identity",
                                  weight_input_identity(record));
  ::testing::Test::RecordProperty("task_qualification",
                                  "none-synthetic-input-only");
}

TEST(ModelsPretrained, Resnet50CpuStrictImportReferenceAndFrozenHeadUpdate) {
  CheckPretrained(false, false);
}
TEST(ModelsPretrained, DinoV2SmallCpuStrictImportReferenceAndFrozenHeadUpdate) {
  CheckPretrained(true, false);
}
#if defined(JYPPX_LIUDING_PRETRAINED_CUDA_TESTS)
TEST(ModelsPretrained, Resnet50CudaStrictImportReferenceAndFrozenHeadUpdate) {
  CheckPretrained(false, true);
}
TEST(ModelsPretrained,
     DinoV2SmallCudaStrictImportReferenceAndFrozenHeadUpdate) {
  CheckPretrained(true, true);
}
#endif
}  // namespace
