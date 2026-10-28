// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "reference_file.hpp"

#include <jyppx/liuding/data_image/batch_geometry.hpp>
#include <jyppx/liuding/data_image/batch_transform.hpp>
#include <jyppx/liuding/experiment/evaluation.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/models.hpp>
#include <jyppx/liuding/optim/sgd_adapter.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>
#include <jyppx/liuding/training/single_device_strategy.hpp>

#include <torch/torch.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

namespace {
using namespace jyppx::liuding;
using namespace jyppx::liuding::models;
namespace img = jyppx::liuding::data::image;

// Filled from the independently generated, pinned Python reference.
constexpr const char* kTaskDigest =
    "a2520fbfa24d67977f20870b6187c2bf14f882a21b468cecc5140e62b5d5f7ae";
constexpr std::uint64_t kTaskSize = 21234728;
constexpr const char* kDatasetDigest =
    "3368eeb4cfa5fbd63461b1529827a66151d9e5916a95b8b5d314723a2414a2a6";

Tensor tensor(const interop::SafeTensorsView& view, const std::string& name) {
  const auto& entries = view.directory().tensors;
  const auto entry =
      std::find_if(entries.begin(), entries.end(),
                   [&](const auto& e) { return e.name == name; });
  if (entry == entries.end())
    throw std::runtime_error("Missing classification reference tensor");
  const auto dtype =
      entry->dtype == interop::SafeTensorDtype::u8    ? torch::kUInt8
      : entry->dtype == interop::SafeTensorDtype::i64 ? torch::kInt64
                                                      : torch::kFloat32;
  std::vector<std::int64_t> shape(entry->shape.begin(), entry->shape.end());
  auto output = torch::empty(shape, dtype);
  const auto bytes = view.tensor_bytes(name);
  if (bytes.size() != output.nbytes())
    throw std::runtime_error("Reference payload mismatch");
  std::memcpy(output.data_ptr(), bytes.data(), bytes.size());
  return output;
}

WeightRecord record(bool dino) {
  const std::string revision = dino
                                   ? "ed25f3a31f01632728cabb09d1542f84ab7b0056"
                                   : "767268603ca0cb0bfe326fa87277f19c419566ef";
  const std::string repo =
      dino ? "facebook/dinov2-small" : "timm/resnet50.a1_in1k";
  return {
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
          "Fixed external pretrained bytes; balanced Fashion-MNIST subset "
          "linear-probe gate only."};
}

Tensor preprocess(const Tensor& raw, bool dino, const torch::Device& device) {
  auto pixels = raw.to(torch::kFloat32).div(255).expand({-1, 3, -1, -1});
  std::vector<img::VisionSample> samples;
  std::vector<img::ImageBatchKey> keys;
  for (std::int64_t i = 0; i < pixels.size(0); ++i) {
    img::VisionSample sample;
    sample.image = pixels[i];
    sample.width = 28;
    sample.height = 28;
    sample.image_path = "fixture://fashion-mnist";
    sample.pixels.dtype = torch::kFloat32;
    sample.pixels.range = img::PixelValueRange::unit_interval;
    sample.geometry.emplace(img::ImageExtent{28, 28});
    samples.push_back(std::move(sample));
    keys.push_back({std::to_string(i)});
  }
  auto batch = img::collate_images(samples, keys);
  if (device.is_cuda()) {
    batch.images = batch.images.to(device);
    batch.valid = batch.valid.to(device);
    batch.labels = batch.labels.to(device);
  }
  const auto shorter = dino ? 256U : 235U;
  const auto geometry_spec = resolve_image_preprocess_geometry(
      {.resize = PreprocessResize::shortest_edge_center_crop,
       .resize_shorter_side = shorter},
      {28, 28});
  const std::vector<img::ImageGeometryStep> steps{
      {.kind = img::ImageGeometryKind::resize, .extent = {shorter, shorter}},
      {.kind = img::ImageGeometryKind::crop,
       .extent = {224, 224},
       .left = geometry_spec.crop_left,
       .top = geometry_spec.crop_top}};
  const std::vector<std::vector<img::ImageGeometryStep>> all_steps(
      samples.size(), steps);
  img::ImageBatchGeometryOptions geometry;
  geometry.joint.interpolation = img::ImageInterpolation::bicubic;
  auto resized =
      img::transform_image_batch_geometry(batch, all_steps, geometry);
  img::ImageBatchTransformOptions numeric;
  numeric.image.mean = {.485, .456, .406};
  numeric.image.std = {.229, .224, .225};
  return img::transform_image_batch(resized, numeric).images;
}

Tensor normalize(const Tensor& value) {
  return value / value.norm(2, 1, true).clamp_min(1e-12);
}

std::string state_digest(nn::ModelAdapter& model) {
  Sha256 digest;
  for (const auto& entries :
       {model.state().named_parameters(), model.state().named_buffers()})
    for (const auto& entry : entries) {
      digest.update(std::as_bytes(std::span(entry.key())));
      auto value = entry.value().detach().cpu().contiguous();
      digest.update({static_cast<const std::byte*>(value.const_data_ptr()),
                     value.nbytes()});
    }
  return sha256_digest_hex(digest.finish());
}

void compare(const Tensor& actual, const Tensor& expected,
             const std::string& name, double tolerance = 5e-4) {
  ASSERT_EQ(actual.sizes(), expected.sizes());
  auto cpu = actual.detach().cpu();
  ASSERT_TRUE(torch::isfinite(cpu).all().item<bool>());
  const auto error = (cpu - expected).abs().max().item<double>();
  EXPECT_TRUE(torch::allclose(cpu, expected, tolerance, tolerance))
      << name << " max_error=" << error;
}

void task(bool dino, bool cuda) {
  const auto* directory =
      std::getenv("JYPPX_LIUDING_CLASSIFICATION_REFERENCE_DIR");
  const auto* weights_directory =
      std::getenv("JYPPX_LIUDING_PRETRAINED_REFERENCE_DIR");
  if (!directory || !weights_directory)
    GTEST_SKIP() << "Explicit pinned external task data and weights required";
  if (cuda && !torch::cuda::is_available()) GTEST_SKIP() << "CUDA unavailable";
  torch::set_num_threads(2);
  const bool old_conv = at::globalContext().allowTF32CuDNN(),
             old_mm = at::globalContext().allowTF32CuBLAS();
  struct Restore {
    bool conv, mm;
    ~Restore() {
      at::globalContext().setAllowTF32CuDNN(conv);
      at::globalContext().setAllowTF32CuBLAS(mm);
    }
  } restore{old_conv, old_mm};
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  const test::MappedReference encoded(
      std::filesystem::path(directory) / "classification-reference.safetensors",
      kTaskSize);
  ASSERT_EQ(sha256_hex(encoded), kTaskDigest);
  const interop::SafeTensorsView view(encoded);
  const auto weight = record(dino);
  const test::MappedReference weights(
      std::filesystem::path(weights_directory) / weight.artifact->logical_name,
      weight.artifact->size_bytes);
  const auto device = torch::Device(cuda ? "cuda:0" : "cpu");
  ModelOptions options;
  options.dinov2.position_antialias = false;
  auto backbone = create_model(weight.model_name, options);
  ASSERT_TRUE(load_weights(backbone, weight, weights).applied);
  reset_classifier(backbone, 0);
  backbone.state().to(device);
  backbone.state().eval();
  for (auto p : backbone.state().parameters()) p.set_requires_grad(false);
  const auto before = state_digest(backbone);
  const std::string name = dino ? "dinov2" : "resnet50";
  auto train_images = tensor(view, "train_images"),
       test_images = tensor(view, "test_images");
  auto train_labels = tensor(view, "train_labels").to(device),
       test_labels = tensor(view, "test_labels").to(device);
  auto expected_train = tensor(view, name + "_train_features"),
       expected_test = tensor(view, name + "_test_features");
  std::vector<Tensor> cache;
  {
    torch::NoGradGuard guard;
    for (std::int64_t i = 0; i < 1000; i += 16) {
      auto x = preprocess(
          train_images.slice(0, i, std::min(i + 16, std::int64_t{1000})), dino,
          device);
      auto feature =
          normalize(backbone.invoke({.positional = {x}}).value.toTensor());
      compare(feature, expected_train.slice(0, i, i + feature.size(0)),
              name + " train feature");
      cache.push_back(feature);
    }
  }
  auto features = torch::cat(cache);
  cache.clear();
  auto head = std::make_shared<torch::nn::LinearImpl>(features.size(1), 10);
  head->to(device);
  {
    torch::NoGradGuard guard;
    head->weight.zero_();
    head->bias.zero_();
  }
  const auto expected_logits = tensor(view, name + "_test_logits");
  std::int64_t prediction_offset = 0;
  auto model = nn::adapt_module(
      head,
      [&](torch::nn::LinearImpl& layer,
          const nn::Invocation& call) -> c10::IValue {
        auto x = call.positional.front().toTensor();
        if (x.dim() == 4) {
          x = normalize(backbone.invoke({.positional = {x}}).value.toTensor());
          const auto end = prediction_offset + x.size(0);
          compare(x, expected_test.slice(0, prediction_offset, end),
                  name + " test feature");
          auto logits = layer.forward(x);
          compare(logits, expected_logits.slice(0, prediction_offset, end),
                  name + " test logits", 3e-3);
          prediction_offset = end;
          return logits;
        }
        return layer.forward(x);
      },
      {.positional = {ValueSchema::tensor()}}, ValueSchema::tensor());
  std::vector<optim::OptimizerParameterBinding> bindings;
  for (const auto& p : model.state().named_parameters())
    bindings.push_back({.parameter_id = p.key(), .parameter = p.value()});
  optim::SgdOptimizerAdapter optimizer("optimizer.probe",
                                       {{.group_id = "head",
                                         .parameters = std::move(bindings),
                                         .options = {.learning_rate = 10.}}});
  training::SingleDeviceStrategy strategy({.device = device});
  const std::array<optim::OptimizerAdapter*, 1> optimizers{&optimizer};
  strategy.prepare(model, optimizers);
  double first = 0, last = 0;
  for (int epoch = 0; epoch < 40; ++epoch) {
    strategy.zero_grad(optimizer);
    auto loss = torch::nn::functional::cross_entropy(head->forward(features),
                                                     train_labels);
    last = loss.item<double>();
    if (epoch == 0) first = last;
    strategy.backward(loss);
    strategy.optimizer_step(optimizer);
  }
  EXPECT_EQ(strategy.completed_optimizer_steps("optimizer.probe"), 40);
  EXPECT_LT(last, first);
  compare(head->weight, tensor(view, name + "_head_weight"),
          name + " trained head", 3e-3);
  compare(head->bias, tensor(view, name + "_head_bias"), name + " trained bias",
          3e-3);
  std::int64_t offset = 0;
  const auto evaluation = training::evaluate_classification(
      model,
      [&](std::stop_token)
          -> std::optional<training::ClassificationEvaluationBatch> {
        if (offset == 1000) return std::nullopt;
        const auto end = std::min(offset + 16, std::int64_t{1000});
        auto x = preprocess(test_images.slice(0, offset, end), dino, device);
        training::ClassificationEvaluationBatch batch{
            .invocation = {.positional = {x}},
            .targets = test_labels.slice(0, offset, end)};
        offset = end;
        return batch;
      },
      {.num_classes = 10,
       .max_batches = 63,
       .max_samples = 1000,
       .max_batch_samples = 16});
  EXPECT_EQ(evaluation.overall.samples, 1000);
  EXPECT_EQ(prediction_offset, 1000);
  ASSERT_TRUE(evaluation.overall.accuracy);
  ASSERT_TRUE(evaluation.overall.macro_f1);
  EXPECT_GE(*evaluation.overall.accuracy, .65);
  EXPECT_GE(*evaluation.overall.macro_f1, .60);
  EXPECT_EQ(state_digest(backbone), before);
  EXPECT_FALSE(backbone.state().is_training());
  for (auto p : backbone.state().parameters()) {
    EXPECT_FALSE(p.requires_grad());
    EXPECT_FALSE(p.grad().defined());
  }
  experiment::EvaluationSuite suite{
      .suite_id = "fashion-mnist.linear-probe",
      .task = "classification",
      .dataset_digest = {.hex = kDatasetDigest},
      .environment_id = std::string(cuda ? "cuda" : "cpu") + ".fp32.tf32-off"};
  for (const auto& point : evaluation.observations) {
    const auto metric = point.definition.metric_id;
    suite.rules.push_back({.rule_id = metric,
                           .definition = point.definition,
                           .context = point.context,
                           .minimum_samples = 1000,
                           .minimum = metric == "classification.accuracy" ? .65
                                      : metric == "classification.macro_f1"
                                          ? .60
                                          : 0.});
  }
  Sha256 model_digest;
  model_digest.update(std::as_bytes(std::span(before)));
  model_digest.update(std::as_bytes(std::span(weight.artifact->digest.hex)));
  for (const auto& p : head->parameters()) {
    auto cpu = p.detach().cpu().contiguous();
    model_digest.update(
        {static_cast<const std::byte*>(cpu.const_data_ptr()), cpu.nbytes()});
  }
  const std::string config =
      "fashion-probe/"
      "v1;classes=10;normalize=l2;steps=40;lr=10;batch=1000;init=zero;rgb="
      "repeat-gray;tf32=off;shorter=" +
      std::to_string(dino ? 256 : 235) +
      ";crop=224,bicubic-aa,nearest-even;model=" + name +
      ";weights=" + weight.artifact->digest.hex;
  experiment::EvaluationIdentity id{
      .evaluation_id = name + ".fashion",
      .run_id = name + ".probe",
      .model_digest = {.hex = sha256_digest_hex(model_digest.finish())},
      .build_fingerprint = build_fingerprint_json(),
      .adapter_id = "training.classification.v1",
      .config_digest = {.hex = sha256_hex(std::as_bytes(std::span(config)))}};
  const auto report = experiment::run_evaluation(
      suite, id,
      [&](const auto&, auto, auto) { return evaluation.observations; });
  EXPECT_EQ(report.verdict, experiment::QualityVerdict::pass);
  std::cout << "TASK " << name << " " << device
            << " accuracy=" << *evaluation.overall.accuracy
            << " macro_f1=" << *evaluation.overall.macro_f1
            << " loss_first=" << first << " loss_last=" << last
            << " backbone_unchanged=true steps=40\n";
  std::cout << "CONFIG " << config << '\n';
  std::cout << "REPORT " << experiment::quality_report_json(report) << '\n';
}
TEST(ModelsClassification, ResNet50FashionCPU) { task(false, false); }
TEST(ModelsClassification, ResNet50FashionCUDA) { task(false, true); }
TEST(ModelsClassification, DinoV2SmallFashionCPU) { task(true, false); }
TEST(ModelsClassification, DinoV2SmallFashionCUDA) { task(true, true); }
}  // namespace
