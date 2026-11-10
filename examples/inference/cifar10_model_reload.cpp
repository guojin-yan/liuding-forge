// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file cifar10_model_reload.cpp
 * @brief Reloads a recipe's complete ResNet18 state and evaluates CIFAR-10. */

#include <jyppx/liuding/data/samplers/random_sampler.hpp>
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/foundation/random.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/version.hpp>
#include <jyppx/liuding/interop/model_weight_import.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/vision.hpp>
#include <jyppx/liuding/training/classification_evaluation.hpp>

#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace image = forge::data::image;
namespace fs = std::filesystem;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::size_t kSamples = 10000;
constexpr std::size_t kBatchSize = 128;
constexpr std::size_t kBatches = 79;
constexpr std::int64_t kSeed = 20261028;
constexpr std::size_t kStateTensors = 122;
constexpr std::array<std::string_view, 10> kClasses{
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"};

void Require(bool accepted, std::string_view message) {
  if (!accepted) throw std::runtime_error(std::string(message));
}

std::vector<std::byte> ReadModel(const fs::path& path) {
  Require(fs::is_regular_file(fs::symlink_status(path)),
          "Expected a regular model.safetensors file");
  const auto size = fs::file_size(path);
  Require(size > 0 && size <= 256 * kMiB,
          "Model file must be within the explicit 256 MiB bound");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(size));
  Require(input.good() && input.peek() == std::char_traits<char>::eof(),
          "Cannot read the complete model file");
  return bytes;
}

forge::interop::ModelWeightImportResult LoadModel(
    forge::nn::ModelAdapter& model, const fs::path& path,
    std::string_view expected_sha) {
  const auto original = ReadModel(path);
  Require(forge::sha256_hex(original) == expected_sha,
          "Saved model SHA-256 differs from the recipe result");
  forge::interop::SafeTensorsLimits limits;
  limits.max_file_bytes = limits.max_payload_bytes = limits.max_tensor_bytes =
      256 * kMiB;
  limits.max_tensors = kStateTensors;
  const forge::interop::SafeTensorsView view(original, limits);
  Require(view.directory().tensors.size() == kStateTensors,
          "Expected the complete ResNet18 parameters and buffers");
  std::vector<forge::interop::SafeTensorSource> mapped;
  for (const auto& entry : view.directory().tensors) {
    std::string_view prefix;
    if (entry.name.starts_with("parameter.")) {
      prefix = "parameter.";
    } else if (entry.name.starts_with("buffer.")) {
      prefix = "buffer.";
    } else {
      throw std::runtime_error("Unexpected recipe state key");
    }
    // Preserve every payload, shape and dtype; only remove the recipe prefix.
    mapped.push_back({.name = entry.name.substr(prefix.size()),
                      .dtype = entry.dtype,
                      .shape = entry.shape,
                      .data = view.tensor_bytes(entry.name)});
  }
  const auto encoded = forge::interop::serialize_safetensors(
      mapped, {{"source_model_sha256", std::string(expected_sha)}}, limits);
  forge::interop::ModelWeightImportOptions options;
  options.mapping.mode = forge::interop::WeightImportMode::strict;
  options.tensor_limits = limits;
  options.max_staging_bytes = 768 * kMiB;
  const auto imported = forge::interop::import_model_weights(
      model.state(), encoded,
      forge::ArtifactDigest{.hex = forge::sha256_hex(encoded)}, options);
  Require(imported.applied && imported.plan.ready &&
              imported.applied_tensors == kStateTensors &&
              imported.plan.source_tensor_count == kStateTensors &&
              imported.plan.target_tensor_count == kStateTensors &&
              imported.plan.conversions.empty() &&
              imported.plan.retained_targets.empty(),
          "Strict complete-model import rejected");
  // Encoded input and import staging are released before decoding images.
  return imported;
}

std::string StateDigest(forge::nn::ModelAdapter& model) {
  std::vector<std::pair<std::string, torch::Tensor>> tensors;
  for (const auto& entry : model.state().named_parameters())
    tensors.emplace_back("parameter." + entry.key(), entry.value());
  for (const auto& entry : model.state().named_buffers())
    tensors.emplace_back("buffer." + entry.key(), entry.value());
  Require(tensors.size() == kStateTensors,
          "Complete model state topology changed");
  std::sort(tensors.begin(), tensors.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });
  forge::Sha256 digest;
  for (const auto& [name, tensor] : tensors) {
    Require(!tensor.grad().defined(),
            "Evaluation unexpectedly created a model gradient");
    const auto cpu = tensor.detach().to(torch::kCPU).contiguous();
    auto description =
        name + ":" + std::to_string(static_cast<int>(cpu.scalar_type())) + ":";
    for (const auto dimension : cpu.sizes())
      description += std::to_string(dimension) + ",";
    description += "\n";
    digest.update(
        std::as_bytes(std::span(description.data(), description.size())));
    digest.update(
        {static_cast<const std::byte*>(cpu.const_data_ptr()), cpu.nbytes()});
  }
  return forge::sha256_digest_hex(digest.finish());
}

// A single batch owns its image leases until the evaluator finishes forward.
class Images final {
 public:
  explicit Images(const fs::path& root) {
    auto index = image::index_image_folder(root);
    Require(index.records().size() == kSamples &&
                index.categories().size() == kClasses.size(),
            "Expected the complete ten-class 10000-image validation split");
    for (std::size_t i = 0; i < kClasses.size(); ++i)
      Require(index.categories()[i].name == kClasses[i],
              "Class order differs from the CIFAR-10 recipe");
    std::array<std::size_t, 10> counts{};
    for (const auto& record : index.records()) {
      Require(
          record.labels.size() == 1 &&
              static_cast<std::uint64_t>(record.labels[0]) < kClasses.size(),
          "Expected one valid class label for every image");
      ++counts[static_cast<std::size_t>(record.labels[0])];
    }
    Require(std::all_of(counts.begin(), counts.end(),
                        [](auto count) { return count == 1000; }),
            "Expected exactly 1000 validation images per class");
    image::ImageResourceBudgetLimits limits;
    limits.max_encoded_bytes = limits.max_decoded_bytes =
        limits.max_codec_scratch_bytes = limits.max_pinned_bytes =
            limits.max_gpu_bytes = limits.max_pipeline_inflight_bytes =
                256 * kMiB;
    limits.max_pipeline_inflight_batches = 1;
    auto budget = std::make_shared<image::ImageResourceBudget>(limits);
    image::ImageDatasetOptions options;
    options.resource_budget = budget;
    options.image.channels = image::ImageChannels::rgb;
    options.image.dtype = image::ImageDtype::float32_normalized;
    options.image.limits.max_input_bytes = kMiB;
    options.image.limits.max_output_bytes = 64 * kMiB;
    dataset_ = std::make_shared<image::ImageDataset>(root, index, options);
    geometry_.interpolation = image::ImageInterpolation::bilinear;
    geometry_.resource_budget = budget;
    collate_.resource_budget = budget;
    collate_.limits.max_samples = kBatchSize;
    collate_.limits.max_output_bytes = 64 * kMiB;
    collate_.limits.max_working_bytes = 64 * kMiB;
    collate_.limits.max_metadata_bytes = kMiB;
  }

  image::ImageBatch batch(std::span<const std::size_t> ids,
                          std::stop_token stop) const {
    Require(!ids.empty() && ids.size() <= kBatchSize, "Batch outside 1..128");
    std::vector<image::VisionSample> samples;
    std::vector<image::ImageBatchKey> keys;
    const image::ImageGeometryStep resize{
        .kind = image::ImageGeometryKind::resize, .extent = {32, 32}};
    for (const auto id : ids) {
      auto sample = dataset_->get(id, stop);
      keys.push_back({sample.image_path, 0, 0});
      samples.push_back(image::transform_sample(sample, std::span(&resize, 1),
                                                geometry_, stop));
    }
    return image::collate_images(samples, keys, collate_, stop);
  }

 private:
  std::shared_ptr<image::ImageDataset> dataset_;
  image::JointTransformOptions geometry_;
  image::ImageCollateOptions collate_;
};

int Run(const fs::path& model_path, std::string_view expected_sha,
        const fs::path& validation_root) {
  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  at::globalContext().setDeterministicAlgorithms(true, false);
  at::globalContext().setDeterministicCuDNN(true);
  at::globalContext().setBenchmarkCuDNN(false);
  at::globalContext().setAllowTF32CuDNN(false);
  at::globalContext().setAllowTF32CuBLAS(false);
  at::globalContext().setFloat32MatmulPrecision("highest");
  forge::core::RngGuard seeded(kSeed, true);
  forge::models::ResNetOptions architecture;
  architecture.num_classes = 10;
  architecture.input_channels = 3;
  auto model = forge::models::adapt(forge::models::resnet18(architecture));
  const auto imported = LoadModel(model, model_path, expected_sha);
  model.state().eval();
  const auto before = StateDigest(model);
  Images images(validation_root);
  forge::data::samplers::RandomSampler sampler(kSamples, kSeed);
  sampler.set_epoch(0);
  const auto order = sampler.indices();
  std::size_t offset = 0;
  std::size_t last_batch_samples = 0;
  std::optional<image::ImageBatch> current;
  const auto scores = forge::training::evaluate_classification(
      model,
      [&](std::stop_token stop)
          -> std::optional<forge::training::ClassificationEvaluationBatch> {
        current.reset();
        if (offset == kSamples) return std::nullopt;
        const auto count = std::min(kBatchSize, kSamples - offset);
        current = images.batch(std::span(order).subspan(offset, count), stop);
        offset += count;
        last_batch_samples = count;
        return forge::training::ClassificationEvaluationBatch{
            .invocation = {.positional = {current->images}, .keyword = {}},
            .targets = current->labels,
            .source_indices = {}};
      },
      {.num_classes = 10,
       .max_batches = kBatches,
       .max_samples = kSamples,
       .max_batch_samples = kBatchSize});
  current.reset();
  Require(offset == kSamples && scores.batches == kBatches &&
              scores.rows_seen == kSamples &&
              scores.overall.samples == kSamples &&
              scores.overall.accuracy.has_value() && last_batch_samples == 16,
          "Evaluation did not consume exactly 10000 images");
  const auto after = StateDigest(model);
  Require(after == before, "Evaluation changed model state");
  const auto accuracy = *scores.overall.accuracy;
  Require(std::isfinite(accuracy), "Nonfinite accuracy");
  std::cout << std::setprecision(17)
            << "{\"schema_id\":\"jyppx.examples.cifar10-model-reload\","
               "\"schema_version\":1,\"device\":\"cpu\",\"model_sha256\":\""
            << expected_sha << "\",\"build\":"
            << forge::build_fingerprint_json(forge::current_build_fingerprint())
            << ",\"applied_tensors\":" << imported.applied_tensors
            << ",\"import_staging_bytes\":" << imported.charged_staging_bytes
            << ",\"parameter_tensors\":"
            << model.state().named_parameters().size()
            << ",\"buffer_tensors\":" << model.state().named_buffers().size()
            << ",\"state_sha256_before\":\"" << before
            << "\",\"state_sha256_after\":\"" << after
            << "\",\"model_state_unchanged\":true,\"training_updates\":0,"
               "\"forward_calls\":"
            << scores.batches
            << ",\"validation_samples\":" << scores.overall.samples
            << ",\"tail_batch_samples\":" << last_batch_samples
            << ",\"accuracy\":" << accuracy
            << ",\"minimum_accuracy\":0.65,\"quality_passed\":"
            << (accuracy >= .65 ? "true" : "false") << ",\"confusion\":[";
  for (std::size_t row = 0; row < scores.overall.confusion.size(); ++row) {
    if (row != 0) std::cout << ',';
    std::cout << '[';
    for (std::size_t column = 0; column < scores.overall.confusion[row].size();
         ++column) {
      if (column != 0) std::cout << ',';
      std::cout << scores.overall.confusion[row][column];
    }
    std::cout << ']';
  }
  std::cout << "]}\n";
  return accuracy >= .65 ? 0 : 4;
}

template <class Character>
int Main(int argc, Character* argv[]) {
  constexpr auto usage =
      "usage: liuding-forge-example-cifar10-model-reload "
      "MODEL_SAFETENSORS MODEL_SHA256 VALIDATION_ROOT\n";
  try {
    if (argc == 2 && fs::path(argv[1]) == fs::path("--help")) {
      std::cout << usage;
      return 0;
    }
    if (argc != 4) {
      std::cerr << usage;
      return 2;
    }
    std::string expected_sha;
    for (const auto* value = argv[2]; *value != 0; ++value) {
      const auto code = static_cast<unsigned int>(*value);
      Require(expected_sha.size() < 64 && ((code >= '0' && code <= '9') ||
                                           (code >= 'a' && code <= 'f')),
              "Expected a lowercase SHA-256 from the recipe result");
      expected_sha.push_back(static_cast<char>(code));
    }
    Require(expected_sha.size() == 64, "Expected a complete model SHA-256");
    return Run(fs::path(argv[1]), expected_sha, fs::path(argv[3]));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) { return Main(argc, argv); }
#else
int main(int argc, char* argv[]) { return Main(argc, argv); }
#endif
