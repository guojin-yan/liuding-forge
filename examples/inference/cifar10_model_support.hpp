// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
/** @file cifar10_model_support.hpp
 * @brief Reloads a recipe's complete ResNet18 state and evaluates CIFAR-10. */

#ifndef JYPPX_EXAMPLE_CIFAR10_MODEL_SUPPORT_HPP
#define JYPPX_EXAMPLE_CIFAR10_MODEL_SUPPORT_HPP

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
#include <atomic>
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

namespace cifar10_example {
namespace forge = jyppx::liuding;
namespace image = forge::data::image;
namespace fs = std::filesystem;
inline constexpr std::uint64_t kMiB = 1ULL << 20U;
inline constexpr std::size_t kSamples = 10000;
inline constexpr std::size_t kBatchSize = 128;
inline constexpr std::size_t kBatches = 79;
inline constexpr std::int64_t kSeed = 20261028;
inline constexpr std::size_t kStateTensors = 122;
inline constexpr std::array<std::string_view, 10> kClasses{
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"};

inline void Require(bool accepted, std::string_view message) {
  if (!accepted) throw std::runtime_error(std::string(message));
}

inline std::vector<std::byte> ReadModel(const fs::path& path) {
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

inline forge::interop::ModelWeightImportResult LoadModel(
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

inline std::string StateDigest(forge::nn::ModelAdapter& model) {
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
  explicit Images(const fs::path& root,
                  std::shared_ptr<image::ImageResourceBudget> budget = {}) {
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
    if (!budget) budget = std::make_shared<image::ImageResourceBudget>(limits);
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
      decoded_samples_.fetch_add(1, std::memory_order_relaxed);
      keys.push_back({sample.image_path, 0, 0});
      samples.push_back(image::transform_sample(sample, std::span(&resize, 1),
                                                geometry_, stop));
    }
    return image::collate_images(samples, keys, collate_, stop);
  }

  std::size_t decoded_samples() const noexcept {
    return decoded_samples_.load(std::memory_order_relaxed);
  }

  const image::ImageBatchLimits& limits() const noexcept {
    return collate_.limits;
  }

 private:
  mutable std::atomic<std::size_t> decoded_samples_{0};
  std::shared_ptr<image::ImageDataset> dataset_;
  image::JointTransformOptions geometry_;
  image::ImageCollateOptions collate_;
};

}  // namespace cifar10_example

#endif
