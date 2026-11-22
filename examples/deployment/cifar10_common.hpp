// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_EXAMPLES_DEPLOYMENT_CIFAR10_COMMON_HPP
#define JYPPX_EXAMPLES_DEPLOYMENT_CIFAR10_COMMON_HPP
// Private example codec: CPU RGB32 NCHW, bounded batch, original class order.
#include <jyppx/liuding/data_image/image_batch.hpp>
#include <jyppx/liuding/data_image/joint_transform.hpp>
#include <jyppx/liuding/deployment/bundle_signing/provider.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx_example::cifar10 {
namespace forge = jyppx::liuding;
namespace image = forge::data::image;
namespace interop = forge::interop;
namespace signing = forge::bundle_signing;
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr std::uint64_t kMiB = 1ULL << 20U;
constexpr std::size_t kSamples = 10000;
constexpr std::array<std::string_view, 10> kClasses{
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"};

inline void Require(bool accepted, std::string_view message) {
  if (!accepted) throw std::runtime_error(std::string(message));
}
inline std::span<const std::byte> Bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
inline forge::ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.hex = forge::sha256_hex(bytes)};
}
inline std::string Text(std::span<const std::byte> bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
inline std::vector<std::byte> Read(const fs::path& path, std::uint64_t limit) {
  const auto size = fs::file_size(path);
  Require(size > 0 && size <= limit, "File exceeds its explicit byte bound");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(size));
  Require(input.good() && input.peek() == std::char_traits<char>::eof(),
          "Cannot read complete file");
  return bytes;
}
inline std::vector<std::byte> Pinned(const fs::path& path,
                                     std::string_view expected,
                                     std::uint64_t limit) {
  auto bytes = Read(path, limit);
  Require(Digest(bytes).hex == expected, "Input SHA-256 differs");
  return bytes;
}
inline void Write(const fs::path& path, std::span<const std::byte> bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  file.close();
  Require(static_cast<bool>(file), "Cannot write complete output");
}
inline void WriteJson(const fs::path& path, const Json& value) {
  Write(path, Bytes(value.dump(2) + "\n"));
}
inline void NewDirectory(const fs::path& path) {
  Require(path.is_absolute() && fs::create_directory(path),
          "Output requires a new absolute directory with an existing parent");
}
inline Json Preprocessing() {
  return {{"schema_version", 1},
          {"stage", "identity"},
          {"external_image_decode", "rgb-float32-0-to-1"},
          {"external_resize", {32, 32}},
          {"interpolation", "bilinear"},
          {"layout", "NCHW"},
          {"normalization", "none"}};
}
inline Json Postprocessing() {
  return {{"schema_version", 1},
          {"stage", "identity"},
          {"output", "logits"},
          {"class_count", 10}};
}
inline Json Descriptor(std::string_view schema, Json content) {
  return {{"schema_id", schema},
          {"schema_version", 1},
          {"content", std::move(content)}};
}

// Decoding/collation occur outside Serving. Its public boundary is CPU NCHW
// tensors, not encoded image bytes. Only one bounded batch is retained.
class Images final {
 public:
  explicit Images(const fs::path& root) {
    auto index = image::index_image_folder(root);
    Require(index.records().size() == kSamples &&
                index.categories().size() == kClasses.size(),
            "Expected the complete ten-class 10000-image validation split");
    for (std::size_t i = 0; i < kClasses.size(); ++i)
      Require(index.categories()[i].name == kClasses[i],
              "Class column order differs from this example");
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
    collate_.limits.max_samples = 128;
    collate_.limits.max_output_bytes = 64 * kMiB;
    collate_.limits.max_working_bytes = 64 * kMiB;
    collate_.limits.max_metadata_bytes = kMiB;
  }
  image::ImageBatch batch(std::span<const std::size_t> ids) const {
    Require(!ids.empty() && ids.size() <= 128, "Batch outside 1..128");
    std::vector<image::VisionSample> samples;
    std::vector<image::ImageBatchKey> keys;
    const image::ImageGeometryStep resize{
        .kind = image::ImageGeometryKind::resize, .extent = {32, 32}};
    for (auto id : ids) {
      auto sample = dataset_->get(id);
      keys.push_back({sample.image_path, 0, 0});
      samples.push_back(
          image::transform_sample(sample, std::span(&resize, 1), geometry_));
    }
    return image::collate_images(samples, keys, collate_);
  }

 private:
  std::shared_ptr<image::ImageDataset> dataset_;
  image::JointTransformOptions geometry_;
  image::ImageCollateOptions collate_;
};

inline interop::ModelBundleVerifyOptions Limits() {
  interop::ModelBundleVerifyOptions options;
  options.limits.max_file_bytes = 256 * kMiB;
  options.limits.max_total_bytes = 512 * kMiB;
  return options;
}
inline void Sign(std::span<const fs::path> args) {
  Require(args.size() == 3, "sign UNSIGNED EXPECTED_MANIFEST_SHA OUTPUT");
  NewDirectory(args[2]);
  const auto signer = signing::Ed25519Signer::generate();
  const auto policy_text =
      signing::serialize_ed25519_trust_policy(signer.public_key());
  const auto policy_digest = Digest(Bytes(policy_text));
  const auto policy_path = args[2] / "public-trust-policy.json";
  Write(policy_path, Bytes(policy_text));
  const auto policy =
      signing::Ed25519TrustPolicy::load(policy_path, policy_digest);
  const auto publication = signing::sign_model_bundle_directory(
      args[0], {.hex = args[1].string()}, args[2] / "bundle", signer, policy,
      Limits().limits);
  WriteJson(args[2] / "signing.json",
            {{"signature_algorithm", "Ed25519"},
             {"ephemeral_example_key", true},
             {"private_key_persisted", false},
             {"production_release_authority", false},
             {"policy_sha256", policy_digest.hex},
             {"manifest_sha256", publication.published.manifest_digest.hex}});
  std::cout << "Signed with an ephemeral in-memory key; public policy SHA-256="
            << policy_digest.hex << '\n';
}

inline at::Tensor Tensor(const interop::SafeTensorsView& view,
                         const std::string& name) {
  const auto& entries = view.directory().tensors;
  const auto found =
      std::find_if(entries.begin(), entries.end(),
                   [&](const auto& entry) { return entry.name == name; });
  Require(found != entries.end() &&
              found->dtype == interop::SafeTensorDtype::f32 &&
              !found->shape.empty() && found->shape.size() <= 4,
          "Warmup requires bounded FP32 tensor storage");
  std::vector<std::int64_t> shape(found->shape.begin(), found->shape.end());
  auto tensor =
      at::empty(shape, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  const auto bytes = view.tensor_bytes(name);
  Require(tensor.nbytes() == bytes.size(), "Warmup payload size differs");
  std::memcpy(tensor.mutable_data_ptr(), bytes.data(), bytes.size());
  Require(at::isfinite(tensor).all().item<bool>(), "Nonfinite warmup tensor");
  return tensor;
}
inline void Append(std::ofstream& stream, const at::Tensor& tensor) {
  Require(tensor.device().is_cpu() && tensor.is_contiguous(),
          "Binary output requires contiguous CPU tensors");
  stream.write(static_cast<const char*>(tensor.const_data_ptr()),
               static_cast<std::streamsize>(tensor.nbytes()));
  Require(static_cast<bool>(stream), "Cannot append evaluation tensor");
}

}  // namespace jyppx_example::cifar10
#endif
