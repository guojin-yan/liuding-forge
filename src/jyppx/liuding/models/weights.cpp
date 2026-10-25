// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>
#include <jyppx/liuding/models/weights.hpp>

#include <array>
#include <atomic>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>
namespace jyppx::liuding::models {
namespace {
constexpr std::uint64_t kHardLimit = 16ULL << 30U;
bool Text(std::string_view value) {
  if (value.empty() || value.size() > 4096U) return false;
  for (const unsigned char ch : value)
    if (ch < 0x20U || ch == 0x7fU) return false;
  return true;
}
bool CanonicalCacheName(std::string_view name) {
  if (name.size() != 76U || name.substr(64) != ".safetensors") return false;
  for (const auto ch : name.substr(0, 64))
    if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
  return true;
}
void Bound(std::uint64_t limit) {
  if (limit == 0 || limit > kHardLimit ||
      limit > std::numeric_limits<std::size_t>::max() ||
      limit > static_cast<std::uint64_t>(
                  std::numeric_limits<std::streamsize>::max()))
    detail::Invalid("weights.limit",
                    "positive encoded limit no greater than 16GiB");
}
template <class Function>
decltype(auto) Filesystem(const char* operation, Function&& function) {
  try {
    return detail::Backend(operation, std::forward<Function>(function));
  } catch (const std::filesystem::filesystem_error&) {
    std::throw_with_nested(
        IoError("Weight cache filesystem operation failed",
                detail::Details(operation, "accessible trusted local cache")));
  }
}
void Regular(const std::filesystem::path& path) {
  const auto status = std::filesystem::symlink_status(path);
  if (std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status))
    throw IoError(
        "Weight cache entry is not a regular file",
        detail::Details("weights.cache", "regular data-only cache file"));
}
std::vector<std::byte> Read(const std::filesystem::path& path,
                            std::uint64_t expected, std::uint64_t limit) {
  Regular(path);
  const auto count = std::filesystem::file_size(path);
  if (count != expected || count > limit)
    throw SerializationError(
        "Weight cache size mismatch",
        detail::Details("weights.cache", "the exact pinned artifact size"));
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input)
    throw IoError(
        "Cannot open weight cache entry",
        detail::Details("weights.cache", "readable data-only cache file"));
  std::vector<std::byte> bytes(static_cast<std::size_t>(count));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw IoError(
        "Cannot read stable weight cache entry",
        detail::Details("weights.cache", "complete unchanged artifact bytes"));
  return bytes;
}
void CacheRoot(const std::filesystem::path& root, bool create) {
  if (root.empty())
    detail::Invalid("weights.cache", "nonempty trusted cache directory");
  if (create) std::filesystem::create_directories(root);
  const auto status = std::filesystem::symlink_status(root);
  if (std::filesystem::is_symlink(status) ||
      !std::filesystem::is_directory(status))
    throw IoError(
        "Invalid weight cache directory",
        detail::Details("weights.cache", "caller-owned regular directory"));
}
struct Staging {
  std::filesystem::path path;
  explicit Staging(const std::filesystem::path& root) {
    static std::atomic<std::uint64_t> sequence{0};
    for (std::uint64_t attempt = 0; attempt < 64U; ++attempt) {
      const auto tick =
          std::chrono::steady_clock::now().time_since_epoch().count();
      path = root / (".jyppx-stage-" + std::to_string(tick) + "-" +
                     std::to_string(sequence.fetch_add(1)));
      if (std::filesystem::create_directory(path)) return;
    }
    throw IoError(
        "Cannot reserve weight cache staging",
        detail::Details("weights.cache", "unique private staging directory"));
  }
  ~Staging() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};
WeightRecord Missing(const char* model, const char* url) {
  return {.id = std::string(model) + "-imagenet1k-v1",
          .model_name = model,
          .source_uri = url,
          .license_spdx = "NOASSERTION",
          .license_uri = "https://pytorch.org/vision/stable/models.html",
          .format = "pytorch-state-dict",
          .availability = WeightAvailability::not_provisioned,
          .artifact = std::nullopt,
          .status_reason =
              "Upstream candidate only; full bytes/hash and weight-specific "
              "license "
              "not provisioned. Native loader accepts independently converted "
              "safetensors only."};
}
}  // namespace
std::vector<WeightRecord> available_weights(std::string_view model) {
  if (model == "resnet18")
    return {
        Missing("resnet18",
                "https://download.pytorch.org/models/resnet18-f37072fd.pth")};
  if (model == "resnet50")
    return {
        Missing("resnet50",
                "https://download.pytorch.org/models/resnet50-0676ba61.pth")};
  if (model == "mobilenet_v3_small")
    return {Missing(
        "mobilenet_v3_small",
        "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth")};
  if (model == "mobilenet_v3_large")
    return {Missing(
        "mobilenet_v3_large",
        "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth")};
  if (model == "transformer_encoder") return {};
  if (model == "vit_small_patch14_dinov2" ||
      model == "vit_base_patch14_dinov2" ||
      model == "vit_large_patch14_dinov2" ||
      model == "vit_giant_patch14_dinov2") {
    auto record = Missing(model.data(), "https://huggingface.co/timm");
    record.id = std::string(model) + ".lvd142m";
    record.format = "safetensors";
    record.status_reason =
        "Candidate timm lvd142m lineage only; upstream revision, exact "
        "bytes, weight license and state-key mapping are not provisioned.";
    return {std::move(record)};
  }
  if (model == "dinov3_vits16" || model == "dinov3_vits16plus" ||
      model == "dinov3_vitb16" || model == "dinov3_convnext_tiny" ||
      model == "dinov3_convnext_small") {
    auto record = Missing(model.data(),
                          "https://github.com/facebookresearch/dinov3");
    record.id = std::string(model) + ".lvd1689m";
    record.format = "safetensors";
    record.license_uri = "https://github.com/facebookresearch/dinov3/blob/"
                         "6876159a11b4df116f30f667f8c9888617df0751/LICENSE";
    record.status_reason =
        "Candidate DINOv3 LVD1689M lineage only; upstream revision, exact "
        "bytes, weight license and state-key mapping are not provisioned.";
    return {std::move(record)};
  }
  detail::Invalid("weights.catalog", "a built-in model architecture");
}
std::string models_preview_manifest_json() {
  return R"JSON({"schema_id":"jyppx.models.preview","schema_version":1,"component":"models","default_enabled":false,"preview":true,"architectures":["resnet18","resnet50","mobilenet_v3_small","mobilenet_v3_large","transformer_encoder","vit_small_patch14_dinov2","vit_base_patch14_dinov2","vit_large_patch14_dinov2","vit_giant_patch14_dinov2","dinov3_vits16","dinov3_vits16plus","dinov3_vitb16","dinov3_convnext_tiny","dinov3_convnext_small"],"state_format":"safetensors","reference_workflow":"tests/unit/models/reference/generate_reference.py","reference_qualification":"requires-recorded-execution","reference_kind":"official-random-initialized-architecture-differential","pretrained_weights":"not-provisioned","pretrained_full_sha256":null,"weight_license":"NOASSERTION","release_qualification":"pending-independent-component-gates"})JSON";
}
void validate_weight_record(const WeightRecord& record) {
  if (!Text(record.id) || !Text(record.model_name) ||
      !Text(record.source_uri) || !Text(record.license_spdx) ||
      !Text(record.license_uri) || !Text(record.format) ||
      !Text(record.status_reason))
    detail::Invalid("weights.record", "bounded complete auditable metadata");
  if (record.availability == WeightAvailability::not_provisioned) {
    if (record.artifact)
      detail::Invalid("weights.record",
                      "missing record with no approved artifact identity");
    return;
  }
  if (record.availability != WeightAvailability::pinned_external ||
      record.format != "safetensors" || !record.artifact ||
      record.license_spdx == "NOASSERTION")
    detail::Invalid("weights.record",
                    "pinned data-only weights with explicit artifact license");
  validate_artifact_digest(record.artifact->digest);
  if (record.artifact->size_bytes == 0 || !Text(record.artifact->producer) ||
      !Text(record.artifact->logical_name) ||
      record.artifact->media_type != "application/x-safetensors")
    detail::Invalid("weights.artifact",
                    "nonempty exact artifact and safetensors media type");
}
void verify_weights(const WeightRecord& record,
                    std::span<const std::byte> bytes, std::uint64_t max_bytes) {
  detail::Backend("weights.verify", [&] {
    Bound(max_bytes);
    validate_weight_record(record);
    if (record.availability != WeightAvailability::pinned_external)
      throw NotReady("Model weights have not been provisioned",
                     detail::Details("weights.verify",
                                     "pinned externally approved weights"));
    if (bytes.size() > max_bytes)
      throw ResourceExhausted(
          "Model weight artifact exceeds encoded limit",
          detail::Details("weights.verify", "bounded artifact bytes"));
    if (bytes.size() != record.artifact->size_bytes ||
        sha256_hex(bytes) != record.artifact->digest.hex)
      throw SerializationError(
          "Model weight artifact identity mismatch",
          detail::Details("weights.verify", "exact pinned size/SHA256"));
    interop::SafeTensorsLimits limits;
    limits.max_file_bytes = max_bytes;
    limits.max_payload_bytes = max_bytes;
    limits.max_tensor_bytes = max_bytes;
    static_cast<void>(interop::SafeTensorsView(bytes, limits));
  });
}
interop::ModelWeightImportResult load_weights(
    nn::ModelAdapter& model, const WeightRecord& record,
    std::span<const std::byte> bytes,
    const interop::ModelWeightImportOptions& options) {
  verify_weights(record, bytes,
                 options.tensor_limits.max_file_bytes <= kHardLimit
                     ? options.tensor_limits.max_file_bytes
                     : kHardLimit);
  return interop::import_model_weights(model.state(), bytes,
                                       record.artifact->digest, options);
}
std::filesystem::path cache_weights(const WeightRecord& record,
                                    const std::filesystem::path& directory,
                                    const WeightFetcher& fetch,
                                    const WeightCacheOptions& options) {
  return Filesystem("weights.cache", [&] {
    Bound(options.max_bytes);
    validate_weight_record(record);
    if (record.availability != WeightAvailability::pinned_external)
      throw NotReady("Model weights have not been provisioned",
                     detail::Details("weights.cache",
                                     "pinned externally approved weights"));
    if (record.artifact->size_bytes > options.max_bytes)
      throw ResourceExhausted(
          "Pinned weight size exceeds cache limit",
          detail::Details("weights.cache", "bounded expected size"));
    CacheRoot(directory, true);
    const auto target =
        directory / (record.artifact->digest.hex + ".safetensors");
    const auto hit = [&] {
      auto bytes = Read(target, record.artifact->size_bytes, options.max_bytes);
      verify_weights(record, bytes, options.max_bytes);
      std::filesystem::last_write_time(
          target, std::filesystem::file_time_type::clock::now());
      return target;
    };
    if (std::filesystem::exists(std::filesystem::symlink_status(target)))
      return hit();
    if (!fetch)
      detail::Invalid("weights.fetch", "an explicit transport on a cache miss");
    const auto bytes = fetch(record, options.max_bytes);
    verify_weights(record, bytes, options.max_bytes);
    Staging staging(directory);
    const auto staged = staging.path / "weights";
    {
      std::ofstream output(staged,
                           std::ios::out | std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
      output.close();
      if (!output)
        throw IoError(
            "Cannot stage model weights",
            detail::Details("weights.cache", "complete private staging write"));
    }
    std::error_code publish_error;
    std::filesystem::create_hard_link(staged, target, publish_error);
    if (publish_error) {
      if (std::filesystem::exists(std::filesystem::symlink_status(target)))
        return hit();
      throw IoError(
          "Cannot publish model weight cache entry",
          detail::Details("weights.cache",
                          "filesystem supporting no-replace hard links"));
    }
    return hit();
  });
}
std::size_t clean_weight_cache(const std::filesystem::path& directory,
                               std::chrono::hours keep_age,
                               std::size_t max_entries) {
  return Filesystem("weights.clean", [&] {
    if (keep_age.count() < 0 || keep_age > std::chrono::hours(24 * 365 * 100) ||
        max_entries == 0 || max_entries > 1000000U)
      detail::Invalid("weights.clean",
                      "nonnegative bounded age and entry count");
    CacheRoot(directory, false);
    const auto cutoff =
        std::filesystem::file_time_type::clock::now() - keep_age;
    std::vector<std::filesystem::path> remove;
    std::size_t inspected = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      if (++inspected > max_entries)
        throw ResourceExhausted(
            "Weight cache directory entry limit exceeded",
            detail::Details("weights.clean", "bounded cache directory"));
      if (!CanonicalCacheName(entry.path().filename().string())) continue;
      const auto status = entry.symlink_status();
      if (!std::filesystem::is_symlink(status) &&
          std::filesystem::is_regular_file(status) &&
          entry.last_write_time() < cutoff)
        remove.push_back(entry.path());
    }
    std::size_t removed = 0;
    for (const auto& path : remove)
      if (std::filesystem::remove(path)) ++removed;
    return removed;
  });
}
}  // namespace jyppx::liuding::models
