// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "dinov2_conversion.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace jyppx::liuding::tools::dinov2 {
namespace {
namespace fs = std::filesystem;
namespace st = interop;
using Json = nlohmann::json;
using Ordered = nlohmann::ordered_json;
constexpr std::string_view kMask = "embeddings.mask_token";
constexpr std::size_t kChunk = 1U << 20U;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::string text(const fs::path& path) {
  const auto value = path.generic_u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

st::SafeTensorsLimits limits() {
  st::SafeTensorsLimits value;
  value.max_file_bytes = kSourceSize;
  value.max_payload_bytes = kSourceSize;
  // fc1/fc2 have 1536*384 elements, greater than the position table.
  value.max_tensor_bytes = 4U * 1536U * 384U;
  value.max_tensor_elements = 1536U * 384U;
  value.max_dimension = 1536;
  value.max_header_bytes = kChunk;
  value.max_tensors = 223;
  value.max_rank = 4;
  value.max_metadata_entries = 16;
  value.max_string_bytes = 4096;
  value.max_total_string_bytes = kChunk;
  value.max_json_nodes = 16384;
  return value;
}

void check_source_path(const fs::path& source) {
  if (!fs::is_regular_file(source) || fs::file_size(source) != kSourceSize)
    fail(
        "Source must match pinned official size 88249960 and complete SHA-256");
}

std::vector<std::byte> read_source(const fs::path& source) {
  check_source_path(source);
  std::ifstream input(source, std::ios::binary);
  if (!input) fail("Cannot open source");
  std::vector<std::byte> bytes(static_cast<std::size_t>(kSourceSize));
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input || input.peek() != std::ifstream::traits_type::eof() ||
      sha256_hex(bytes) != kSourceSha)
    fail("Source differs from pinned official complete size/SHA-256");
  return bytes;
}

// Reopen and stream-check source immediately before publication. No tensor or
// model runtime is loaded, and hashing retains at most one MiB beyond bytes.
void check_file(const fs::path& path, std::uint64_t size,
                std::string_view expected_sha) {
  if (!fs::is_regular_file(path) || fs::file_size(path) != size)
    fail("Complete artifact size changed: " + text(path));
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("Cannot reopen artifact");
  std::vector<std::byte> chunk(kChunk);
  Sha256 hash;
  std::uint64_t remaining = size;
  while (remaining != 0) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, chunk.size()));
    input.read(reinterpret_cast<char*>(chunk.data()),
               static_cast<std::streamsize>(count));
    if (!input) fail("Artifact truncated during recheck");
    hash.update(std::span(chunk).first(count));
    remaining -= count;
  }
  if (input.peek() != std::ifstream::traits_type::eof() ||
      sha256_digest_hex(hash.finish()) != expected_sha)
    fail("Complete artifact SHA-256 changed: " + text(path));
}

class NewFile final {
 public:
  explicit NewFile(const fs::path& path) {
#ifdef _WIN32
    handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) fail("Cannot create new stage file");
#else
    descriptor_ =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor_ < 0) fail("Cannot create new stage file");
#endif
  }
  ~NewFile() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
    if (descriptor_ >= 0) ::close(descriptor_);
#endif
  }
  NewFile(const NewFile&) = delete;
  NewFile& operator=(const NewFile&) = delete;
  void write(std::span<const std::byte> bytes) {
    while (!bytes.empty()) {
      const auto count = std::min(bytes.size(), kChunk);
#ifdef _WIN32
      DWORD written = 0;
      if (!WriteFile(handle_, bytes.data(), static_cast<DWORD>(count), &written,
                     nullptr) ||
          written == 0)
        fail("Stage write failed");
      bytes = bytes.subspan(written);
#else
      const auto written = ::write(descriptor_, bytes.data(), count);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) fail("Stage write failed");
      bytes = bytes.subspan(static_cast<std::size_t>(written));
#endif
    }
  }
  void flush() {
#ifdef _WIN32
    if (!FlushFileBuffers(handle_)) fail("Stage flush failed");
#else
    if (::fsync(descriptor_) != 0) fail("Stage flush failed");
#endif
  }

 private:
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int descriptor_ = -1;
#endif
};

fs::path stage_directory(const fs::path& parent) {
  std::random_device random;
  for (unsigned attempt = 0; attempt < 64; ++attempt) {
    const auto name = ".liuding-dinov2-" + std::to_string(random()) + "-" +
                      std::to_string(random());
    const auto result = parent / name;
#ifdef _WIN32
    std::error_code error;
    if (fs::create_directory(result, error)) return result;
    if (error && error != std::errc::file_exists)
      throw fs::filesystem_error("Cannot create stage", result, error);
#else
    if (::mkdir(result.c_str(), 0700) == 0) return result;
    const int creation_error = errno;
    if (creation_error != EEXIST)
      throw fs::filesystem_error(
          "Cannot create private stage", result,
          std::error_code(creation_error, std::generic_category()));
#endif
  }
  fail("Cannot allocate unique private stage directory");
}

fs::path new_output(const fs::path& path) {
  if (path.filename().empty() || path.filename() == "." ||
      path.filename() == "..")
    fail("Output must name a new regular file");
  // Canonical parents resolve aliases before comparing source/output paths.
  const auto result =
      fs::canonical(fs::absolute(path).parent_path()) / path.filename();
  if (!fs::is_directory(result.parent_path())) fail("Output parent is missing");
  const auto status = fs::symlink_status(result);
  if (status.type() != fs::file_type::not_found)
    fail(
        "Output/report already exists (including symlink); refusing overwrite");
  return result;
}

void write_file(const fs::path& path, std::span<const std::byte> bytes) {
  NewFile output(path);
  output.write(bytes);
  output.flush();
}

std::span<const std::byte> bytes_of(const std::string& value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}

Json conversion_report(std::span<const std::byte> source) {
  const st::SafeTensorsView view(source, limits());
  Json mappings = Json::array();
  for (const auto& entry : mapping())
    mappings.push_back(
        {{"target", entry.target},
         {"sources", entry.sources},
         {"operation", entry.sources.size() > 1 ? "concat-dim0" : "identity"}});
  return {
      {"schema_id", "jyppx.liuding.weight-conversion"},
      {"schema_version", 1},
      {"mapping_version", kMappingVersion},
      {"source_uri", "https://huggingface.co/facebook/dinov2-small/resolve/" +
                         std::string(kRevision) + "/model.safetensors"},
      {"source_revision", kRevision},
      {"license_spdx", "Apache-2.0"},
      {"license_uri", "https://huggingface.co/facebook/dinov2-small/blob/" +
                          std::string(kRevision) + "/README.md"},
      {"source",
       {{"sha256", kSourceSha}, {"size_bytes", kSourceSize}, {"tensors", 223}}},
      {"output",
       {{"sha256", kOutputSha}, {"size_bytes", kOutputSize}, {"tensors", 174}}},
      {"required_model_options",
       {{"dinov2.position_antialias", false}, {"dinov2.num_classes", 0}}},
      {"mappings", std::move(mappings)},
      {"omitted", Json::array({{{"key", kMask},
                                {"reason", "unmasked-image-encoder-only"},
                                {"payload_sha256",
                                 sha256_hex(view.tensor_bytes(kMask))}}})},
      {"qualification",
       "data-layout conversion only; task accuracy and "
       "fine-tuning need separate validation"}};
}

void record_failure(const fs::path& stage,
                    const std::string& message) noexcept {
  if (stage.empty()) return;
  try {
    const auto raw = Json{{"status", "conversion_failed"},
                          {"message", message},
                          {"qualification", "not a published weight artifact"}}
                         .dump(2) +
                     "\n";
    write_file(stage / "FAILURE.json", bytes_of(raw));
  } catch (...) {
    // Preserve the original error and remaining bytes even if the disk is full.
  }
}
}  // namespace

std::vector<MappingEntry> mapping() {
  std::vector<MappingEntry> result{
      {"cls_token", {1, 1, 384}, {"embeddings.cls_token"}},
      {"pos_embed", {1, 1370, 384}, {"embeddings.position_embeddings"}},
      {"patch_embed.proj.weight",
       {384, 3, 14, 14},
       {"embeddings.patch_embeddings.projection.weight"}},
      {"patch_embed.proj.bias",
       {384},
       {"embeddings.patch_embeddings.projection.bias"}},
      {"norm.weight", {384}, {"layernorm.weight"}},
      {"norm.bias", {384}, {"layernorm.bias"}},
  };
  for (unsigned block = 0; block < 12; ++block) {
    const auto src = "encoder.layer." + std::to_string(block) + ".";
    const auto dst = "blocks." + std::to_string(block) + ".";
    for (const auto suffix : {"weight", "bias"}) {
      std::vector<std::uint64_t> shape{1152};
      if (std::string_view(suffix) == "weight") shape.push_back(384);
      std::vector<std::string> keys;
      for (const auto qkv : {"query", "key", "value"})
        keys.push_back(src + "attention.attention." + qkv + "." + suffix);
      result.push_back({dst + "attn.qkv." + suffix, shape, keys});
    }
    const auto add = [&](std::string_view target, std::string_view source,
                         std::vector<std::uint64_t> shape) {
      result.push_back({dst + std::string(target),
                        std::move(shape),
                        {src + std::string(source)}});
    };
    add("attn.proj.weight", "attention.output.dense.weight", {384, 384});
    add("attn.proj.bias", "attention.output.dense.bias", {384});
    add("ls1.gamma", "layer_scale1.lambda1", {384});
    add("ls2.gamma", "layer_scale2.lambda1", {384});
    for (const auto norm : {"norm1", "norm2"})
      for (const auto suffix : {"weight", "bias"}) {
        const auto key = std::string(norm) + "." + suffix;
        add(key, key, {384});
      }
    add("mlp.fc1.weight", "mlp.fc1.weight", {1536, 384});
    add("mlp.fc1.bias", "mlp.fc1.bias", {1536});
    add("mlp.fc2.weight", "mlp.fc2.weight", {384, 1536});
    add("mlp.fc2.bias", "mlp.fc2.bias", {384});
  }
  std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.target < b.target; });
  return result;
}

std::vector<std::byte> encode_layout(std::span<const std::byte> source) {
  const st::SafeTensorsView view(source, limits());
  std::map<std::string, const st::SafeTensorInfo*> descriptors;
  for (const auto& tensor : view.directory().tensors)
    descriptors.emplace(tensor.name, &tensor);
  if (descriptors.size() != 223 || !descriptors.contains(std::string(kMask)))
    fail("Expected complete 223-tensor official DINOv2-S inventory");
  const auto& mask = *descriptors.at(std::string(kMask));
  if (mask.dtype != st::SafeTensorDtype::f32 ||
      mask.shape != std::vector<std::uint64_t>{1, 384})
    fail("Unexpected mask token descriptor");
  std::set<std::string> consumed{std::string(kMask)};
  Ordered header = Ordered::object();
  std::uint64_t offset = 0;
  const auto entries = mapping();
  for (const auto& entry : entries) {
    std::uint64_t total = 0;
    auto expected = entry.shape;
    expected.front() /= entry.sources.size();
    for (const auto& key : entry.sources) {
      const auto found = descriptors.find(key);
      if (found == descriptors.end() || !consumed.insert(key).second)
        fail("Missing or duplicate source binding: " + key);
      const auto& descriptor = *found->second;
      if (descriptor.dtype != st::SafeTensorDtype::f32 ||
          descriptor.shape != expected)
        fail("Unexpected source dtype/shape: " + key);
      total += view.tensor_bytes(key).size();
    }
    // Fixed compatibility wire order: ordinary nlohmann::json alphabetizes
    // these fields and cannot preserve the reviewed whole-file SHA identity.
    Ordered descriptor = Ordered::object();
    descriptor["dtype"] = "F32";
    descriptor["shape"] = entry.shape;
    descriptor["data_offsets"] = Ordered::array({offset, offset + total});
    header[entry.target] = std::move(descriptor);
    offset += total;
  }
  if (entries.size() != 174 || consumed.size() != descriptors.size())
    fail("Incomplete upstream state accounting");
  auto raw = header.dump(-1, ' ', true);
  raw.append((8U - raw.size() % 8U) % 8U, ' ');
  if (raw.size() > kChunk || 8U + raw.size() + offset != kOutputSize)
    fail("Canonical output size/header bound mismatch");
  std::vector<std::byte> result;
  result.reserve(static_cast<std::size_t>(kOutputSize));
  for (unsigned byte = 0; byte < 8; ++byte)
    result.push_back(static_cast<std::byte>(
        (static_cast<std::uint64_t>(raw.size()) >> (8U * byte)) & 0xffU));
  const auto header_bytes = bytes_of(raw);
  result.insert(result.end(), header_bytes.begin(), header_bytes.end());
  for (const auto& entry : entries)
    for (const auto& key : entry.sources) {
      const auto payload = view.tensor_bytes(key);
      result.insert(result.end(), payload.begin(), payload.end());
    }
  // Reuse the complete bounded codec for both input and output validation.
  const st::SafeTensorsView output(result, limits());
  if (output.directory().tensors.size() != 174)
    fail("Canonical output inventory mismatch");
  return result;
}

Json convert(const fs::path& input, const fs::path& output,
             const fs::path& report_path) {
  const auto source = fs::canonical(input);
  const auto destination = new_output(output);
  const auto report = new_output(report_path);
  if (destination == report || destination == source || report == source)
    fail("Source, output and report must be distinct files");
  // All file/shape/SHA checks precede any publication. Peak explicit byte
  // storage is source + output (~169 MiB), plus <=1 MiB hashing and metadata.
  const auto original = read_source(source);
  const auto encoded = encode_layout(original);
  if (encoded.size() != kOutputSize || sha256_hex(encoded) != kOutputSha)
    fail("Converted artifact differs from reviewed canonical size/SHA-256");
  const auto result = conversion_report(original);
  const auto raw_report = result.dump(2, ' ', true) + "\n";
  fs::path model_stage;
  fs::path report_stage;
  bool model_linked = false;
  bool report_linked = false;
  try {
    model_stage = stage_directory(destination.parent_path());
    report_stage = stage_directory(report.parent_path());
    write_file(model_stage / "weights.safetensors", encoded);
    write_file(report_stage / "conversion.json", bytes_of(raw_report));
    check_file(model_stage / "weights.safetensors", kOutputSize, kOutputSha);
    check_file(source, kSourceSize, kSourceSha);
    // Both same-filesystem hard links are create-only; no replacing rename.
    fs::create_hard_link(model_stage / "weights.safetensors", destination);
    model_linked = true;
    fs::create_hard_link(report_stage / "conversion.json", report);
    report_linked = true;
  } catch (const std::exception& error) {
    std::string message = error.what();
    if (model_linked && !report_linked) {
      std::error_code rollback;
      const bool still_ours = fs::equivalent(
          model_stage / "weights.safetensors", destination, rollback);
      if (rollback || !still_ours || !fs::remove(destination, rollback) ||
          rollback)
        message += "; output rollback failed or identity changed: " +
                   text(destination);
    }
    message += "; retained model stage=" + text(model_stage) +
               "; report stage=" + text(report_stage);
    record_failure(model_stage, message);
    record_failure(report_stage, message);
    fail(message);
  }
  // Cleanup is best effort after both publications succeed. Never recursively
  // remove directories or report a complete pair as failed due to cleanup.
  std::error_code ignored;
  fs::remove(model_stage / "weights.safetensors", ignored);
  fs::remove(report_stage / "conversion.json", ignored);
  fs::remove(model_stage, ignored);
  fs::remove(report_stage, ignored);
  return result;
}
}  // namespace jyppx::liuding::tools::dinov2
