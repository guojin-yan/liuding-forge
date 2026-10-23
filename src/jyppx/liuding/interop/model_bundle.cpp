// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "metadata_validation.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <system_error>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#endif

namespace jyppx::liuding::interop {
namespace {

namespace md = metadata_detail;
using Json = md::Json;
constexpr std::string_view kComponent = "interop.model_bundle";
constexpr std::string_view kSigningPrefix = "JYPPX-MODEL-BUNDLE-SIGNATURE-V1\n";

[[noreturn]] void Invalid(std::string_view operation,
                          std::string_view expected) {
  md::Invalid(kComponent, operation, expected);
}

[[noreturn]] void Malformed(std::string_view operation,
                            std::string_view expected) {
  md::Malformed(kComponent, operation, expected);
}

[[noreturn]] void Resource(std::string_view operation, std::uint64_t limit) {
  md::Resource(kComponent, operation, limit);
}

[[noreturn]] void Io(std::string_view operation) {
  throw IoError("Model bundle filesystem operation failed",
                md::Details(kComponent, operation,
                            "an accessible local unlinked regular file or "
                            "directory with the declared integrity"));
}

[[noreturn]] void Unsupported(std::string_view operation,
                              std::string_view expected) {
  throw CompatibilityError("Unsupported model bundle contract",
                           md::Details(kComponent, operation, expected));
}

template <typename Function>
auto Boundary(Function&& function, bool encoded = false) {
  try {
    return std::forward<Function>(function)();
  } catch (const InvalidArgument&) {
    if (encoded) Malformed("metadata", "valid schema-v1 metadata");
    throw;
  } catch (const Json::exception&) {
    Malformed("json", "strict bounded schema-v1 JSON");
  } catch (const std::bad_alloc&) {
    Resource("allocation", 0U);
  } catch (const std::length_error&) {
    Resource("allocation", 0U);
  } catch (const std::filesystem::filesystem_error&) {
    Io("filesystem");
  }
}

void Limits(const ModelBundleLimits& limits) {
  if (limits.max_manifest_bytes == 0U ||
      limits.max_manifest_bytes > 64U * 1024U * 1024U ||
      limits.max_files == 0U || limits.max_models == 0U ||
      limits.max_tensors == 0U || limits.max_rank == 0U ||
      limits.max_dimension == 0U || limits.max_tensor_elements == 0U ||
      limits.max_file_bytes == 0U || limits.max_total_bytes == 0U ||
      limits.max_string_bytes == 0U || limits.max_string_bytes > 4096U ||
      limits.max_path_bytes == 0U || limits.max_path_bytes > 4096U ||
      limits.max_signatures == 0U || limits.max_signature_bytes == 0U ||
      limits.max_signature_bytes > 2048U ||
      limits.max_file_bytes > static_cast<std::uint64_t>(
                                  (std::numeric_limits<std::size_t>::max)()) ||
      limits.max_file_bytes > static_cast<std::uint64_t>(
                                  (std::numeric_limits<std::ptrdiff_t>::max)()))
    Invalid("limits", "positive representable bounded resource limits");
}

void Text(std::string_view text, const ModelBundleLimits& limits,
          bool empty = false) {
  md::Text(text, limits.max_string_bytes, kComponent, "text", empty);
}

void Token(std::string_view text, const ModelBundleLimits& limits) {
  Text(text, limits);
  md::Token(text, kComponent, "identity");
}

void Count(std::size_t count, std::size_t maximum, std::string_view operation) {
  if (count > maximum) Resource(operation, maximum);
}

void Add(std::uint64_t& value, std::uint64_t amount, std::uint64_t maximum,
         std::string_view operation) {
  md::Add(value, amount, maximum, kComponent, operation);
}

void Path(std::string_view path, const ModelBundleLimits& limits) {
  Count(path.size(), limits.max_path_bytes, "path.bytes");
  if (path.empty() || path.front() == '/' || path.back() == '/')
    Invalid("path", "a nonempty portable relative file path");
  std::size_t begin = 0U;
  std::size_t components = 0U;
  while (begin < path.size()) {
    const auto slash = path.find('/', begin);
    const auto end = slash == std::string_view::npos ? path.size() : slash;
    const auto part = path.substr(begin, end - begin);
    if (++components > 16U) Resource("path.depth", 16U);
    if (part.empty() || part == "." || part == ".." || part.size() > 128U ||
        part.back() == '.' ||
        !std::all_of(part.begin(), part.end(), [](unsigned char byte) {
          return (byte >= 'a' && byte <= 'z') || (byte >= '0' && byte <= '9') ||
                 byte == '-' || byte == '_' || byte == '.';
        }))
      Invalid("path",
              "lowercase portable components without traversal/aliases");
    const auto base = part.substr(0U, part.find('.'));
    if (base == "con" || base == "prn" || base == "aux" || base == "nul" ||
        (base.size() == 4U &&
         (base.substr(0U, 3U) == "com" || base.substr(0U, 3U) == "lpt") &&
         base[3U] >= '0' && base[3U] <= '9'))
      Invalid("path", "a name without reserved Windows device aliases");
    begin = end + 1U;
  }
  constexpr std::array<std::string_view, 19U> executable{
      ".exe", ".dll", ".so",  ".dylib",  ".pyd", ".py",  ".pyc",
      ".sh",  ".bat", ".cmd", ".ps1",    ".com", ".scr", ".msi",
      ".pt",  ".pth", ".pkl", ".pickle", ".zip"};
  for (const auto suffix : executable) {
    if (path.ends_with(suffix) ||
        (suffix == ".so" && path.find(".so.") != std::string_view::npos))
      Unsupported("file.executable",
                  "data-only files without script/library/archive roles");
  }
}

std::uint64_t TensorBytes(const BundleTensor& tensor,
                          const ModelBundleLimits& limits) {
  Text(tensor.name, limits);
  if (tensor.name == "__metadata__")
    Invalid("tensor.name", "a tensor identity");
  Count(tensor.shape.size(), limits.max_rank, "tensor.rank");
  const auto width = safetensors_dtype_size(tensor.dtype);
  if (width == 0U) Unsupported("tensor.dtype", "a supported safetensors dtype");
  bool zero = false;
  for (const auto dimension : tensor.shape) {
    if (dimension > limits.max_dimension)
      Resource("tensor.dimension", limits.max_dimension);
    zero = zero || dimension == 0U;
  }
  std::uint64_t elements = zero ? 0U : 1U;
  if (!zero) {
    for (const auto dimension : tensor.shape) {
      if (dimension > limits.max_tensor_elements / elements)
        Resource("tensor.elements", limits.max_tensor_elements);
      elements *= dimension;
    }
  }
  if (elements > limits.max_tensor_elements)
    Resource("tensor.elements", limits.max_tensor_elements);
  if (elements > limits.max_file_bytes / width)
    Resource("tensor.bytes", limits.max_file_bytes);
  if (tensor.device_intent != "cpu" && tensor.device_intent != "cuda" &&
      tensor.device_intent != "any")
    Invalid("tensor.device_intent", "cpu, cuda, or any");
  Token(tensor.shard_file_id, limits);
  return elements * width;
}

void Version(std::string_view version, const ModelBundleLimits& limits) {
  Text(version, limits);
  if (version == "latest" ||
      version.find_first_of("*<>=^~ \t") != std::string_view::npos)
    Invalid("runtime.version", "an exact pinned version");
}

void Validate(const ModelBundleManifest& manifest,
              const ModelBundleLimits& limits) {
  Limits(limits);
  if (manifest.schema_version != ModelBundleManifest::format_version)
    Unsupported("schema_version", "model bundle schema 1");
  if (manifest.compression != "none" || manifest.encryption != "none")
    Unsupported("encoding", "uncompressed, unencrypted v1 data");
  Text(manifest.bundle_id, limits);
  const auto& build = manifest.build;
  for (const auto* field :
       {&build.jyppx_version, &build.git_commit, &build.compiler,
        &build.cxx_standard, &build.os_arch, &build.build_type,
        &build.libtorch_version, &build.libtorch_build,
        &build.accelerator_runtime, &build.dependency_lock_digest})
    Text(*field, limits);
  Count(manifest.files.size(), limits.max_files, "files.count");
  Count(manifest.models.size(), limits.max_models, "models.count");
  if (manifest.files.empty() || manifest.models.empty())
    Invalid("inventory", "at least one model with a complete file inventory");
  std::map<std::string, const BundleFile*> files;
  std::set<std::string> paths;
  std::uint64_t total = 0U;
  for (const auto& file : manifest.files) {
    Token(file.id, limits);
    Path(file.relative_path, limits);
    if (file.relative_path == kModelBundleManifest ||
        file.relative_path.starts_with(std::string(kModelBundleManifest) +
                                       "/") ||
        !files.emplace(file.id, &file).second ||
        !paths.insert(file.relative_path).second)
      Invalid("files.identity",
              "unique file ids and paths outside manifest.json");
    const auto role = static_cast<unsigned>(file.role);
    if (role < 1U || role > 9U)
      Unsupported("file.role", "a declared v1 data role");
    md::Artifact(file.artifact, kComponent, limits.max_file_bytes);
    Text(file.artifact.logical_name, limits, true);
    Text(file.artifact.media_type, limits);
    Text(file.artifact.producer, limits, true);
    Add(total, file.artifact.size_bytes, limits.max_total_bytes, "files.bytes");
    if (file.role == BundleFileRole::weights &&
        (!file.relative_path.ends_with(".safetensors") ||
         file.artifact.media_type != "application/safetensors"))
      Invalid("weights.format", "explicit application/safetensors data shards");
    if (file.role == BundleFileRole::model &&
        (!file.relative_path.ends_with(".onnx") ||
         file.artifact.media_type != "application/onnx"))
      Invalid("model.format", "explicit application/onnx model data");
  }
  for (const auto& path : paths) {
    for (auto slash = path.find('/'); slash != std::string::npos;
         slash = path.find('/', slash + 1U)) {
      if (paths.contains(path.substr(0U, slash)))
        Invalid("files.prefix", "no file/directory path collisions");
    }
  }
  std::set<std::string> owned;
  const auto claim = [&](const std::string& id, BundleFileRole role) {
    const auto found = files.find(id);
    if (found == files.end() || found->second->role != role ||
        !owned.insert(id).second)
      Invalid("reference", "one unique owner of a file with the declared role");
  };
  const auto descriptor = [&](const BundleDescriptorRef& ref,
                              BundleFileRole role) {
    Token(ref.schema_id, limits);
    if (ref.schema_version == 0U)
      Invalid("descriptor.version", "a positive owner codec version");
    Token(ref.file_id, limits);
    claim(ref.file_id, role);
  };
  descriptor(manifest.input_schema, BundleFileRole::schema);
  descriptor(manifest.output_schema, BundleFileRole::schema);
  descriptor(manifest.preprocessing, BundleFileRole::prepost);
  descriptor(manifest.postprocessing, BundleFileRole::prepost);
  if (manifest.labels) descriptor(*manifest.labels, BundleFileRole::labels);
  descriptor(manifest.evaluation, BundleFileRole::evaluation);
  descriptor(manifest.sbom, BundleFileRole::sbom);
  descriptor(manifest.provenance, BundleFileRole::provenance);
  Count(manifest.weight_import_reports.size(), limits.max_files,
        "reports.count");
  for (const auto& report : manifest.weight_import_reports)
    descriptor(report, BundleFileRole::weight_import_report);
  std::set<std::string> models;
  std::uint64_t tensors = 0U;
  for (const auto& model : manifest.models) {
    Token(model.id, limits);
    if (!models.insert(model.id).second || model.file_ids.empty())
      Invalid("model.identity", "unique model ids with complete data files");
    Count(model.file_ids.size(), limits.max_files, "shards.count");
    const bool safe = model.format == BundleModelFormat::safetensors;
    if (!safe && model.format != BundleModelFormat::onnx)
      Unsupported("model.format", "safetensors or single-file ONNX data");
    if (!safe && (model.file_ids.size() != 1U || !model.tensors.empty()))
      Unsupported("onnx.inventory",
                  "one opaque ONNX file without external-data resolution");
    std::set<std::string> shards;
    for (const auto& id : model.file_ids) {
      Token(id, limits);
      claim(id, safe ? BundleFileRole::weights : BundleFileRole::model);
      shards.insert(id);
    }
    Add(tensors, model.tensors.size(), limits.max_tensors, "tensors.count");
    std::set<std::string> names;
    for (const auto& tensor : model.tensors) {
      static_cast<void>(TensorBytes(tensor, limits));
      if (!names.insert(tensor.name).second ||
          !shards.contains(tensor.shard_file_id))
        Invalid("tensor.mapping", "unique names mapped to model-owned shards");
    }
  }
  if (owned.size() != files.size())
    Invalid("files.unreferenced", "every file referenced by exactly one owner");
  const auto& runtime = manifest.runtime;
  Token(runtime.backend_id, limits);
  Version(runtime.backend_version, limits);
  validate_artifact_digest(runtime.environment_lock);
  Count(runtime.dependencies.size(), limits.max_files, "runtime.dependencies");
  std::set<std::string> dependencies;
  for (const auto& dependency : runtime.dependencies) {
    Token(dependency.name, limits);
    Version(dependency.version, limits);
    if (!dependencies.insert(dependency.name).second)
      Invalid("runtime.dependencies", "unique pinned dependencies");
  }
  for (const auto* values :
       {&runtime.devices, &runtime.dtypes, &runtime.os_arch}) {
    if (values->empty())
      Invalid("runtime.matrix", "a nonempty explicit matrix");
    Count(values->size(), limits.max_files, "runtime.matrix");
    std::set<std::string> unique;
    for (const auto& value : *values) {
      Token(value, limits);
      if (!unique.insert(value).second)
        Invalid("runtime.matrix", "unique explicit runtime values");
    }
  }
  Count(manifest.signatures.size(), limits.max_signatures, "signatures.count");
  std::set<std::pair<std::string, std::string>> signers;
  for (const auto& signature : manifest.signatures) {
    Token(signature.algorithm, limits);
    Text(signature.key_id, limits);
    Count(signature.signature_hex.size(), limits.max_signature_bytes * 2U,
          "signature.bytes");
    if (signature.signature_hex.empty() ||
        signature.signature_hex.size() % 2U != 0U ||
        !std::all_of(signature.signature_hex.begin(),
                     signature.signature_hex.end(),
                     [](unsigned char byte) {
                       return (byte >= '0' && byte <= '9') ||
                              (byte >= 'a' && byte <= 'f');
                     }) ||
        !signers.emplace(signature.algorithm, signature.key_id).second)
      Invalid("signature.metadata",
              "unique signers with nonempty canonical hex");
  }
}

Json DescriptorJson(const BundleDescriptorRef& ref) {
  return {{"schema_id", ref.schema_id},
          {"schema_version", ref.schema_version},
          {"file_id", ref.file_id}};
}

template <typename Values, typename Key>
auto Sorted(const Values& values, Key key) {
  using Value = typename Values::value_type;
  std::vector<const Value*> sorted;
  sorted.reserve(values.size());
  for (const auto& value : values) sorted.push_back(&value);
  std::sort(sorted.begin(), sorted.end(),
            [&](const auto* a, const auto* b) { return key(*a) < key(*b); });
  return sorted;
}

struct SafeTensorName {
  const std::string& operator()(const SafeTensorSource& tensor) const {
    return tensor.name;
  }
};

Json SortedStrings(std::vector<std::string> values) {
  std::sort(values.begin(), values.end());
  return Json(values);
}

Json ManifestJson(const ModelBundleManifest& manifest, bool signatures) {
  Json files = Json::array();
  for (const auto* file :
       Sorted(manifest.files, [](const auto& f) { return f.id; }))
    files.push_back({{"id", file->id},
                     {"path", file->relative_path},
                     {"role", static_cast<unsigned>(file->role)},
                     {"artifact", md::ArtifactJson(file->artifact)}});
  Json models = Json::array();
  for (const auto* model :
       Sorted(manifest.models, [](const auto& m) { return m.id; })) {
    Json tensors = Json::array();
    for (const auto* tensor :
         Sorted(model->tensors, [](const auto& t) { return t.name; }))
      tensors.push_back({{"name", tensor->name},
                         {"dtype", safetensors_dtype_name(tensor->dtype)},
                         {"shape", tensor->shape},
                         {"device_intent", tensor->device_intent},
                         {"shard_file_id", tensor->shard_file_id}});
    models.push_back({{"id", model->id},
                      {"format", model->format == BundleModelFormat::safetensors
                                     ? "safetensors"
                                     : "onnx"},
                      {"file_ids", SortedStrings(model->file_ids)},
                      {"tensors", std::move(tensors)}});
  }
  Json reports = Json::array();
  for (const auto* report : Sorted(manifest.weight_import_reports,
                                   [](const auto& r) { return r.file_id; }))
    reports.push_back(DescriptorJson(*report));
  const auto& runtime = manifest.runtime;
  Json dependencies = Json::array();
  for (const auto* dependency :
       Sorted(runtime.dependencies, [](const auto& d) { return d.name; }))
    dependencies.push_back(
        {{"name", dependency->name}, {"version", dependency->version}});
  Json result{{"format", "jyppx.model_bundle"},
              {"schema_version", manifest.schema_version},
              {"bundle_id", manifest.bundle_id},
              {"build", Json::parse(build_fingerprint_json(manifest.build))},
              {"compression", manifest.compression},
              {"encryption", manifest.encryption},
              {"files", std::move(files)},
              {"models", std::move(models)},
              {"input_schema", DescriptorJson(manifest.input_schema)},
              {"output_schema", DescriptorJson(manifest.output_schema)},
              {"preprocessing", DescriptorJson(manifest.preprocessing)},
              {"postprocessing", DescriptorJson(manifest.postprocessing)},
              {"labels", manifest.labels ? DescriptorJson(*manifest.labels)
                                         : Json(nullptr)},
              {"evaluation", DescriptorJson(manifest.evaluation)},
              {"sbom", DescriptorJson(manifest.sbom)},
              {"provenance", DescriptorJson(manifest.provenance)},
              {"weight_import_reports", std::move(reports)},
              {"runtime",
               {{"backend_id", runtime.backend_id},
                {"backend_version", runtime.backend_version},
                {"environment_lock", md::DigestJson(runtime.environment_lock)},
                {"dependencies", std::move(dependencies)},
                {"devices", SortedStrings(runtime.devices)},
                {"dtypes", SortedStrings(runtime.dtypes)},
                {"os_arch", SortedStrings(runtime.os_arch)},
                {"dynamic_shapes", runtime.dynamic_shapes}}}};
  if (signatures) {
    result["signatures"] = Json::array();
    for (const auto* signature : Sorted(manifest.signatures, [](const auto& s) {
           return std::pair(s.algorithm, s.key_id);
         }))
      result["signatures"].push_back(
          {{"algorithm", signature->algorithm},
           {"key_id", signature->key_id},
           {"signature_hex", signature->signature_hex}});
  }
  return result;
}

Json Parse(std::string_view bytes, const ModelBundleLimits& limits) {
  Limits(limits);
  Count(bytes.size(), limits.max_manifest_bytes, "manifest.bytes");
  if (bytes.empty() || bytes.front() != '{' || !md::Utf8(bytes))
    Malformed("manifest.utf8", "a strict UTF-8 JSON object");
  std::vector<std::set<std::string>> keys;
  std::uint64_t nodes = 0U;
  auto callback = [&](int depth, Json::parse_event_t event, Json& value) {
    if ((event == Json::parse_event_t::object_start ||
         event == Json::parse_event_t::array_start) &&
        depth >= 16)
      Resource("manifest.depth", 16U);
    Add(nodes, 1U, limits.max_manifest_bytes, "manifest.nodes");
    if (event == Json::parse_event_t::object_start) keys.emplace_back();
    if (event == Json::parse_event_t::key &&
        (keys.empty() || !keys.back().insert(value.get<std::string>()).second))
      Malformed("manifest.keys", "unique decoded JSON object keys");
    if (event == Json::parse_event_t::object_end) keys.pop_back();
    if ((event == Json::parse_event_t::key ||
         event == Json::parse_event_t::value) &&
        value.is_string())
      Count(value.get_ref<const std::string&>().size(),
            std::max({limits.max_string_bytes, limits.max_path_bytes,
                      limits.max_signature_bytes * 2U}),
            "manifest.string");
    return true;
  };
  return Json::parse(bytes.begin(), bytes.end(), callback, true, false);
}

std::uint32_t Uint32(const Json& value) {
  const auto number = md::Uint(value, kComponent);
  if (number > (std::numeric_limits<std::uint32_t>::max)())
    Malformed("integer", "a uint32 version");
  return static_cast<std::uint32_t>(number);
}

const Json& Array(const Json& value, std::size_t maximum) {
  if (!value.is_array()) Malformed("array", "a bounded JSON array");
  Count(value.size(), maximum, "array.count");
  return value;
}

std::string String(const Json& value) { return md::String(value, kComponent); }

BundleDescriptorRef ReadDescriptor(const Json& value) {
  md::Fields(value, {"schema_id", "schema_version", "file_id"}, kComponent);
  return {.schema_id = String(value.at("schema_id")),
          .schema_version = Uint32(value.at("schema_version")),
          .file_id = String(value.at("file_id"))};
}

std::vector<std::string> ReadStrings(const Json& value, std::size_t maximum) {
  std::vector<std::string> result;
  for (const auto& item : Array(value, maximum)) result.push_back(String(item));
  return result;
}

ModelBundleManifest ReadManifest(const Json& value,
                                 const ModelBundleLimits& limits) {
  md::Fields(value,
             {"format", "schema_version", "bundle_id", "build", "compression",
              "encryption", "files", "models", "input_schema", "output_schema",
              "preprocessing", "postprocessing", "labels", "evaluation", "sbom",
              "provenance", "weight_import_reports", "runtime", "signatures"},
             kComponent);
  if (String(value.at("format")) != "jyppx.model_bundle")
    Unsupported("format", "jyppx.model_bundle");
  ModelBundleManifest result;
  result.schema_version = Uint32(value.at("schema_version"));
  if (result.schema_version != ModelBundleManifest::format_version)
    Unsupported("schema_version", "model bundle schema 1");
  result.bundle_id = String(value.at("bundle_id"));
  result.compression = String(value.at("compression"));
  result.encryption = String(value.at("encryption"));
  const auto& build = value.at("build");
  md::Fields(build,
             {"jyppx_version", "git_commit", "compiler", "cxx_standard",
              "os_arch", "build_type", "libtorch_version", "libtorch_build",
              "accelerator_runtime", "dependency_lock_digest"},
             kComponent);
  result.build = {
      .jyppx_version = String(build.at("jyppx_version")),
      .git_commit = String(build.at("git_commit")),
      .compiler = String(build.at("compiler")),
      .cxx_standard = String(build.at("cxx_standard")),
      .os_arch = String(build.at("os_arch")),
      .build_type = String(build.at("build_type")),
      .libtorch_version = String(build.at("libtorch_version")),
      .libtorch_build = String(build.at("libtorch_build")),
      .accelerator_runtime = String(build.at("accelerator_runtime")),
      .dependency_lock_digest = String(build.at("dependency_lock_digest"))};
  for (const auto& file : Array(value.at("files"), limits.max_files)) {
    md::Fields(file, {"id", "path", "role", "artifact"}, kComponent);
    const auto role = md::Uint(file.at("role"), kComponent);
    if (role < 1U || role > 9U) Unsupported("file.role", "a v1 data role");
    result.files.push_back(
        {.id = String(file.at("id")),
         .relative_path = String(file.at("path")),
         .role = static_cast<BundleFileRole>(role),
         .artifact = md::ReadArtifact(file.at("artifact"), kComponent)});
  }
  std::uint64_t tensor_count = 0U;
  for (const auto& model : Array(value.at("models"), limits.max_models)) {
    md::Fields(model, {"id", "format", "file_ids", "tensors"}, kComponent);
    const auto format = String(model.at("format"));
    if (format != "safetensors" && format != "onnx")
      Unsupported("model.format", "safetensors or onnx");
    BundleModel candidate{
        .id = String(model.at("id")),
        .format = format == "safetensors" ? BundleModelFormat::safetensors
                                          : BundleModelFormat::onnx,
        .file_ids = ReadStrings(model.at("file_ids"), limits.max_files),
        .tensors = {}};
    for (const auto& tensor : Array(model.at("tensors"), limits.max_tensors)) {
      Add(tensor_count, 1U, limits.max_tensors, "tensors.count");
      md::Fields(tensor,
                 {"name", "dtype", "shape", "device_intent", "shard_file_id"},
                 kComponent);
      BundleTensor entry{
          .name = String(tensor.at("name")),
          .dtype = safetensors_dtype_from_name(String(tensor.at("dtype"))),
          .shape = {},
          .device_intent = String(tensor.at("device_intent")),
          .shard_file_id = String(tensor.at("shard_file_id"))};
      for (const auto& dimension : Array(tensor.at("shape"), limits.max_rank))
        entry.shape.push_back(md::Uint(dimension, kComponent));
      candidate.tensors.push_back(std::move(entry));
    }
    result.models.push_back(std::move(candidate));
  }
  result.input_schema = ReadDescriptor(value.at("input_schema"));
  result.output_schema = ReadDescriptor(value.at("output_schema"));
  result.preprocessing = ReadDescriptor(value.at("preprocessing"));
  result.postprocessing = ReadDescriptor(value.at("postprocessing"));
  if (!value.at("labels").is_null())
    result.labels = ReadDescriptor(value.at("labels"));
  result.evaluation = ReadDescriptor(value.at("evaluation"));
  result.sbom = ReadDescriptor(value.at("sbom"));
  result.provenance = ReadDescriptor(value.at("provenance"));
  for (const auto& report :
       Array(value.at("weight_import_reports"), limits.max_files))
    result.weight_import_reports.push_back(ReadDescriptor(report));
  const auto& runtime = value.at("runtime");
  md::Fields(runtime,
             {"backend_id", "backend_version", "environment_lock",
              "dependencies", "devices", "dtypes", "os_arch", "dynamic_shapes"},
             kComponent);
  result.runtime.backend_id = String(runtime.at("backend_id"));
  result.runtime.backend_version = String(runtime.at("backend_version"));
  result.runtime.environment_lock =
      md::ReadDigest(runtime.at("environment_lock"), kComponent);
  result.runtime.devices = ReadStrings(runtime.at("devices"), limits.max_files);
  result.runtime.dtypes = ReadStrings(runtime.at("dtypes"), limits.max_files);
  result.runtime.os_arch = ReadStrings(runtime.at("os_arch"), limits.max_files);
  result.runtime.dynamic_shapes =
      md::Boolean(runtime.at("dynamic_shapes"), kComponent);
  for (const auto& dependency :
       Array(runtime.at("dependencies"), limits.max_files)) {
    md::Fields(dependency, {"name", "version"}, kComponent);
    result.runtime.dependencies.push_back(
        {.name = String(dependency.at("name")),
         .version = String(dependency.at("version"))});
  }
  for (const auto& signature :
       Array(value.at("signatures"), limits.max_signatures)) {
    md::Fields(signature, {"algorithm", "key_id", "signature_hex"}, kComponent);
    result.signatures.push_back(
        {.algorithm = String(signature.at("algorithm")),
         .key_id = String(signature.at("key_id")),
         .signature_hex = String(signature.at("signature_hex"))});
  }
  Validate(result, limits);
  return result;
}

SafeTensorsLimits TensorLimits(const ModelBundleLimits& limits) {
  SafeTensorsLimits result;
  result.max_file_bytes = limits.max_file_bytes;
  result.max_payload_bytes = limits.max_file_bytes;
  result.max_tensor_bytes = limits.max_file_bytes;
  result.max_tensor_elements = limits.max_tensor_elements;
  result.max_dimension = limits.max_dimension;
  result.max_tensors = limits.max_tensors;
  result.max_rank = limits.max_rank;
  result.max_string_bytes = limits.max_string_bytes;
  result.max_header_bytes =
      (std::min)(limits.max_manifest_bytes, result.max_header_bytes);
  return result;
}

std::span<const std::byte> Bytes(std::string_view value) {
  return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

ArtifactDigest Digest(std::span<const std::byte> value) {
  return {.algorithm = ArtifactDigestAlgorithm::sha256,
          .hex = sha256_hex(value)};
}

BundleSignatureStatus Signatures(const ModelBundleManifest& manifest,
                                 const ModelBundleVerifyOptions& options) {
  if (manifest.signatures.empty()) {
    if (options.require_trusted_signature)
      Unsupported("signature.required", "at least one trusted signature");
    return BundleSignatureStatus::unsigned_bundle;
  }
  if (!options.signature_verifier) {
    if (options.require_trusted_signature)
      Unsupported("signature.verifier",
                  "an explicit trusted cryptographic verifier");
    return BundleSignatureStatus::present_unverified;
  }
  const auto payload = model_bundle_signing_payload(manifest, options.limits);
  for (const auto& signature : manifest.signatures) {
    bool accepted = false;
    try {
      accepted = options.signature_verifier(signature, Bytes(payload));
    } catch (...) {
      throw BackendError(
          "Model bundle signature verifier failed",
          md::Details(kComponent, "signature.callback",
                      "a successful trusted verification decision"));
    }
    if (!accepted)
      Malformed("signature.trust", "all signatures accepted by trusted policy");
  }
  return BundleSignatureStatus::trusted;
}

// Native handles below pin each traversed directory and never follow a leaf
// symlink/reparse point. POSIX reads additionally detect ordinary in-place
// mutations with fstat before/after; a verification report is not a read lease.
#ifdef _WIN32
using HandleValue = HANDLE;
const HandleValue kInvalidHandle = INVALID_HANDLE_VALUE;
#else
using HandleValue = int;
constexpr HandleValue kInvalidHandle = -1;
#endif

class Handle final {
 public:
  explicit Handle(HandleValue value = kInvalidHandle) : value_(value) {}
  ~Handle() { close(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept
      : value_(std::exchange(other.value_, kInvalidHandle)) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other) {
      close();
      value_ = std::exchange(other.value_, kInvalidHandle);
    }
    return *this;
  }
  HandleValue get() const noexcept { return value_; }

 private:
  void close() noexcept {
    if (value_ == kInvalidHandle) return;
#ifdef _WIN32
    CloseHandle(value_);
#else
    ::close(value_);
#endif
    value_ = kInvalidHandle;
  }
  HandleValue value_;
};

std::filesystem::path Absolute(const std::filesystem::path& path) {
  if (path.empty()) Invalid("directory.path", "a nonempty local path");
  for (const auto& component : path) {
    if (component == "..") Invalid("directory.path", "no parent traversal");
  }
  std::error_code error;
  auto result = std::filesystem::absolute(path, error).lexically_normal();
  if (error) Io("path.absolute");
#ifdef _WIN32
  const auto root = result.root_name().native();
  if (root.size() != 2U || root[1U] != L':')
    Unsupported("path.root",
                "a local Windows drive path, not UNC/device paths");
#endif
  return result;
}

class Directory final {
 public:
  explicit Directory(const std::filesystem::path& path)
      : path_(Absolute(path)) {
    auto prefix = path_.root_path();
#ifdef _WIN32
    const auto open = [&](const std::filesystem::path& part) {
      Handle handle(CreateFileW(
          part.c_str(), FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES,
          FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
          FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
      if (handle.get() == kInvalidHandle) Io("directory.open");
      BY_HANDLE_FILE_INFORMATION info{};
      if (!GetFileInformationByHandle(handle.get(), &info) ||
          (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0U ||
          (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U)
        Io("directory.no_follow");
      handles_.push_back(std::move(handle));
    };
    open(prefix);
    for (const auto& component : path_.relative_path()) {
      if (component == "." || component.empty()) continue;
      prefix /= component;
      open(prefix);
    }
#else
    Handle root(::open(prefix.c_str(),
                       O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (root.get() == kInvalidHandle) Io("directory.open");
    handles_.push_back(std::move(root));
    for (const auto& component : path_.relative_path()) {
      if (component == "." || component.empty()) continue;
      Handle child(::openat(get(), component.c_str(),
                            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
      if (child.get() == kInvalidHandle) Io("directory.no_follow");
      handles_.push_back(std::move(child));
    }
#endif
  }
  const std::filesystem::path& path() const noexcept { return path_; }
  HandleValue get() const noexcept { return handles_.back().get(); }
  void sync() const {
#ifndef _WIN32
    if (::fsync(get()) != 0) Io("directory.flush");
#endif
  }
  void create(std::string_view name) const {
    const std::string local(name);
#ifdef _WIN32
    if (!CreateDirectoryW((path_ / local).c_str(), nullptr))
      Io("directory.create");
#else
    if (::mkdirat(get(), local.c_str(), 0700) != 0) Io("directory.create");
#endif
  }
  struct Entry {
    std::string name;
    bool directory;
  };
  std::vector<Entry> entries(std::size_t maximum) const {
    std::vector<Entry> result;
#ifdef _WIN32
    std::error_code error;
    std::filesystem::directory_iterator it(path_, error);
    if (error) Io("directory.list");
    for (const auto& entry : it) {
      const auto utf8 = entry.path().filename().generic_u8string();
      const std::string name(reinterpret_cast<const char*>(utf8.data()),
                             utf8.size());
      const auto attributes = GetFileAttributesW(entry.path().c_str());
      if (attributes == INVALID_FILE_ATTRIBUTES ||
          (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U)
        Io("directory.entry.no_follow");
      Count(result.size() + 1U, maximum, "directory.entries");
      result.push_back({name, (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0U});
    }
#else
    const int duplicate = ::dup(get());
    if (duplicate < 0) Io("directory.list");
    DIR* raw = ::fdopendir(duplicate);
    if (!raw) {
      ::close(duplicate);
      Io("directory.list");
    }
    const auto closer = [](DIR* directory) { ::closedir(directory); };
    const std::unique_ptr<DIR, decltype(closer)> directory(raw, closer);
    ::rewinddir(directory.get());
    while (true) {
      errno = 0;
      const auto* entry = ::readdir(directory.get());
      if (!entry) {
        if (errno != 0) Io("directory.list");
        break;
      }
      const std::string name(entry->d_name);
      if (name == "." || name == "..") continue;
      struct stat info {};
      if (::fstatat(get(), name.c_str(), &info, AT_SYMLINK_NOFOLLOW) != 0 ||
          (!S_ISDIR(info.st_mode) && !S_ISREG(info.st_mode)))
        Io("directory.entry.no_follow");
      Count(result.size() + 1U, maximum, "directory.entries");
      result.push_back({name, S_ISDIR(info.st_mode)});
    }
#endif
    return result;
  }

 private:
  std::filesystem::path path_;
  std::vector<Handle> handles_;
};

class Input final {
 public:
  Input(const Directory& parent, const std::filesystem::path& filename) {
#ifdef _WIN32
    handle_ = Handle(CreateFileW(
        (parent.path() / filename).c_str(), GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING,
        FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
    BY_HANDLE_FILE_INFORMATION info{};
    LARGE_INTEGER size{};
    if (handle_.get() == kInvalidHandle ||
        !GetFileInformationByHandle(handle_.get(), &info) ||
        (info.dwFileAttributes &
         (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) != 0U ||
        GetFileType(handle_.get()) != FILE_TYPE_DISK ||
        info.nNumberOfLinks != 1U || !GetFileSizeEx(handle_.get(), &size) ||
        size.QuadPart < 0)
      Io("file.open.no_follow");
    size_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    handle_ = Handle(::openat(parent.get(), filename.c_str(),
                              O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
    if (handle_.get() == kInvalidHandle ||
        ::fstat(handle_.get(), &before_) != 0 || !S_ISREG(before_.st_mode) ||
        before_.st_nlink != 1U || before_.st_size < 0)
      Io("file.open.no_follow");
    size_ = static_cast<std::uint64_t>(before_.st_size);
#endif
  }
  std::uint64_t size() const noexcept { return size_; }
  std::size_t read(std::span<std::byte> bytes) {
#ifdef _WIN32
    DWORD count = 0U;
    if (!ReadFile(handle_.get(), bytes.data(), static_cast<DWORD>(bytes.size()),
                  &count, nullptr))
      Io("file.read");
    return count;
#else
    ssize_t count;
    do {
      count = ::read(handle_.get(), bytes.data(), bytes.size());
    } while (count < 0 && errno == EINTR);
    if (count < 0) Io("file.read");
    return static_cast<std::size_t>(count);
#endif
  }
  void stable() const {
#ifndef _WIN32
    struct stat after {};
    if (::fstat(handle_.get(), &after) != 0 ||
        before_.st_size != after.st_size ||
        before_.st_nlink != after.st_nlink ||
        before_.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before_.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before_.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before_.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
      Io("file.changed_during_read");
#endif
  }

 private:
  Handle handle_;
  std::uint64_t size_ = 0U;
#ifndef _WIN32
  struct stat before_ {};
#endif
};

class Output final {
 public:
  Output(const Directory& parent, const std::filesystem::path& filename) {
#ifdef _WIN32
    handle_ = Handle(CreateFileW(
        (parent.path() / filename).c_str(), GENERIC_WRITE, 0U, nullptr,
        CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT,
        nullptr));
#else
    handle_ = Handle(
        ::openat(parent.get(), filename.c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
#endif
    if (handle_.get() == kInvalidHandle) Io("stage.file.create");
  }
  void write(std::span<const std::byte> bytes) {
    while (!bytes.empty()) {
      const auto bounded = (std::min)(bytes.size(), std::size_t{65536U});
#ifdef _WIN32
      DWORD count = 0U;
      if (!WriteFile(handle_.get(), bytes.data(), static_cast<DWORD>(bounded),
                     &count, nullptr) ||
          count == 0U)
        Io("stage.file.write");
#else
      ssize_t count;
      do {
        count = ::write(handle_.get(), bytes.data(), bounded);
      } while (count < 0 && errno == EINTR);
      if (count <= 0) Io("stage.file.write");
#endif
      bytes = bytes.subspan(static_cast<std::size_t>(count));
    }
  }
  void sync() const {
#ifdef _WIN32
    if (!FlushFileBuffers(handle_.get())) Io("stage.file.flush");
#else
    if (::fsync(handle_.get()) != 0) Io("stage.file.flush");
#endif
  }

 private:
  Handle handle_;
};

bool ExecutableMagic(std::span<const std::byte> bytes) {
  const auto byte = [&](std::size_t index) {
    return std::to_integer<unsigned>(bytes[index]);
  };
  if (bytes.size() >= 2U && ((byte(0U) == '#' && byte(1U) == '!') ||
                             (byte(0U) == 'M' && byte(1U) == 'Z')))
    return true;
  if (bytes.size() < 4U) return false;
  const auto magic =
      (byte(0U) << 24U) | (byte(1U) << 16U) | (byte(2U) << 8U) | byte(3U);
  return magic == 0x7f454c46U || magic == 0xfeedfaceU || magic == 0xcefaedfeU ||
         magic == 0xfeedfacfU || magic == 0xcffaedfeU || magic == 0xcafebabeU;
}

struct ReadResult {
  ArtifactDigest digest;
  std::vector<std::byte> bytes;
};

ReadResult Consume(Input& input, std::uint64_t maximum, bool retain,
                   Output* output = nullptr) {
  if (input.size() > maximum) Resource("file.bytes", maximum);
  ReadResult result;
  if (retain) result.bytes.reserve(static_cast<std::size_t>(input.size()));
  Sha256 hash;
  std::array<std::byte, 65536U> buffer{};
  std::uint64_t total = 0U;
  bool first = true;
  while (true) {
    const auto count = input.read(buffer);
    if (count == 0U) break;
    const auto bytes = std::span<const std::byte>(buffer).first(count);
    if (first && ExecutableMagic(bytes))
      Unsupported("file.executable_magic",
                  "data without executable/script magic");
    first = false;
    Add(total, count, input.size(), "file.changed_size");
    hash.update(bytes);
    if (retain)
      result.bytes.insert(result.bytes.end(), bytes.begin(), bytes.end());
    if (output) output->write(bytes);
  }
  if (total != input.size()) Io("file.truncated");
  input.stable();
  result.digest = {.algorithm = ArtifactDigestAlgorithm::sha256,
                   .hex = sha256_digest_hex(hash.finish())};
  return result;
}

void Walk(const Directory& directory, std::string_view relative,
          const std::set<std::string>& expected_files,
          const std::set<std::string>& expected_directories,
          std::set<std::string>& seen_files,
          std::set<std::string>& seen_directories,
          const ModelBundleLimits& limits, bool synchronize = false) {
  for (const auto& entry : directory.entries(
           expected_files.size() + expected_directories.size() + 1U)) {
    const auto path = relative.empty()
                          ? entry.name
                          : std::string(relative) + "/" + entry.name;
    Path(path, limits);
    if (entry.directory) {
      if (!expected_directories.contains(path) ||
          !seen_directories.insert(path).second)
        Malformed("directory.inventory",
                  "exactly declared file parent directories");
      const Directory child(directory.path() / entry.name);
      Walk(child, path, expected_files, expected_directories, seen_files,
           seen_directories, limits, synchronize);
    } else if (!expected_files.contains(path) ||
               !seen_files.insert(path).second) {
      Malformed("directory.inventory", "exactly declared regular data files");
    }
  }
  if (synchronize) directory.sync();
}

void Inventory(const Directory& root, const ModelBundleManifest& manifest,
               const ModelBundleLimits& limits, bool synchronize = false) {
  std::set<std::string> files{std::string(kModelBundleManifest)};
  std::set<std::string> directories;
  for (const auto& file : manifest.files) {
    files.insert(file.relative_path);
    for (auto slash = file.relative_path.find('/'); slash != std::string::npos;
         slash = file.relative_path.find('/', slash + 1U))
      directories.insert(file.relative_path.substr(0U, slash));
  }
  std::set<std::string> seen_files;
  std::set<std::string> seen_directories;
  Walk(root, "", files, directories, seen_files, seen_directories, limits,
       synchronize);
  if (files != seen_files || directories != seen_directories)
    Malformed("directory.missing",
              "all declared files and directories present");
}

ModelBundleVerification Verify(const Directory& directory,
                               const ModelBundleVerifyOptions& options) {
  Input manifest_input(directory, std::string(kModelBundleManifest));
  auto manifest_bytes =
      Consume(manifest_input, options.limits.max_manifest_bytes, true);
  const auto json = std::string_view(
      reinterpret_cast<const char*>(manifest_bytes.bytes.data()),
      manifest_bytes.bytes.size());
  ModelBundleVerification result;
  result.manifest = deserialize_model_bundle_manifest(json, options.limits);
  result.manifest_digest = std::move(manifest_bytes.digest);
  result.signature_status = Signatures(result.manifest, options);
  Inventory(directory, result.manifest, options.limits);
  std::map<std::string, std::map<std::string, const BundleTensor*>> expected;
  for (const auto& model : result.manifest.models) {
    if (model.format == BundleModelFormat::onnx) ++result.integrity_only_models;
    for (const auto& tensor : model.tensors)
      expected[tensor.shard_file_id].emplace(tensor.name, &tensor);
  }
  for (const auto& file : result.manifest.files) {
    const auto path = directory.path() / file.relative_path;
    const Directory parent(path.parent_path());
    Input input(parent, path.filename());
    if (input.size() != file.artifact.size_bytes)
      Malformed("file.size", "the declared exact file size");
    auto content = Consume(input, options.limits.max_file_bytes,
                           file.role == BundleFileRole::weights);
    if (!(content.digest == file.artifact.digest))
      Malformed("file.digest", "the declared full-file SHA-256 digest");
    Add(result.verified_file_bytes, input.size(),
        options.limits.max_total_bytes, "files.bytes");
    if (file.role == BundleFileRole::weights) {
      const SafeTensorsView view(content.bytes, TensorLimits(options.limits));
      const auto& actual = view.directory().tensors;
      const auto& declared = expected[file.id];
      if (actual.size() != declared.size())
        Malformed("tensor.coverage",
                  "complete tensor inventory for each shard");
      for (const auto& tensor : actual) {
        const auto found = declared.find(tensor.name);
        if (found == declared.end() || found->second->dtype != tensor.dtype ||
            found->second->shape != tensor.shape)
          Malformed("tensor.mapping",
                    "the actual shard tensor name, dtype and shape");
      }
      result.verified_tensors += actual.size();
    }
  }
  // Check membership again after all reads. This detects ordinary concurrent
  // namespace edits, while the API still promises only a verification snapshot.
  Inventory(directory, result.manifest, options.limits);
  return result;
}

class Stage final {
 public:
  explicit Stage(const Directory& parent) : parent_(parent.path()) {
    static std::atomic<std::uint64_t> sequence{0U};
    // Exclusive mkdir, not the name entropy, provides uniqueness. No existing
    // directory is ever adopted or cleared. Cross-process collisions retry.
    for (unsigned attempt = 0U; attempt < 64U; ++attempt) {
      name_ = ".jyppx-bundle-stage-" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()) +
              "-" + std::to_string(++sequence);
#ifdef _WIN32
      if (CreateDirectoryW((parent_ / name_).c_str(), nullptr)) return;
      if (GetLastError() != ERROR_ALREADY_EXISTS) Io("stage.create");
#else
      if (::mkdirat(parent.get(), name_.c_str(), 0700) == 0) return;
      if (errno != EEXIST) Io("stage.create");
#endif
    }
    Resource("stage.collisions", 64U);
  }
  ~Stage() {
    if (!published_) {
      std::error_code ignored;
      std::filesystem::remove_all(path(), ignored);
    }
  }
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;
  std::filesystem::path path() const { return parent_ / name_; }
  void publish(const Directory& parent, const std::filesystem::path& filename) {
#ifdef _WIN32
    if (!MoveFileExW(path().c_str(), (parent.path() / filename).c_str(),
                     MOVEFILE_WRITE_THROUGH))
      Io("stage.atomic_no_replace");
#elif defined(__linux__)
    if (::syscall(SYS_renameat2, parent.get(), name_.c_str(), parent.get(),
                  filename.c_str(), 1U /* RENAME_NOREPLACE */) != 0)
      Io("stage.atomic_no_replace");
#else
    Unsupported("stage.atomic_no_replace",
                "Windows or Linux atomic publication");
#endif
    published_ = true;
    parent.sync();
  }

 private:
  std::filesystem::path parent_;
  std::string name_;
  bool published_ = false;
};

void MakeParents(const Directory& stage,
                 const std::filesystem::path& relative) {
  auto path = stage.path();
  for (const auto& part : relative.parent_path()) {
    const Directory parent(path);
    const auto child = path / part;
    std::error_code error;
    const auto status = std::filesystem::symlink_status(child, error);
    if (!std::filesystem::exists(status)) {
      if (error && error != std::errc::no_such_file_or_directory)
        Io("stage.parent.inspect");
      parent.create(part.string());
    }
    const Directory checked(child);
    path = child;
  }
}

}  // namespace

void validate_model_bundle_manifest(const ModelBundleManifest& manifest,
                                    ModelBundleLimits limits) {
  Boundary([&] { Validate(manifest, limits); });
}

std::string serialize_model_bundle_manifest(const ModelBundleManifest& manifest,
                                            ModelBundleLimits limits) {
  return Boundary([&] {
    Validate(manifest, limits);
    auto result = ManifestJson(manifest, true).dump();
    Count(result.size(), limits.max_manifest_bytes, "manifest.bytes");
    return result;
  });
}

ModelBundleManifest deserialize_model_bundle_manifest(
    std::string_view json, ModelBundleLimits limits) {
  return Boundary([&] { return ReadManifest(Parse(json, limits), limits); },
                  true);
}

std::string model_bundle_signing_payload(const ModelBundleManifest& manifest,
                                         ModelBundleLimits limits) {
  return Boundary([&] {
    Validate(manifest, limits);
    auto json = ManifestJson(manifest, false).dump();
    Count(json.size(), limits.max_manifest_bytes, "manifest.bytes");
    return std::string(kSigningPrefix) + json;
  });
}

ModelBundleVerification verify_model_bundle_directory(
    const std::filesystem::path& directory,
    const ModelBundleVerifyOptions& options) {
  return Boundary([&] {
    Limits(options.limits);
    const Directory root(directory);
    return Verify(root, options);
  });
}

ModelBundleVerification publish_model_bundle(
    const std::filesystem::path& destination,
    const ModelBundleManifest& manifest,
    std::span<const BundlePublishSource> sources,
    const ModelBundleVerifyOptions& options) {
  return Boundary([&] {
    const auto json = serialize_model_bundle_manifest(manifest, options.limits);
    static_cast<void>(Signatures(manifest, options));
    const auto target = Absolute(destination);
    if (target.filename().empty() || target == target.root_path())
      Invalid("destination", "a new named bundle directory");
    const Directory parent(target.parent_path());
    if (sources.size() != manifest.files.size())
      Invalid("sources", "exactly one local source per inventory file");
    std::map<std::string, std::filesystem::path> source_map;
    for (const auto& source : sources) {
      Token(source.file_id, options.limits);
      if (!source_map.emplace(source.file_id, Absolute(source.source_path))
               .second)
        Invalid("sources.duplicate", "unique local source identities");
    }
    Stage stage(parent);
    {
      const Directory root(stage.path());
      for (const auto& file : manifest.files) {
        const auto found = source_map.find(file.id);
        if (found == source_map.end())
          Invalid("sources.missing", "every declared file source");
        const Directory input_parent(found->second.parent_path());
        Input input(input_parent, found->second.filename());
        if (input.size() != file.artifact.size_bytes)
          Malformed("source.size", "the declared exact source size");
        const std::filesystem::path relative(file.relative_path);
        MakeParents(root, relative);
        const Directory output_parent((stage.path() / relative).parent_path());
        Output output(output_parent, relative.filename());
        const auto content =
            Consume(input, options.limits.max_file_bytes, false, &output);
        if (!(content.digest == file.artifact.digest))
          Malformed("source.digest", "the declared source SHA-256 digest");
        output.sync();
      }
      {
        Output output(root, std::string(kModelBundleManifest));
        output.write(Bytes(json));
        output.sync();
      }
    }
    ModelBundleVerification result;
    {
      const Directory root(stage.path());
      result = Verify(root, options);
      Inventory(root, manifest, options.limits, true);
    }
    stage.publish(parent, target.filename());
    return result;
  });
}

std::vector<EncodedBundleShard> shard_safetensors_for_bundle(
    std::span<const SafeTensorSource> tensors, std::uint64_t max_shard_bytes,
    std::string_view name_prefix, std::string_view device_intent,
    ModelBundleLimits limits) {
  return Boundary([&] {
    Limits(limits);
    Token(name_prefix, limits);
    Path(name_prefix, limits);
    if (name_prefix.size() > 64U)
      Invalid("shard.prefix", "at most 64 token bytes");
    if (max_shard_bytes < 16U || max_shard_bytes > limits.max_file_bytes)
      Invalid("shard.bytes", "a bound from 16 through max_file_bytes");
    if (device_intent != "cpu" && device_intent != "cuda" &&
        device_intent != "any")
      Invalid("tensor.device_intent", "cpu, cuda, or any");
    Count(tensors.size(), limits.max_tensors, "tensors.count");
    std::set<std::string> names;
    std::uint64_t total = 0U;
    for (const auto& tensor : tensors) {
      const BundleTensor metadata{.name = tensor.name,
                                  .dtype = tensor.dtype,
                                  .shape = tensor.shape,
                                  .device_intent = std::string(device_intent),
                                  .shard_file_id = "pending"};
      const auto size = TensorBytes(metadata, limits);
      if (size != tensor.data.size() || !names.insert(tensor.name).second)
        Invalid("tensor.source",
                "unique tensor names and exact shape/dtype bytes");
      if (tensor.byte_order != SafeTensorByteOrder::little)
        Unsupported("tensor.endianness", "explicit little-endian source bytes");
      Add(total, size, limits.max_total_bytes, "tensor.bytes");
    }
    std::vector<EncodedBundleShard> result;
    std::vector<SafeTensorSource> group;
    std::uint64_t payload = 0U;
    std::uint64_t header = 2U;  // JSON outer braces; each entry adds a comma.
    std::uint64_t encoded_total = 0U;
    const auto flush = [&] {
      Count(result.size() + 1U, limits.max_files, "shards.count");
      const auto bounded_header = header + (8U - header % 8U) % 8U;
      auto bounded_output = encoded_total;
      Add(bounded_output, 8U, limits.max_total_bytes, "shards.bytes");
      Add(bounded_output, bounded_header, limits.max_total_bytes,
          "shards.bytes");
      Add(bounded_output, payload, limits.max_total_bytes, "shards.bytes");
      auto tensor_limits = TensorLimits(limits);
      tensor_limits.max_file_bytes = max_shard_bytes;
      auto bytes = serialize_safetensors(group, {}, tensor_limits);
      Add(encoded_total, bytes.size(), limits.max_total_bytes, "shards.bytes");
      const auto id =
          std::string(name_prefix) + "." + std::to_string(result.size());
      const auto digest = Digest(bytes);
      const auto byte_count = bytes.size();
      EncodedBundleShard shard{
          .file = {.id = id,
                   .relative_path =
                       std::string(name_prefix) + "/" + id + ".safetensors",
                   .role = BundleFileRole::weights,
                   .artifact = {.digest = digest,
                                .size_bytes = byte_count,
                                .logical_name = id,
                                .media_type = "application/safetensors",
                                .producer =
                                    "interop.model_bundle.sharding.v1"}},
          .tensors = {},
          .bytes = std::move(bytes)};
      Path(shard.file.relative_path, limits);
      for (const auto& tensor : group)
        shard.tensors.push_back({.name = tensor.name,
                                 .dtype = tensor.dtype,
                                 .shape = tensor.shape,
                                 .device_intent = std::string(device_intent),
                                 .shard_file_id = id});
      result.push_back(std::move(shard));
      group.clear();
      payload = 0U;
      header = 2U;
    };
    const auto sorted_tensors = Sorted(tensors, SafeTensorName{});
    for (const auto* tensor : sorted_tensors) {
      // Offsets cannot exceed max_shard_bytes. Rendering those maximum offsets
      // bounds every real header entry without repeatedly encoding payloads.
      const Json one{{tensor->name,
                      {{"dtype", safetensors_dtype_name(tensor->dtype)},
                       {"shape", tensor->shape},
                       {"data_offsets",
                        Json::array({max_shard_bytes, max_shard_bytes})}}}};
      const auto entry = one.dump().size() - 1U;
      const auto fits = [&] {
        if (header > max_shard_bytes || entry > max_shard_bytes - header)
          return false;
        const auto next_header = header + entry;
        const auto padded = next_header + (8U - next_header % 8U) % 8U;
        if (padded > max_shard_bytes || max_shard_bytes - padded < 8U)
          return false;
        const auto remaining = max_shard_bytes - padded - 8U;
        return payload <= remaining &&
               tensor->data.size() <= remaining - payload;
      };
      if (!fits() && !group.empty()) flush();
      if (!fits()) Resource("shard.indivisible_tensor", max_shard_bytes);
      group.push_back(*tensor);
      payload += tensor->data.size();
      header += entry;
    }
    if (!group.empty() || tensors.empty()) flush();
    return result;
  });
}

}  // namespace jyppx::liuding::interop
