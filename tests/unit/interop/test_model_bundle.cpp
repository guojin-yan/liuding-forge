// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace interop = jyppx::liuding::interop;

std::span<const std::byte> Bytes(std::string_view text) {
  return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

jyppx::liuding::ArtifactDigest Digest(std::span<const std::byte> bytes) {
  return {.algorithm = jyppx::liuding::ArtifactDigestAlgorithm::sha256,
          .hex = jyppx::liuding::sha256_hex(bytes)};
}

void Write(const std::filesystem::path& path,
           std::span<const std::byte> bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("Cannot write model bundle test data");
}

std::string Read(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot read model bundle test data");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string Replace(std::string text, std::string_view from,
                    std::string_view to) {
  const auto found = text.find(from);
  if (found == std::string::npos)
    throw std::runtime_error("Missing model bundle test replacement");
  text.replace(found, from.size(), to);
  return text;
}

struct TensorFixture {
  std::vector<std::vector<std::byte>> payloads;
  std::vector<interop::SafeTensorSource> tensors;

  TensorFixture() {
    payloads.reserve(3U);
    tensors.reserve(3U);
    for (std::size_t index = 0U; index < 3U; ++index) {
      auto& payload = payloads.emplace_back(128U);
      for (std::size_t offset = 0U; offset < payload.size(); ++offset)
        payload[offset] = static_cast<std::byte>((index * 13U + offset) % 256U);
      tensors.push_back({.name = "layer." + std::to_string(index),
                         .dtype = interop::SafeTensorDtype::f32,
                         .shape = {32U},
                         .data = payload,
                         .byte_order = interop::SafeTensorByteOrder::little});
    }
  }
};

class ModelBundleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<std::uint64_t> sequence{0U};
    base_ = std::filesystem::temp_directory_path() /
            ("jyppx-bundle-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "-" + std::to_string(++sequence));
    // Unicode/spaces are supported for the trusted outer directory. Portable
    // lowercase ASCII is required only inside the untrusted bundle namespace.
    base_ /= std::filesystem::path(u8"bundle 测试");
    std::filesystem::create_directories(base_ / "sources");
    target_ = base_ / "published";
    manifest_.bundle_id = "unit-model-v1";
    manifest_.build = {.jyppx_version = "1.0.0",
                       .git_commit = "fixture-commit",
                       .compiler = "fixture-compiler",
                       .cxx_standard = "202002",
                       .os_arch = "fixture-x86_64",
                       .build_type = "Release",
                       .libtorch_version = "2.5.1",
                       .libtorch_build = "cpu",
                       .accelerator_runtime = "cpu",
                       .dependency_lock_digest = "unlocked"};
    manifest_.runtime = {
        .backend_id = "safetensors.adapter",
        .backend_version = "1.0.0",
        .environment_lock = Digest(Bytes("fixture-lock")),
        .dependencies = {{.name = "safetensors", .version = "0.6.2"}},
        .devices = {"cpu"},
        .dtypes = {"float32"},
        .os_arch = {"linux-x86_64", "windows-x86_64"},
        .dynamic_shapes = false};
    TensorFixture tensors;
    auto shards = interop::shard_safetensors_for_bundle(tensors.tensors, 384U);
    interop::BundleModel model{
        .id = "primary",
        .format = interop::BundleModelFormat::safetensors,
        .file_ids = {},
        .tensors = {}};
    for (auto& shard : shards) {
      model.file_ids.push_back(shard.file.id);
      model.tensors.insert(model.tensors.end(), shard.tensors.begin(),
                           shard.tensors.end());
      SaveSource(shard.file.id, shard.bytes);
      manifest_.files.push_back(std::move(shard.file));
    }
    manifest_.models.push_back(std::move(model));
    manifest_.input_schema =
        Descriptor("input", interop::BundleFileRole::schema);
    manifest_.output_schema =
        Descriptor("output", interop::BundleFileRole::schema);
    manifest_.preprocessing =
        Descriptor("preprocess", interop::BundleFileRole::prepost);
    manifest_.postprocessing =
        Descriptor("postprocess", interop::BundleFileRole::prepost);
    manifest_.evaluation =
        Descriptor("evaluation", interop::BundleFileRole::evaluation);
    manifest_.sbom = Descriptor("sbom", interop::BundleFileRole::sbom);
    manifest_.provenance =
        Descriptor("provenance", interop::BundleFileRole::provenance);
  }

  void TearDown() override {
    std::error_code ignored;
    std::filesystem::remove_all(base_.parent_path(), ignored);
  }

  void SaveSource(std::string_view id, std::span<const std::byte> bytes) {
    const auto path = base_ / "sources" / (std::string(id) + ".data");
    Write(path, bytes);
    for (auto& source : sources_) {
      if (source.file_id == id) {
        source.source_path = path;
        return;
      }
    }
    sources_.push_back({.file_id = std::string(id), .source_path = path});
  }

  interop::BundleDescriptorRef Descriptor(std::string id,
                                          interop::BundleFileRole role) {
    const auto json = "{\"schema_version\":1,\"descriptor\":\"" + id + "\"}";
    SaveSource(id, Bytes(json));
    manifest_.files.push_back({.id = id,
                               .relative_path = "metadata/" + id + ".json",
                               .role = role,
                               .artifact = {.digest = Digest(Bytes(json)),
                                            .size_bytes = json.size(),
                                            .logical_name = id,
                                            .media_type = "application/json",
                                            .producer = "unit.fixture"}});
    return {.schema_id = "fixture." + id,
            .schema_version = 1U,
            .file_id = std::move(id)};
  }

  interop::BundleFile& File(std::string_view id) {
    const auto found =
        std::find_if(manifest_.files.begin(), manifest_.files.end(),
                     [&](const auto& file) { return file.id == id; });
    if (found == manifest_.files.end())
      throw std::runtime_error("Unknown fixture file");
    return *found;
  }

  std::filesystem::path Source(std::string_view id) const {
    for (const auto& source : sources_) {
      if (source.file_id == id) return source.source_path;
    }
    throw std::runtime_error("Unknown fixture source");
  }

  void SourceContent(std::string_view id, std::string_view content) {
    SaveSource(id, Bytes(content));
    auto& file = File(id);
    file.artifact.digest = Digest(Bytes(content));
    file.artifact.size_bytes = content.size();
  }

  void SavePublishedManifest() const {
    Write(target_ / std::string(interop::kModelBundleManifest),
          Bytes(interop::serialize_model_bundle_manifest(manifest_)));
  }

  interop::ModelBundleVerification Publish(
      const interop::ModelBundleVerifyOptions& options = {}) {
    return interop::publish_model_bundle(target_, manifest_, sources_, options);
  }

  std::size_t Stages() const {
    std::size_t count = 0U;
    for (const auto& entry : std::filesystem::directory_iterator(base_)) {
      if (entry.path().filename().string().starts_with(".jyppx-bundle-stage-"))
        ++count;
    }
    return count;
  }

  std::filesystem::path base_;
  std::filesystem::path target_;
  interop::ModelBundleManifest manifest_;
  std::vector<interop::BundlePublishSource> sources_;
};

TEST_F(ModelBundleTest,
       CanonicalCodecPreservesEveryDescriptorAndPinnedRuntime) {
  manifest_.labels = Descriptor("labels", interop::BundleFileRole::labels);
  manifest_.weight_import_reports.push_back(
      Descriptor("import", interop::BundleFileRole::weight_import_report));
  const auto encoded = interop::serialize_model_bundle_manifest(manifest_);
  std::reverse(manifest_.files.begin(), manifest_.files.end());
  std::reverse(manifest_.models[0U].file_ids.begin(),
               manifest_.models[0U].file_ids.end());
  std::reverse(manifest_.models[0U].tensors.begin(),
               manifest_.models[0U].tensors.end());
  std::reverse(manifest_.runtime.os_arch.begin(),
               manifest_.runtime.os_arch.end());
  EXPECT_EQ(encoded, interop::serialize_model_bundle_manifest(manifest_));
  const auto decoded = interop::deserialize_model_bundle_manifest(encoded);
  ASSERT_TRUE(decoded.labels.has_value());
  EXPECT_EQ(decoded.labels->schema_id, "fixture.labels");
  ASSERT_EQ(decoded.weight_import_reports.size(), 1U);
  EXPECT_EQ(decoded.weight_import_reports[0U].file_id, "import");
  EXPECT_EQ(decoded.runtime.environment_lock,
            manifest_.runtime.environment_lock);
  EXPECT_EQ(decoded.build.git_commit, "fixture-commit");
  EXPECT_EQ(interop::serialize_model_bundle_manifest(decoded), encoded);
}

TEST_F(ModelBundleTest,
       RejectsDuplicateDecodedKeysUnknownFieldsAndUnsupportedVersions) {
  const auto encoded = interop::serialize_model_bundle_manifest(manifest_);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   "{\"bundle_\\u0069d\":\"duplicate\"," + encoded.substr(1U)),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   "{\"critical_future_field\":true," + encoded.substr(1U)),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   Replace(encoded, "\"schema_version\":1,\"signatures\"",
                           "\"schema_version\":2,\"signatures\"")),
               jyppx::liuding::CompatibilityError);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   Replace(encoded, "\"schema_version\":1,\"signatures\"",
                           "\"schema_version\":0,\"signatures\"")),
               jyppx::liuding::CompatibilityError);
  EXPECT_THROW(
      interop::deserialize_model_bundle_manifest(Replace(
          encoded, "\"compression\":\"none\"", "\"compression\":\"gzip\"")),
      jyppx::liuding::CompatibilityError);
}

TEST_F(ModelBundleTest, RejectsMalformedIntegersUtf8DepthAndDigestSyntax) {
  const auto encoded = interop::serialize_model_bundle_manifest(manifest_);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   Replace(encoded, "\"schema_version\":1,\"signatures\"",
                           "\"schema_version\":1.0,\"signatures\"")),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   Replace(encoded, "\"schema_version\":1,\"signatures\"",
                           "\"schema_version\":-1,\"signatures\"")),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   std::string("{\"x\":\"") + char(0xc0) + char(0x80) + "\"}"),
               jyppx::liuding::SerializationError);
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(
                   "{\"x\":" + std::string(17U, '[') + "0" +
                   std::string(17U, ']') + "}"),
               jyppx::liuding::ResourceExhausted);
  manifest_.files[0U].artifact.digest.hex[0U] = 'G';
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
               jyppx::liuding::InvalidArgument);
}

TEST_F(ModelBundleTest,
       EnforcesIndependentCountsBytesDimensionsAndStringBounds) {
  const auto encoded = interop::serialize_model_bundle_manifest(manifest_);
  interop::ModelBundleLimits limits;
  limits.max_manifest_bytes = encoded.size() - 1U;
  EXPECT_THROW(interop::deserialize_model_bundle_manifest(encoded, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_files = manifest_.files.size() - 1U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_tensors = 2U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_dimension = 31U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_file_bytes = 1U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_total_bytes = 1U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::ResourceExhausted);
  limits = {};
  limits.max_rank = 0U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::InvalidArgument);
  limits = {};
  limits.max_string_bytes = 4U;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_, limits),
               jyppx::liuding::ResourceExhausted);
}

TEST_F(ModelBundleTest, RejectsTraversalDeviceNamesCaseAliasesAndFilePrefixes) {
  const auto original = File("input").relative_path;
  for (const auto path :
       {"../escape", "/absolute", "a/../escape", "a//b", "a/./b", "a\\b",
        "c:stream", "Input.json", "a/con.json", "a/aux", "a/lpt1.txt", "a/COM1",
        "a/file.", "manifest.json", "manifest.json/child"}) {
    File("input").relative_path = path;
    EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
                 jyppx::liuding::InvalidArgument)
        << path;
  }
  File("input").relative_path = original;
  File("output").relative_path = original;
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
               jyppx::liuding::InvalidArgument);
  File("output").relative_path = original + "/child";
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
               jyppx::liuding::InvalidArgument);
}

TEST_F(ModelBundleTest, RejectsScriptLibraryPickleAndUnknownRoles) {
  for (const auto path :
       {"metadata/a.py", "metadata/a.dll", "metadata/a.so.1", "metadata/a.ps1",
        "metadata/a.pkl", "metadata/a.pt"}) {
    File("input").relative_path = path;
    EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
                 jyppx::liuding::CompatibilityError)
        << path;
  }
  File("input").relative_path = "metadata/input.json";
  File("input").role = static_cast<interop::BundleFileRole>(255U);
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
               jyppx::liuding::CompatibilityError);
}

TEST_F(ModelBundleTest, RequiresCompleteUniqueFileOwnershipAndTensorMappings) {
  auto changed = manifest_;
  changed.input_schema.file_id = "absent";
  EXPECT_THROW(interop::validate_model_bundle_manifest(changed),
               jyppx::liuding::InvalidArgument);
  changed = manifest_;
  changed.output_schema.file_id = changed.input_schema.file_id;
  EXPECT_THROW(interop::validate_model_bundle_manifest(changed),
               jyppx::liuding::InvalidArgument);
  changed = manifest_;
  changed.files.push_back({.id = "unreferenced",
                           .relative_path = "metadata/orphan.json",
                           .role = interop::BundleFileRole::schema,
                           .artifact = changed.files.back().artifact});
  EXPECT_THROW(interop::validate_model_bundle_manifest(changed),
               jyppx::liuding::InvalidArgument);
  changed = manifest_;
  changed.models[0U].tensors[0U].shard_file_id = "absent";
  EXPECT_THROW(interop::validate_model_bundle_manifest(changed),
               jyppx::liuding::InvalidArgument);
  changed = manifest_;
  changed.models[0U].tensors[1U].name = changed.models[0U].tensors[0U].name;
  EXPECT_THROW(interop::validate_model_bundle_manifest(changed),
               jyppx::liuding::InvalidArgument);
  changed = manifest_;
  changed.runtime.dependencies[0U].version = ">=0.6";
  EXPECT_THROW(interop::validate_model_bundle_manifest(changed),
               jyppx::liuding::InvalidArgument);
}

TEST_F(ModelBundleTest, PublishesVerifiesAndHashesEveryRealFileAndTensor) {
  const auto published = Publish();
  const auto verified = interop::verify_model_bundle_directory(target_);
  EXPECT_EQ(published.manifest_digest, verified.manifest_digest);
  EXPECT_EQ(verified.manifest_digest,
            Digest(Bytes(
                Read(target_ / std::string(interop::kModelBundleManifest)))));
  EXPECT_EQ(verified.signature_status,
            interop::BundleSignatureStatus::unsigned_bundle);
  EXPECT_EQ(verified.verified_tensors, 3U);
  std::uint64_t bytes = 0U;
  for (const auto& file : manifest_.files) {
    bytes += file.artifact.size_bytes;
    EXPECT_EQ(Digest(Bytes(Read(target_ / file.relative_path))),
              file.artifact.digest);
  }
  EXPECT_EQ(verified.verified_file_bytes, bytes);
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest, RejectsCorruptedFileEvenWhenSizeIsUnchanged) {
  Publish();
  const auto path = target_ / File("input").relative_path;
  auto content = Read(path);
  content[0U] = '[';
  Write(path, Bytes(content));
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
}

TEST_F(ModelBundleTest, RejectsMissingExtraFilesAndUndeclaredEmptyDirectories) {
  Publish();
  const auto file = target_ / File("input").relative_path;
  const auto content = Read(file);
  std::filesystem::remove(file);
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
  Write(file, Bytes(content));
  Write(target_ / "unlisted.json", Bytes("{}"));
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
  std::filesystem::remove(target_ / "unlisted.json");
  std::filesystem::create_directory(target_ / "unlisted");
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
}

TEST_F(ModelBundleTest, RejectsDeclaredSizeMismatchBeforeReadingPayload) {
  Publish();
  ++File("input").artifact.size_bytes;
  SavePublishedManifest();
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
}

TEST_F(ModelBundleTest, RejectsOmittedOrWrongActualTensorMetadata) {
  Publish();
  const auto complete = manifest_;
  manifest_.models[0U].tensors.pop_back();
  SavePublishedManifest();
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
  manifest_ = complete;
  const auto original = manifest_.models[0U].tensors[0U].shape;
  manifest_.models[0U].tensors[0U].shape = {16U, 2U};
  SavePublishedManifest();
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
  manifest_.models[0U].tensors[0U].shape = original;
}

TEST_F(ModelBundleTest, RejectsTensorAssignedToTheWrongDeclaredShard) {
  Publish();
  ASSERT_GE(manifest_.models[0U].file_ids.size(), 2U);
  auto& tensors = manifest_.models[0U].tensors;
  std::swap(tensors[0U].shard_file_id, tensors[1U].shard_file_id);
  SavePublishedManifest();
  EXPECT_THROW(interop::verify_model_bundle_directory(target_),
               jyppx::liuding::SerializationError);
}

TEST_F(ModelBundleTest,
       FailedPublicationCleansStageAndPreservesExistingBundle) {
  const auto previous = Publish();
  const auto previous_json =
      Read(target_ / std::string(interop::kModelBundleManifest));
  manifest_.bundle_id = "new-generation";
  EXPECT_THROW(Publish(), jyppx::liuding::IoError);
  EXPECT_EQ(Read(target_ / std::string(interop::kModelBundleManifest)),
            previous_json);
  EXPECT_EQ(interop::verify_model_bundle_directory(target_).manifest_digest,
            previous.manifest_digest);
  EXPECT_EQ(Stages(), 0U);
  target_ = base_ / "failed-new-generation";
  const auto source_content = Read(Source("input"));
  Write(Source("input"), Bytes("broken-source"));
  EXPECT_THROW(Publish(), jyppx::liuding::SerializationError);
  EXPECT_FALSE(std::filesystem::exists(target_));
  EXPECT_EQ(Stages(), 0U);
  auto same_size_corruption = source_content;
  same_size_corruption[0U] = '[';
  Write(Source("input"), Bytes(same_size_corruption));
  EXPECT_THROW(Publish(), jyppx::liuding::SerializationError);
  EXPECT_FALSE(std::filesystem::exists(target_));
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest, RejectsMalformedShardEvenWithMatchingSizeAndDigest) {
  const auto id = manifest_.models[0U].file_ids[0U];
  auto content = Read(Source(id));
  ASSERT_GT(content.size(), 8U);
  content[8U] = '[';
  SourceContent(id, content);
  EXPECT_THROW(Publish(), jyppx::liuding::SerializationError);
  EXPECT_FALSE(std::filesystem::exists(target_));
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest, RejectsExecutableMagicDespiteCorrectDeclaredDigest) {
  SourceContent("input", "#!/bin/sh\necho must-never-execute\n");
  EXPECT_THROW(Publish(), jyppx::liuding::CompatibilityError);
  EXPECT_FALSE(std::filesystem::exists(target_));
  EXPECT_EQ(Stages(), 0U);
  SourceContent("input", std::string("\x7f") + "ELFbinary");
  EXPECT_THROW(Publish(), jyppx::liuding::CompatibilityError);
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest,
       RejectsDuplicateMissingAndUndeclaredPublicationSources) {
  auto sources = sources_;
  sources.back() = sources.front();
  EXPECT_THROW(interop::publish_model_bundle(target_, manifest_, sources),
               jyppx::liuding::InvalidArgument);
  sources = sources_;
  sources.back().file_id = "undeclared";
  EXPECT_THROW(interop::publish_model_bundle(target_, manifest_, sources),
               jyppx::liuding::InvalidArgument);
  sources.pop_back();
  EXPECT_THROW(interop::publish_model_bundle(target_, manifest_, sources),
               jyppx::liuding::InvalidArgument);
  EXPECT_FALSE(std::filesystem::exists(target_));
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest, RejectsFileSymlinksWithoutFollowingThem) {
  Publish();
  const auto leaf = target_ / File("input").relative_path;
  std::filesystem::remove(leaf);
  std::error_code error;
  std::filesystem::create_symlink(Source("input"), leaf, error);
  if (error)
    GTEST_SKIP() << "File symlink creation is unavailable on this host";
  EXPECT_THROW(interop::verify_model_bundle_directory(target_), jyppx::liuding::IoError);
}

TEST_F(ModelBundleTest, RejectsDirectorySymlinksInBundleOrSourceAncestry) {
  Publish();
  std::error_code error;
  std::filesystem::rename(target_ / "metadata", base_ / "external-metadata",
                          error);
  ASSERT_FALSE(error);
  std::filesystem::create_directory_symlink(base_ / "external-metadata",
                                            target_ / "metadata", error);
  if (error)
    GTEST_SKIP() << "Directory symlink creation is unavailable on this host";
  EXPECT_THROW(interop::verify_model_bundle_directory(target_), jyppx::liuding::IoError);
  const auto source_alias = base_ / "source-alias";
  std::filesystem::create_directory_symlink(base_ / "sources", source_alias,
                                            error);
  ASSERT_FALSE(error);
  auto sources = sources_;
  sources[0U].source_path = source_alias / sources[0U].source_path.filename();
  EXPECT_THROW(
      interop::publish_model_bundle(base_ / "via-alias", manifest_, sources),
      jyppx::liuding::IoError);
  EXPECT_FALSE(std::filesystem::exists(base_ / "via-alias"));
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest, RejectsHardLinkAliasesOfDeclaredFiles) {
  Publish();
  const auto leaf = target_ / File("input").relative_path;
  std::filesystem::remove(leaf);
  std::error_code error;
  std::filesystem::create_hard_link(Source("input"), leaf, error);
  if (error) GTEST_SKIP() << "Hard links are unavailable on this filesystem";
  EXPECT_THROW(interop::verify_model_bundle_directory(target_), jyppx::liuding::IoError);
}

TEST_F(ModelBundleTest, UnsignedAndUnverifiedSignaturesNeverBecomeTrusted) {
  interop::ModelBundleVerifyOptions required;
  required.require_trusted_signature = true;
  EXPECT_THROW(Publish(required), jyppx::liuding::CompatibilityError);
  manifest_.signatures.push_back({.algorithm = "fixture-test-only",
                                  .key_id = "untrusted-key",
                                  .signature_hex = "01020304"});
  EXPECT_THROW(Publish(required), jyppx::liuding::CompatibilityError);
  const auto published = Publish();
  EXPECT_EQ(published.signature_status,
            interop::BundleSignatureStatus::present_unverified);
  EXPECT_EQ(interop::verify_model_bundle_directory(target_).signature_status,
            interop::BundleSignatureStatus::present_unverified);
}

TEST_F(ModelBundleTest,
       TrustCallbackReceivesActualDomainSeparatedCanonicalPayload) {
  // This callback tests payload transport and trust policy only. It is NOT a
  // signature algorithm and never represents production cryptographic evidence.
  const auto payload = interop::model_bundle_signing_payload(manifest_);
  ASSERT_TRUE(payload.starts_with("JYPPX-MODEL-BUNDLE-SIGNATURE-V1\n{"));
  EXPECT_EQ(payload.find("\"signatures\""), std::string::npos);
  manifest_.signatures.push_back({.algorithm = "fixture-test-only",
                                  .key_id = "fixture-key",
                                  .signature_hex = "01020304"});
  EXPECT_EQ(interop::model_bundle_signing_payload(manifest_), payload);
  interop::ModelBundleVerifyOptions options;
  options.require_trusted_signature = true;
  std::size_t calls = 0U;
  options.signature_verifier = [&](const auto& signature,
                                   std::span<const std::byte> bytes) {
    ++calls;
    return signature.key_id == "fixture-key" &&
           std::equal(bytes.begin(), bytes.end(), Bytes(payload).begin(),
                      Bytes(payload).end());
  };
  EXPECT_EQ(Publish(options).signature_status,
            interop::BundleSignatureStatus::trusted);
  EXPECT_GE(calls, 1U);
  manifest_.bundle_id = "tampered-metadata";
  SavePublishedManifest();
  EXPECT_THROW(interop::verify_model_bundle_directory(target_, options),
               jyppx::liuding::SerializationError);
}

TEST_F(ModelBundleTest, CallbackFailureIsRedactedAndCannotPublish) {
  manifest_.signatures.push_back({.algorithm = "fixture-test-only",
                                  .key_id = "fixture-key",
                                  .signature_hex = "01"});
  interop::ModelBundleVerifyOptions options;
  options.signature_verifier = [](const auto&, auto) -> bool {
    throw std::runtime_error("private-key-service-secret");
  };
  try {
    Publish(options);
    FAIL() << "Expected callback failure";
  } catch (const jyppx::liuding::BackendError& error) {
    EXPECT_EQ(std::string(error.what()).find("private-key-service-secret"),
              std::string::npos);
  }
  EXPECT_FALSE(std::filesystem::exists(target_));
  EXPECT_EQ(Stages(), 0U);
}

TEST_F(ModelBundleTest, EveryDeclaredSignatureMustPassTrustPolicy) {
  manifest_.signatures = {{.algorithm = "fixture-test-only",
                           .key_id = "accepted",
                           .signature_hex = "01"},
                          {.algorithm = "fixture-test-only",
                           .key_id = "rejected",
                           .signature_hex = "02"}};
  interop::ModelBundleVerifyOptions options;
  options.signature_verifier = [](const auto& signature, auto) {
    return signature.key_id == "accepted";
  };
  EXPECT_THROW(Publish(options), jyppx::liuding::SerializationError);
  EXPECT_FALSE(std::filesystem::exists(target_));
}

TEST_F(ModelBundleTest, OnnxReferencesHaveExplicitIntegrityOnlyScope) {
  const auto old_files = manifest_.models[0U].file_ids;
  std::erase_if(manifest_.files, [&](const auto& file) {
    return std::find(old_files.begin(), old_files.end(), file.id) !=
           old_files.end();
  });
  std::erase_if(sources_, [&](const auto& source) {
    return std::find(old_files.begin(), old_files.end(), source.file_id) !=
           old_files.end();
  });
  // Deliberately opaque fixture, never described as a valid ONNX protobuf.
  const std::string opaque = "official-runtime-checking-is-another-contract";
  SaveSource("model", Bytes(opaque));
  manifest_.files.push_back({.id = "model",
                             .relative_path = "model.onnx",
                             .role = interop::BundleFileRole::model,
                             .artifact = {.digest = Digest(Bytes(opaque)),
                                          .size_bytes = opaque.size(),
                                          .logical_name = "opaque-fixture",
                                          .media_type = "application/onnx",
                                          .producer = "unit.fixture"}});
  manifest_.models = {{.id = "primary",
                       .format = interop::BundleModelFormat::onnx,
                       .file_ids = {"model"},
                       .tensors = {}}};
  const auto result = Publish();
  EXPECT_EQ(result.verified_tensors, 0U);
  EXPECT_EQ(result.integrity_only_models, 1U);
  EXPECT_EQ(result.manifest.models[0U].format,
            interop::BundleModelFormat::onnx);
  manifest_.models[0U].file_ids.push_back("external-data");
  EXPECT_THROW(interop::validate_model_bundle_manifest(manifest_),
               jyppx::liuding::CompatibilityError);
}

TEST(ModelBundleSharding, IsDeterministicBoundedAndCoversAllTensors) {
  TensorFixture tensors;
  const auto original =
      interop::shard_safetensors_for_bundle(tensors.tensors, 384U);
  std::reverse(tensors.tensors.begin(), tensors.tensors.end());
  const auto reversed =
      interop::shard_safetensors_for_bundle(tensors.tensors, 384U);
  ASSERT_EQ(original.size(), 3U);
  ASSERT_EQ(reversed.size(), original.size());
  std::set<std::string> names;
  for (std::size_t index = 0U; index < original.size(); ++index) {
    const auto& shard = original[index];
    EXPECT_LE(shard.bytes.size(), 384U);
    EXPECT_EQ(shard.file.artifact.size_bytes, shard.bytes.size());
    EXPECT_EQ(shard.file.artifact.digest, Digest(shard.bytes));
    EXPECT_EQ(shard.bytes, reversed[index].bytes);
    EXPECT_EQ(shard.file.relative_path, reversed[index].file.relative_path);
    const interop::SafeTensorsView view(shard.bytes);
    ASSERT_EQ(view.directory().tensors.size(), shard.tensors.size());
    for (const auto& tensor : shard.tensors) {
      EXPECT_TRUE(names.insert(tensor.name).second);
      EXPECT_EQ(tensor.shard_file_id, shard.file.id);
      const auto found = std::find_if(
          tensors.tensors.begin(), tensors.tensors.end(),
          [&](const auto& source) { return source.name == tensor.name; });
      ASSERT_NE(found, tensors.tensors.end());
      const auto bytes = view.tensor_bytes(tensor.name);
      EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), found->data.begin(),
                             found->data.end()));
    }
  }
  EXPECT_EQ(names.size(), tensors.tensors.size());
}

TEST(ModelBundleSharding,
     RejectsIndivisibleDuplicateWrongByteOrderAndOversizeInputs) {
  TensorFixture tensors;
  EXPECT_THROW(interop::shard_safetensors_for_bundle(tensors.tensors, 16U),
               jyppx::liuding::ResourceExhausted);
  tensors.tensors[1U].name = tensors.tensors[0U].name;
  EXPECT_THROW(interop::shard_safetensors_for_bundle(tensors.tensors, 384U),
               jyppx::liuding::InvalidArgument);
  tensors.tensors[1U].name = "layer.1";
  tensors.tensors[0U].byte_order = interop::SafeTensorByteOrder::big;
  EXPECT_THROW(interop::shard_safetensors_for_bundle(tensors.tensors, 384U),
               jyppx::liuding::CompatibilityError);
  tensors.tensors[0U].byte_order = interop::SafeTensorByteOrder::little;
  tensors.tensors[0U].shape = {33U};
  EXPECT_THROW(interop::shard_safetensors_for_bundle(tensors.tensors, 384U),
               jyppx::liuding::InvalidArgument);
  interop::ModelBundleLimits limits;
  limits.max_file_bytes = 256U;
  EXPECT_THROW(interop::shard_safetensors_for_bundle(tensors.tensors, 384U,
                                                     "weights", "cpu", limits),
               jyppx::liuding::InvalidArgument);
}

TEST(ModelBundleSharding, HandlesEmptyTensorSetZeroDimensionsAndScalars) {
  const auto empty = interop::shard_safetensors_for_bundle({}, 16U);
  ASSERT_EQ(empty.size(), 1U);
  EXPECT_EQ(empty[0U].bytes.size(), 16U);
  EXPECT_TRUE(
      interop::SafeTensorsView(empty[0U].bytes).directory().tensors.empty());
  const std::array<std::byte, 4U> scalar{std::byte{0U}, std::byte{0U},
                                         std::byte{128U}, std::byte{63U}};
  const std::array<interop::SafeTensorSource, 2U> tensors{
      {{.name = "scalar",
        .dtype = interop::SafeTensorDtype::f32,
        .shape = {},
        .data = scalar},
       {.name = "empty",
        .dtype = interop::SafeTensorDtype::f32,
        .shape = {3U, 0U},
        .data = {}}}};
  const auto shards =
      interop::shard_safetensors_for_bundle(tensors, 512U, "weights", "any");
  ASSERT_EQ(shards.size(), 1U);
  const interop::SafeTensorsView view(shards[0U].bytes);
  EXPECT_EQ(view.tensor_bytes("scalar").size(), 4U);
  EXPECT_TRUE(view.tensor_bytes("empty").empty());
  EXPECT_EQ(shards[0U].tensors[0U].device_intent, "any");
}

}  // namespace
