// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/deployment/bundle_signing/provider.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace signing = jyppx::liuding::bundle_signing;
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
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("Cannot write example data");
}

/** @brief Create a tiny data-only fixture; no accuracy or runtime
 * qualification. */
interop::ModelBundleVerification UnsignedBundle(
    const std::filesystem::path& work) {
  interop::ModelBundleManifest manifest;
  manifest.bundle_id = "ephemeral-signing-example";
  manifest.build = jyppx::liuding::current_build_fingerprint();
  auto runtime_os_arch = manifest.build.os_arch;
  std::transform(runtime_os_arch.begin(), runtime_os_arch.end(),
                 runtime_os_arch.begin(), [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  manifest.runtime = {
      .backend_id = "safetensors.adapter",
      .backend_version = "1.0.0",
      .environment_lock = Digest(Bytes("ephemeral-example-lock")),
      .dependencies = {{.name = "safetensors", .version = "0.6.2"}},
      .devices = {"cpu"},
      .dtypes = {"uint8"},
      .os_arch = {runtime_os_arch},
      .dynamic_shapes = false};
  std::vector<interop::BundlePublishSource> sources;
  const auto save = [&](const std::string& id,
                        std::span<const std::byte> bytes) {
    const auto path = work / "inputs" / (id + ".data");
    Write(path, bytes);
    sources.push_back({.file_id = id, .source_path = path});
  };
  const std::array<std::byte, 4U> data{std::byte{1U}, std::byte{2U},
                                       std::byte{3U}, std::byte{4U}};
  const std::array<interop::SafeTensorSource, 1U> tensors{
      interop::SafeTensorSource{
          .name = "example.values",
          .dtype = interop::SafeTensorDtype::u8,
          .shape = {4U},
          .data = data,
          .byte_order = interop::SafeTensorByteOrder::little}};
  auto shards = interop::shard_safetensors_for_bundle(tensors, 4096U);
  interop::BundleModel model{.id = "primary",
                             .format = interop::BundleModelFormat::safetensors,
                             .file_ids = {},
                             .tensors = {}};
  for (auto& shard : shards) {
    save(shard.file.id, shard.bytes);
    model.file_ids.push_back(shard.file.id);
    model.tensors.insert(model.tensors.end(), shard.tensors.begin(),
                         shard.tensors.end());
    manifest.files.push_back(std::move(shard.file));
  }
  manifest.models.push_back(std::move(model));
  const auto descriptor = [&](std::string id, interop::BundleFileRole role) {
    const auto text = "{\"example_only\":true,\"descriptor\":\"" + id + "\"}";
    save(id, Bytes(text));
    manifest.files.push_back(
        {.id = id,
         .relative_path = "metadata/" + id + ".json",
         .role = role,
         .artifact = {.digest = Digest(Bytes(text)),
                      .size_bytes = text.size(),
                      .logical_name = id,
                      .media_type = "application/json",
                      .producer = "example.ephemeral_signing"}});
    return interop::BundleDescriptorRef{.schema_id = "example." + id,
                                        .schema_version = 1U,
                                        .file_id = std::move(id)};
  };
  manifest.input_schema = descriptor("input", interop::BundleFileRole::schema);
  manifest.output_schema =
      descriptor("output", interop::BundleFileRole::schema);
  manifest.preprocessing =
      descriptor("preprocess", interop::BundleFileRole::prepost);
  manifest.postprocessing =
      descriptor("postprocess", interop::BundleFileRole::prepost);
  manifest.evaluation =
      descriptor("evaluation", interop::BundleFileRole::evaluation);
  manifest.sbom = descriptor("sbom", interop::BundleFileRole::sbom);
  manifest.provenance =
      descriptor("provenance", interop::BundleFileRole::provenance);
  return interop::publish_model_bundle(work / "unsigned", manifest, sources);
}

/** @brief Deletes only this example's freshly and exclusively created temp dir.
 */
struct WorkDirectory {
  std::filesystem::path path;
  bool temporary;
  ~WorkDirectory() {
    if (temporary) {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  }
};

int Run(const std::filesystem::path& work) {
  // The private key remains inside EVP; it is never written into the bundle.
  const auto signer = signing::Ed25519Signer::generate();
  const auto policy_json =
      signing::serialize_ed25519_trust_policy(signer.public_key());
  const auto expected_policy = Digest(Bytes(policy_json));
  Write(work / "public-trust-policy.json", Bytes(policy_json));

  // Here the application owns the ephemeral key and authorizes it locally for
  // this lesson. Production consumers obtain this expected digest independently
  // from their trusted configuration, never from the received bundle or policy.
  const auto policy = signing::Ed25519TrustPolicy::load(
      work / "public-trust-policy.json", expected_policy);
  const auto source = UnsignedBundle(work);
  const auto publication = signing::sign_model_bundle_directory(
      work / "unsigned", source.manifest_digest, work / "signed", signer,
      policy);
  const auto trusted_options = signing::trusted_ed25519_options(policy);
  const auto verified =
      interop::verify_model_bundle_directory(work / "signed", trusted_options);
  if (verified.signature_status != interop::BundleSignatureStatus::trusted)
    throw std::runtime_error("Expected authenticated signed example");

  // An attacker can re-encode a manifest and keep every ordinary file hash
  // consistent, but cannot sign its changed product identity without the key.
  auto forged = publication.published.manifest;
  forged.bundle_id = "changed-without-private-key";
  std::vector<interop::BundlePublishSource> original_files;
  for (const auto& file : forged.files)
    original_files.push_back(
        {.file_id = file.id,
         .source_path = work / "signed" / file.relative_path});
  const auto integrity_only =
      interop::publish_model_bundle(work / "tampered", forged, original_files);
  if (integrity_only.signature_status !=
      interop::BundleSignatureStatus::present_unverified)
    throw std::runtime_error("Expected integrity-only tampered fixture");
  bool rejected = false;
  try {
    static_cast<void>(interop::verify_model_bundle_directory(work / "tampered",
                                                             trusted_options));
  } catch (const jyppx::liuding::Error&) {
    rejected = true;
  }
  if (!rejected) throw std::runtime_error("Tampered metadata was not rejected");
  std::cout << "signature_status=trusted\n"
            << "tampered_metadata=rejected\n"
            << "private_key=ephemeral_memory_only\n"
            << "key_id=" << policy.key_id() << '\n'
            << "policy_sha256=" << policy.digest().hex << '\n'
            << "signed_manifest_sha256=" << verified.manifest_digest.hex
            << '\n';
  return 0;
}

}  // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  try {
    if (argc > 2) {
      std::cerr << "Usage: liuding-forge-example-bundle-signing "
                   "[new-output-directory]\n";
      return 2;
    }
    const bool temporary = argc == 1;
    const auto path =
        temporary ? std::filesystem::temp_directory_path() /
                        ("liuding-bundle-signing-" +
                         std::to_string(std::chrono::steady_clock::now()
                                            .time_since_epoch()
                                            .count()))
                  : std::filesystem::absolute(std::filesystem::path(argv[1]));
    // Exclusive create; no existing directory is adopted, cleared or
    // overwritten.
    if (!std::filesystem::create_directory(path))
      throw std::runtime_error("Example output directory must be new");
    const WorkDirectory work{path, temporary};
    return Run(work.path);
  } catch (const std::exception&) {
    // This lesson never prints private input or backend diagnostic text.
    std::cerr << "Bundle signing example failed; check explicit inputs and "
                 "crypto SDK.\n";
    return 1;
  }
}
