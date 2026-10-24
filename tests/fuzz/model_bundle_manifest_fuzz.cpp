// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/model_bundle.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace interop = jyppx::liuding::interop;
namespace fuzz = jyppx::liuding::fuzz;

interop::ModelBundleLimits limits() {
  return {.max_manifest_bytes = fuzz::kMaxInputBytes,
          .max_files = 32U,
          .max_models = 4U,
          .max_tensors = 64U,
          .max_rank = 8U,
          .max_dimension = 4096U,
          .max_tensor_elements = 65536U,
          .max_file_bytes = 1024U * 1024U,
          .max_total_bytes = 4U * 1024U * 1024U,
          .max_string_bytes = 512U,
          .max_path_bytes = 256U,
          .max_signatures = 4U,
          .max_signature_bytes = 128U};
}

void exercise(std::span<const std::byte> input) {
  const auto parsed =
      fuzz::exercise_json_codec(fuzz::text(input), limits(),
                                &interop::ModelBundleLimits::max_manifest_bytes,
                                interop::deserialize_model_bundle_manifest,
                                interop::serialize_model_bundle_manifest);
  if (!parsed) return;
  auto canonical_limits = limits();
  canonical_limits.max_manifest_bytes = fuzz::kCanonicalBytes;
  const auto canonical =
      interop::serialize_model_bundle_manifest(*parsed, canonical_limits);
  const auto reparsed =
      interop::deserialize_model_bundle_manifest(canonical, canonical_limits);
  fuzz::require(
      interop::model_bundle_signing_payload(*parsed, canonical_limits) ==
          interop::model_bundle_signing_payload(reparsed, canonical_limits),
      "model bundle canonicalization changed signing payload");
}

interop::ModelBundleManifest seed_manifest() {
  interop::ModelBundleManifest manifest;
  manifest.bundle_id = "fuzz-bundle";
  manifest.build = fuzz::seed_build();
  manifest.runtime = {
      .backend_id = "safetensors.adapter",
      .backend_version = "1.0.0",
      .environment_lock = {.hex = std::string(64U, 'b')},
      .dependencies = {{.name = "safetensors", .version = "0.6.2"}},
      .devices = {"cpu", "cuda"},
      .dtypes = {"float32"},
      .os_arch = {"linux-x86_64", "windows-x86_64"}};
  const auto descriptor = [&](std::string id, interop::BundleFileRole role) {
    manifest.files.push_back(
        {.id = id,
         .relative_path = "metadata/" + id + ".json",
         .role = role,
         .artifact = {.digest = {.hex = std::string(64U, 'a')},
                      .size_bytes = 2U,
                      .logical_name = id,
                      .media_type = "application/json",
                      .producer = "fuzz.fixture"}});
    return interop::BundleDescriptorRef{.schema_id = "fuzz." + id,
                                        .schema_version = 1U,
                                        .file_id = std::move(id)};
  };
  manifest.input_schema = descriptor("input", interop::BundleFileRole::schema);
  manifest.output_schema =
      descriptor("output", interop::BundleFileRole::schema);
  manifest.preprocessing = descriptor("pre", interop::BundleFileRole::prepost);
  manifest.postprocessing =
      descriptor("post", interop::BundleFileRole::prepost);
  manifest.labels = descriptor("labels", interop::BundleFileRole::labels);
  manifest.evaluation =
      descriptor("quality", interop::BundleFileRole::evaluation);
  manifest.sbom = descriptor("sbom", interop::BundleFileRole::sbom);
  manifest.provenance =
      descriptor("provenance", interop::BundleFileRole::provenance);
  manifest.weight_import_reports = {descriptor(
      "import-report", interop::BundleFileRole::weight_import_report)};
  const std::vector<std::byte> payload(16U, std::byte{0x2a});
  const interop::SafeTensorSource tensor{.name = "layer.weight",
                                         .dtype = interop::SafeTensorDtype::f32,
                                         .shape = {4U},
                                         .data = payload};
  auto shards =
      interop::shard_safetensors_for_bundle(std::span(&tensor, 1U), 4096U);
  interop::BundleModel model{.id = "primary"};
  for (auto& shard : shards) {
    model.file_ids.push_back(shard.file.id);
    model.tensors.insert(model.tensors.end(), shard.tensors.begin(),
                         shard.tensors.end());
    manifest.files.push_back(std::move(shard.file));
  }
  manifest.models.push_back(std::move(model));
  manifest.signatures = {{.algorithm = "ed25519",
                          .key_id = "fuzz-key",
                          .signature_hex = "0011aabb"}};
  return manifest;
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    auto manifest = seed_manifest();
    std::vector<std::vector<std::byte>> result;
    result.push_back(fuzz::copy_bytes(
        interop::serialize_model_bundle_manifest(manifest, limits())));
    manifest.labels.reset();
    std::erase_if(manifest.files, [](const auto& file) {
      return file.role == interop::BundleFileRole::labels;
    });
    manifest.signatures.clear();
    result.push_back(fuzz::copy_bytes(
        interop::serialize_model_bundle_manifest(manifest, limits())));
    for (const auto& seed : result) {
      static_cast<void>(interop::deserialize_model_bundle_manifest(
          fuzz::text(seed), limits()));
      exercise(seed);
    }
    return result;
  }();
  return value;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  return fuzz::exercise_raw_and_seed(data, size, seeds(), exercise);
}
