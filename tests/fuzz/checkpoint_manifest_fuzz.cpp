// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"

#include <jyppx/liuding/interop/checkpoint_manifest.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace {
namespace interop = jyppx::liuding::interop;
namespace fuzz = jyppx::liuding::fuzz;

interop::CheckpointManifestLimits limits() {
  return {.max_manifest_bytes = fuzz::kMaxInputBytes,
          .max_depth = 16U,
          .max_nodes = 8192U,
          .max_string_bytes = 512U,
          .max_total_string_bytes = 32U * 1024U,
          .max_participants = 16U,
          .max_blobs = 64U,
          .max_path_bytes = 256U,
          .max_total_blob_bytes = 1024U * 1024U};
}

void exercise(std::span<const std::byte> input) {
  static_cast<void>(fuzz::exercise_json_codec(
      fuzz::text(input), limits(),
      &interop::CheckpointManifestLimits::max_manifest_bytes,
      interop::parse_checkpoint_artifact_manifest_json,
      interop::checkpoint_artifact_manifest_json));
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    interop::CheckpointArtifactManifest manifest{
        .build = fuzz::seed_build(),
        .participants = {
            {.state = {.participant_id = "fuzz.participant",
                       .schema_version = 1U,
                       .restore_level = jyppx::liuding::RestoreLevel::exact,
                       .fields = {{.name = "state",
                                   .media_type = "application/octet-stream",
                                   .max_bytes = 64U},
                                  {.name = "optional",
                                   .media_type = "application/octet-stream",
                                   .max_bytes = 32U,
                                   .required = false}}},
             .blobs = {{.field_name = "state",
                        .relative_path = "blobs/state.bin",
                        .size_bytes = 8U,
                        .digest = {.hex = std::string(64U, 'a')}}}}}};
    std::vector<std::vector<std::byte>> result;
    for (auto level : {jyppx::liuding::RestoreLevel::exact,
                       jyppx::liuding::RestoreLevel::bounded,
                       jyppx::liuding::RestoreLevel::best_effort}) {
      auto& state = manifest.participants.front().state;
      state.restore_level = level;
      if (level != jyppx::liuding::RestoreLevel::exact)
        state.degradation_reasons = {"topology may differ"};
      result.push_back(fuzz::copy_bytes(
          interop::checkpoint_artifact_manifest_json(manifest, limits())));
    }
    for (const auto& seed : result) {
      static_cast<void>(interop::parse_checkpoint_artifact_manifest_json(
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
