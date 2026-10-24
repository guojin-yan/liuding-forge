// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"

#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {
namespace interop = jyppx::liuding::interop;
namespace fuzz = jyppx::liuding::fuzz;

interop::SafeTensorsLimits limits() {
  return {.max_file_bytes = fuzz::kMaxInputBytes,
          .max_payload_bytes = fuzz::kMaxInputBytes,
          .max_tensor_bytes = fuzz::kMaxInputBytes,
          .max_tensor_elements = fuzz::kMaxInputBytes,
          .max_dimension = fuzz::kMaxInputBytes,
          .max_header_bytes = 32U * 1024U,
          .max_tensors = 64U,
          .max_rank = 8U,
          .max_metadata_entries = 16U,
          .max_string_bytes = 512U,
          .max_total_string_bytes = 16U * 1024U,
          .max_json_nodes = 8192U,
          .max_json_depth = 8U};
}

std::vector<interop::SafeTensorSource> sources(
    const interop::SafeTensorsView& view) {
  std::vector<interop::SafeTensorSource> result;
  for (const auto& tensor : view.directory().tensors)
    result.push_back({.name = tensor.name,
                      .dtype = tensor.dtype,
                      .shape = tensor.shape,
                      .data = view.tensor_bytes(tensor.name)});
  return result;
}

void exercise(std::span<const std::byte> input) {
  const auto parsed = fuzz::parse_input(
      [&] { return interop::SafeTensorsView(input, limits()); });
  if (!parsed) return;
  const auto& original = *parsed;
  fuzz::require(original.digest().hex == jyppx::liuding::sha256_hex(input),
                "safetensors reader changed original artifact digest");
  const auto tensors = sources(original);
  auto canonical_limits = limits();
  // An accepted original may use a more compact spelling/header layout than
  // the deterministic encoder. Measure encoder budgets from its own output.
  canonical_limits.max_file_bytes = fuzz::kCanonicalBytes;
  canonical_limits.max_header_bytes = 256U * 1024U;
  const auto canonical = interop::serialize_safetensors(
      tensors, original.directory().metadata, canonical_limits);
  const interop::SafeTensorsView reparsed(canonical, canonical_limits);
  fuzz::require(
      original.directory().metadata == reparsed.directory().metadata &&
          original.directory().tensors.size() ==
              reparsed.directory().tensors.size(),
      "safetensors roundtrip changed directory metadata");
  for (std::size_t index = 0U; index < original.directory().tensors.size();
       ++index) {
    const auto& before = original.directory().tensors[index];
    const auto& after = reparsed.directory().tensors[index];
    fuzz::require(before.name == after.name && before.dtype == after.dtype &&
                      before.shape == after.shape,
                  "safetensors roundtrip changed tensor association");
    const auto before_bytes = original.tensor_bytes(before.name);
    const auto after_bytes = reparsed.tensor_bytes(after.name);
    fuzz::require(std::equal(before_bytes.begin(), before_bytes.end(),
                             after_bytes.begin(), after_bytes.end()),
                  "safetensors roundtrip changed opaque tensor payload bits");
  }
  const auto canonical_sources = sources(reparsed);
  fuzz::require(interop::serialize_safetensors(canonical_sources,
                                               reparsed.directory().metadata,
                                               canonical_limits) == canonical,
                "safetensors canonicalization was not idempotent");
  auto exact = canonical_limits;
  exact.max_file_bytes = canonical.size();
  static_cast<void>(interop::SafeTensorsView(canonical, exact));
  fuzz::require(interop::serialize_safetensors(
                    tensors, original.directory().metadata, exact) == canonical,
                "safetensors exact file-byte budget changed encoded content");
  --exact.max_file_bytes;
  fuzz::require_resource_rejection(
      [&] { static_cast<void>(interop::SafeTensorsView(canonical, exact)); });
  fuzz::require_resource_rejection([&] {
    static_cast<void>(interop::serialize_safetensors(
        tensors, original.directory().metadata, exact));
  });
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    const std::array<std::byte, 8U> payload{
        std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x3f},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x80}};
    const std::array tensors{
        interop::SafeTensorSource{.name = "layer.weight",
                                  .dtype = interop::SafeTensorDtype::f32,
                                  .shape = {2U},
                                  .data = payload},
        interop::SafeTensorSource{.name = "empty",
                                  .dtype = interop::SafeTensorDtype::u8,
                                  .shape = {0U, 3U},
                                  .data = std::span<const std::byte>{}}};
    std::vector<std::vector<std::byte>> result;
    result.push_back(interop::serialize_safetensors(
        tensors, {{"producer", "fuzz.fixture"}}, limits()));
    result.push_back(interop::serialize_safetensors({}, {}, limits()));
    for (const auto& seed : result) {
      static_cast<void>(interop::SafeTensorsView(seed, limits()));
      exercise(seed);
    }
    return result;
  }();
  return value;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  return fuzz::exercise_raw_and_seed(data, size, seeds(), exercise, 8U);
}
