// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "fuzz_support.hpp"
#include "telemetry_seed.hpp"

#include <jyppx/foundation/sha256.hpp>
#include <jyppx/foundation/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {
namespace jy = jyppx;
namespace fuzz = jy::fuzz;

void exercise(std::span<const std::byte> input) {
  const auto limits = fuzz::telemetry_limits();
  if (auto envelope = fuzz::parse_input(
          [&] { return jy::decode_telemetry_envelope(input, limits); })) {
    const auto canonical = jy::encode_telemetry_envelope(*envelope, limits);
    fuzz::require(std::equal(canonical.begin(), canonical.end(), input.begin(),
                             input.end()),
                  "telemetry decoder accepted noncanonical binary bytes");
    const auto reparsed = jy::decode_telemetry_envelope(canonical, limits);
    fuzz::require(
        jy::encode_telemetry_envelope(reparsed, limits) == canonical &&
            jy::telemetry_content_digest(reparsed, limits).hex ==
                jy::sha256_hex(canonical),
        "telemetry roundtrip changed content/hash");
    auto exact = limits;
    exact.max_envelope_bytes = canonical.size();
    fuzz::require(jy::encode_telemetry_envelope(*envelope, exact) == canonical,
                  "telemetry exact encoding budget rejected accepted content");
    static_cast<void>(jy::decode_telemetry_envelope(canonical, exact));
    --exact.max_envelope_bytes;
    fuzz::require_resource_rejection([&] {
      static_cast<void>(jy::decode_telemetry_envelope(canonical, exact));
    });
    fuzz::require_resource_rejection([&] {
      static_cast<void>(jy::encode_telemetry_envelope(*envelope, exact));
    });
  }
  if (auto definition = fuzz::parse_input([&] {
        return jy::decode_telemetry_metric_definition(input, limits);
      })) {
    const auto canonical =
        jy::encode_telemetry_metric_definition(*definition, limits);
    fuzz::require(std::equal(canonical.begin(), canonical.end(), input.begin(),
                             input.end()),
                  "metric decoder accepted noncanonical binary bytes");
    const auto reparsed =
        jy::decode_telemetry_metric_definition(canonical, limits);
    fuzz::require(
        jy::encode_telemetry_metric_definition(reparsed, limits) == canonical &&
            jy::telemetry_metric_definition_digest(reparsed, limits).hex ==
                jy::sha256_hex(canonical),
        "metric definition roundtrip changed canonical content/hash");
    auto exact = limits;
    exact.max_envelope_bytes = canonical.size();
    static_cast<void>(jy::decode_telemetry_metric_definition(canonical, exact));
    fuzz::require(
        jy::encode_telemetry_metric_definition(*definition, exact) == canonical,
        "metric definition exact encoding budget rejected content");
    --exact.max_envelope_bytes;
    fuzz::require_resource_rejection([&] {
      static_cast<void>(
          jy::decode_telemetry_metric_definition(canonical, exact));
    });
    fuzz::require_resource_rejection([&] {
      static_cast<void>(
          jy::encode_telemetry_metric_definition(*definition, exact));
    });
  }
}

const std::vector<std::vector<std::byte>>& seeds() {
  static const auto value = [] {
    std::vector<std::vector<std::byte>> result;
    for (unsigned kind = 0U; kind < 6U; ++kind)
      result.push_back(jy::encode_telemetry_envelope(fuzz::seed_envelope(kind),
                                                     fuzz::telemetry_limits()));
    result.push_back(jy::encode_telemetry_metric_definition(
        fuzz::seed_definition(), fuzz::telemetry_limits()));
    for (std::size_t index = 0U; index < result.size(); ++index) {
      if (index < 6U)
        static_cast<void>(jy::decode_telemetry_envelope(
            result[index], fuzz::telemetry_limits()));
      else
        static_cast<void>(jy::decode_telemetry_metric_definition(
            result[index], fuzz::telemetry_limits()));
      exercise(result[index]);
    }
    return result;
  }();
  return value;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,
                                      std::size_t size) {
  return fuzz::exercise_raw_and_seed(data, size, seeds(), exercise, 37U);
}
