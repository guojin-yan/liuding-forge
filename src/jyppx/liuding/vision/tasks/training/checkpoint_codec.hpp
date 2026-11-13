// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <jyppx/liuding/foundation/checkpoint.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <bit>
#include <span>
#include <string>
#include <vector>

namespace jyppx::liuding::vision::tasks::state {
inline void Append(std::vector<std::byte>& bytes, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    bytes.push_back(static_cast<std::byte>((value >> (i * 8)) & 255U));
}
inline std::uint64_t Read(std::span<const std::byte> bytes,
                          std::size_t& offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    throw SerializationError("Truncated vision training state");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= std::uint64_t(std::to_integer<unsigned char>(bytes[offset++]))
             << (i * 8);
  return value;
}
inline void VerifyManifest(const StateManifest& stored,
                           const StateManifest& current, RestorePolicy policy) {
  validate_state_manifest(stored);
  if (!restore_policy_accepts(policy, stored.restore_level) ||
      stored.participant_id != current.participant_id ||
      stored.schema_version != current.schema_version ||
      stored.restore_level != current.restore_level ||
      stored.degradation_reasons != current.degradation_reasons ||
      !state_fields_equal(stored.fields, current.fields))
    throw CompatibilityError("Incompatible vision training state manifest");
}
inline void VerifyText(const StateReader& reader, std::string_view name,
                       std::string_view expected, std::uint64_t limit) {
  const auto bytes = reader.read_blob(name, limit);
  const auto current = std::as_bytes(std::span(expected));
  if (!std::ranges::equal(bytes, current))
    throw CompatibilityError("Vision training recipe differs from checkpoint");
}
inline bool Identifier(std::string_view value) {
  return !value.empty() && value.size() <= 4096 &&
         std::ranges::all_of(value, [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
         });
}
}  // namespace jyppx::liuding::vision::tasks::state
