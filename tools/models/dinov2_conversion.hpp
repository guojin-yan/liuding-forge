// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DINOV2_CONVERSION_HPP
#define JYPPX_TOOLS_DINOV2_CONVERSION_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::tools::dinov2 {
// Private tool contract, not a new installed models API or generic converter.
inline constexpr std::string_view kRevision =
    "ed25f3a31f01632728cabb09d1542f84ab7b0056";
inline constexpr std::string_view kSourceSha =
    "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1";
inline constexpr std::string_view kOutputSha =
    "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5";
inline constexpr std::uint64_t kSourceSize = 88249960;
inline constexpr std::uint64_t kOutputSize = 88240512;
inline constexpr std::string_view kMappingVersion =
    "jyppx.liuding.dinov2-hf-small.unmasked.v1";

struct MappingEntry {
  std::string target;
  std::vector<std::uint64_t> shape;
  std::vector<std::string> sources;
};

std::vector<MappingEntry> mapping();
// Tensor-neutral internal layout operation, independently testable with
// fixtures. Never publishes or qualifies bytes; convert() applies both fixed
// SHA gates.
std::vector<std::byte> encode_layout(std::span<const std::byte> source);
// Requires distinct new regular-file output names in existing trusted parents.
// Source and published output retain their fixed complete identities. On errors
// after staging starts, preserve private stages and emit their paths in error.
// Two links are not a transaction: see README.md for crash/rollback boundaries.
nlohmann::json convert(const std::filesystem::path& source,
                       const std::filesystem::path& destination,
                       const std::filesystem::path& report);
}  // namespace jyppx::liuding::tools::dinov2
#endif  // JYPPX_TOOLS_DINOV2_CONVERSION_HPP
