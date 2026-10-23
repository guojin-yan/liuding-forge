// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_TESTS_FUZZ_SUPPORT_HPP
#define JYPPX_TESTS_FUZZ_SUPPORT_HPP

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace jyppx::liuding::fuzz {

inline constexpr std::size_t kMaxInputBytes = 64U * 1024U;
// Canonical JSON may escape/reformat accepted source bytes. Its byte budget
// must not accidentally turn a valid roundtrip into an input rejection.
inline constexpr std::size_t kCanonicalBytes = 1024U * 1024U;

inline void require(bool condition, const char* message) {
  if (!condition) {
    std::fputs(message, stderr);
    std::fputc('\n', stderr);
    std::abort();
  }
}

// Catch only public input-rejection categories. InternalError, unexpected
// framework errors, std exceptions and sanitizer failures must reach libFuzzer.
template <typename Function>
auto parse_input(Function&& function, bool backend_error_is_input = false)
    -> std::optional<std::invoke_result_t<Function>> {
  try {
    return std::invoke(std::forward<Function>(function));
  } catch (const Error& error) {
    switch (error.code()) {
      case ErrorCode::invalid_argument:
      case ErrorCode::serialization:
      case ErrorCode::compatibility:
      case ErrorCode::resource_exhausted:
        return std::nullopt;
      case ErrorCode::backend:
        if (backend_error_is_input) return std::nullopt;
        throw;
      default:
        throw;
    }
  }
}

template <typename Function>
void require_resource_rejection(Function&& function) {
  try {
    std::invoke(std::forward<Function>(function));
  } catch (const ResourceExhausted&) {
    return;
  }
  require(false, "parser/encoder accepted a byte budget one below actual size");
}

inline std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

inline std::string_view text(std::span<const std::byte> value) {
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}

inline std::vector<std::byte> copy_bytes(std::string_view value) {
  const auto view = bytes(value);
  return {view.begin(), view.end()};
}

inline BuildFingerprint seed_build() {
  return {.jyppx_version = "1.0.0",
          .git_commit = "fuzz-seed",
          .compiler = "clang-fuzz",
          .cxx_standard = "202002",
          .os_arch = "linux-x86_64",
          .build_type = "Debug",
          .libtorch_version = "2.5.1",
          .libtorch_build = "cpu",
          .accelerator_runtime = "cpu",
          .dependency_lock_digest = "unlocked"};
}

// In addition to raw input, apply bounded edits inside an actual production
// encoding. Some mutations preserve the binary framing/domain to explore fields
// past the initial magic check. The raw path still mutates every framing byte.
inline std::vector<std::byte> mutate_seed(std::span<const std::byte> seed,
                                          std::span<const std::byte> mutations,
                                          std::size_t protected_prefix = 0U) {
  require(!seed.empty() && seed.size() <= kMaxInputBytes,
          "fuzz seed is empty or exceeds the bounded input cap");
  std::vector<std::byte> result(seed.begin(), seed.end());
  if (mutations.empty()) return result;
  const auto control = std::to_integer<unsigned>(mutations.front());
  const auto begin = (control & 1U) == 0U
                         ? (std::min)(protected_prefix, seed.size() - 1U)
                         : 0U;
  const auto operations = (std::min)(std::size_t{8U}, mutations.size() / 4U);
  for (std::size_t index = 0U; index < operations; ++index) {
    const auto offset = index * 4U;
    const auto value = std::to_integer<unsigned>(mutations[offset]);
    const auto coordinate =
        std::to_integer<unsigned>(mutations[offset + 1U]) |
        (std::to_integer<unsigned>(mutations[offset + 2U]) << 8U);
    const auto position = begin + coordinate % (result.size() - begin);
    switch (value & 3U) {
      case 0U:
        result[position] ^= mutations[offset + 3U];
        break;
      case 1U:
        result[position] = mutations[offset + 3U];
        break;
      case 2U:
        if (result.size() < kMaxInputBytes)
          result.insert(result.begin() + static_cast<std::ptrdiff_t>(position),
                        mutations[offset + 3U]);
        break;
      default:
        if (result.size() > begin + 1U)
          result.erase(result.begin() + static_cast<std::ptrdiff_t>(position));
        break;
    }
  }
  return result;
}

template <typename Exercise>
int exercise_raw_and_seed(const std::uint8_t* data, std::size_t size,
                          const std::vector<std::vector<std::byte>>& seeds,
                          Exercise&& exercise,
                          std::size_t protected_prefix = 0U) {
  if (size > kMaxInputBytes) return 0;
  const auto input = std::as_bytes(std::span(data, size));
  std::invoke(exercise, input);
  require(!seeds.empty(), "fuzzer has no valid structured seeds");
  const auto selected = size == 0U ? 0U : data[0] % seeds.size();
  const auto mutated = mutate_seed(seeds[selected], input, protected_prefix);
  std::invoke(exercise, std::span<const std::byte>(mutated));
  return 0;
}

// A successful initial decode is the only operation allowed to reject input.
// All later failures are bugs, including a supposedly valid canonical encoding
// that fails to reparse. Exact byte budgets are measured from canonical output,
// never from whitespace/order in the untrusted source.
template <typename Limits, typename Decode, typename Encode>
auto exercise_json_codec(std::string_view input, Limits limits,
                         std::size_t Limits::*byte_limit, Decode&& decode,
                         Encode&& encode)
    -> std::optional<std::invoke_result_t<Decode, std::string_view, Limits>> {
  auto parsed = parse_input([&] { return std::invoke(decode, input, limits); });
  if (!parsed) return std::nullopt;
  auto canonical_limits = limits;
  canonical_limits.*byte_limit = kCanonicalBytes;
  const auto canonical = std::invoke(encode, *parsed, canonical_limits);
  require(canonical.size() > 1U && canonical.size() <= kCanonicalBytes,
          "canonical JSON exceeded its independent bounded budget");
  auto exact = canonical_limits;
  exact.*byte_limit = canonical.size();
  require(std::invoke(encode, *parsed, exact) == canonical,
          "JSON encoding changes at its exact byte budget");
  const auto reparsed = std::invoke(decode, std::string_view(canonical), exact);
  require(std::invoke(encode, reparsed, exact) == canonical,
          "parse/encode/reparse changed canonical JSON");
  --(exact.*byte_limit);
  require_resource_rejection([&] {
    static_cast<void>(std::invoke(decode, std::string_view(canonical), exact));
  });
  require_resource_rejection(
      [&] { static_cast<void>(std::invoke(encode, *parsed, exact)); });
  return parsed;
}

}  // namespace jyppx::liuding::fuzz

#endif  // JYPPX_TESTS_FUZZ_SUPPORT_HPP
