// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_BUNDLE_SIGNING_SECURE_FILE_HPP
#define JYPPX_LIUDING_BUNDLE_SIGNING_SECURE_FILE_HPP

#include <cstddef>
#include <filesystem>
#include <span>
#include <vector>

namespace jyppx::liuding::bundle_signing::detail {

/** @brief An exception-safe temporary input allocation erased before release.
 */
class CleansedBytes final {
 public:
  explicit CleansedBytes(std::size_t size);
  ~CleansedBytes();
  CleansedBytes(const CleansedBytes&) = delete;
  CleansedBytes& operator=(const CleansedBytes&) = delete;
  CleansedBytes(CleansedBytes&&) noexcept;
  CleansedBytes& operator=(CleansedBytes&&) = delete;
  std::span<std::byte> bytes() noexcept;
  std::span<const std::byte> bytes() const noexcept;

 private:
  std::vector<std::byte> bytes_;
};

/** @brief Reads one bounded ordinary file using pinned no-follow ancestors. */
CleansedBytes ReadBoundedFile(const std::filesystem::path& path,
                              std::size_t maximum);

}  // namespace jyppx::liuding::bundle_signing::detail

#endif  // JYPPX_LIUDING_BUNDLE_SIGNING_SECURE_FILE_HPP
