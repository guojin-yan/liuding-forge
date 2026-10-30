// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_DATASETS_PREPARATION_SUPPORT_HPP
#define JYPPX_TOOLS_DATASETS_PREPARATION_SUPPORT_HPP
#include <filesystem>
#include <fstream>
#include <string_view>

namespace jyppx::liuding::datasets::detail {
// Caller keeps ancestors stable for this invocation. This is not a hostile
// concurrent-filesystem sandbox. Links are checked before lexical
// normalization.
void reject_links(const std::filesystem::path& path);
bool present(const std::filesystem::path& path);
std::filesystem::path external_root(const std::filesystem::path& root);
void close_checked(std::ofstream& stream);

// Owns only its uniquely reserved sibling staging directory. Destruction
// removes it on failure; publication uses OS no-replacement rename and disarms
// cleanup.
class StagingDirectory final {
 public:
  StagingDirectory(const std::filesystem::path& root,
                   std::string_view destination, std::string_view prefix);
  ~StagingDirectory();
  StagingDirectory(const StagingDirectory&) = delete;
  StagingDirectory& operator=(const StagingDirectory&) = delete;
  const std::filesystem::path& path() const noexcept { return staging_; }
  const std::filesystem::path& destination() const noexcept {
    return destination_;
  }
  void publish();

 private:
  std::filesystem::path staging_;
  std::filesystem::path destination_;
};
}  // namespace jyppx::liuding::datasets::detail
#endif  // JYPPX_TOOLS_DATASETS_PREPARATION_SUPPORT_HPP
