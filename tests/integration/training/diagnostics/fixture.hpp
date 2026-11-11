// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef LIUDING_TEST_DIAGNOSTIC_FIXTURE_HPP
#define LIUDING_TEST_DIAGNOSTIC_FIXTURE_HPP

#include <filesystem>
#include <string>

namespace diagnostic_recovery {
void require(bool condition, const char* message);
void seed_checkpoint(const std::filesystem::path& root);
std::string checkpoint_report(const std::filesystem::path& root,
                              bool conflicting_model = false);
std::string read_file(const std::filesystem::path& path);
void write_file(const std::filesystem::path& path, const std::string& bytes);
}  // namespace diagnostic_recovery

#endif
