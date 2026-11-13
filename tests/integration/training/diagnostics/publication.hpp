// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef LIUDING_TEST_DIAGNOSTIC_PUBLICATION_HPP
#define LIUDING_TEST_DIAGNOSTIC_PUBLICATION_HPP

#include <filesystem>
#include <string>
#include <string_view>

namespace diagnostic_recovery {
void install_commit_hook();
void arm_publication(std::filesystem::path root, std::string boundary,
                     std::string digest);
[[noreturn]] void fence(std::string_view boundary);
}  // namespace diagnostic_recovery

#endif
