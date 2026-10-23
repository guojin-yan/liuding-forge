// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_DETAIL_ARTIFACT_PUBLISH_HPP
#define JYPPX_EXPERIMENT_DETAIL_ARTIFACT_PUBLISH_HPP

#include <filesystem>
#include <system_error>

namespace jyppx::liuding::experiment::detail {

// Internal same-filesystem publication; neither declaration is installed.
// Keep the forwarding definition in its own archive member so integration
// executables can supervise the native call without changing product APIs.
std::error_code publish_artifact_file(const std::filesystem::path& source,
                                      const std::filesystem::path& destination);

// Performs the actual platform publication with the existing native flags.
std::error_code native_publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) noexcept;

}  // namespace jyppx::liuding::experiment::detail

#endif  // JYPPX_EXPERIMENT_DETAIL_ARTIFACT_PUBLISH_HPP
