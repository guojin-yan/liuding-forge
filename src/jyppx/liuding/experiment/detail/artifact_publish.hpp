// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_DETAIL_ARTIFACT_PUBLISH_HPP
#define JYPPX_EXPERIMENT_DETAIL_ARTIFACT_PUBLISH_HPP

#include <filesystem>
#include <string>
#include <system_error>

namespace jyppx::liuding::experiment::detail {

// Internal same-filesystem publication; neither declaration is installed.
// Keep the forwarding definition in its own archive member so integration
// executables can supervise the native call without changing product APIs.
std::error_code publish_artifact_file(const std::filesystem::path& source,
                                      const std::filesystem::path& destination);

#ifdef _WIN32
// Resolve ordinary DOS/UNC paths without MAX_PATH-sized buffers before adding
// the extended-length prefix. Already qualified namespaces retain exact bytes.
// This private helper may allocate; native Win32 resolution errors are returned
// in error, and allocation exceptions are contained by the noexcept publisher.
std::wstring windows_artifact_path(const std::filesystem::path& path,
                                   std::error_code& error);
#endif

// Performs the actual platform publication with the existing native flags.
std::error_code native_publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) noexcept;

}  // namespace jyppx::liuding::experiment::detail

#endif  // JYPPX_EXPERIMENT_DETAIL_ARTIFACT_PUBLISH_HPP
