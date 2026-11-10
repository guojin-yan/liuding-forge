// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_DETAIL_FILE_PUBLICATION_HPP
#define JYPPX_EXPERIMENT_DETAIL_FILE_PUBLICATION_HPP

#include <jyppx/liuding/experiment/artifact_store.hpp>

#include <filesystem>
#include <string_view>
#include <system_error>

namespace jyppx::liuding::experiment::detail {

// Same-directory, no-replace native boundary. Separate archive members allow
// process tests to fence the real native operation without a product hook.
std::error_code publish_output_file(const std::filesystem::path& staging,
                                    const std::filesystem::path& destination);
std::error_code native_publish_output_file(
    const std::filesystem::path& staging,
    const std::filesystem::path& destination) noexcept;

// A verified CAS reader remains alive under the caller's SQLite writer lock.
// The deterministic private stage is reclaimable after process death; final
// files are never replaced, and an existing final must match the entire blob.
void materialize_publication(ArtifactReader& content,
                             const std::filesystem::path& destination,
                             std::string_view intent_digest,
                             std::stop_token stop, TelemetryDeadline deadline);

}  // namespace jyppx::liuding::experiment::detail

#endif  // JYPPX_EXPERIMENT_DETAIL_FILE_PUBLICATION_HPP
