// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_CLI_RECIPES_DIAGNOSTICS_FAILURE_REPORT_HPP
#define JYPPX_CLI_RECIPES_DIAGNOSTICS_FAILURE_REPORT_HPP

#include <jyppx/liuding/training/diagnostics.hpp>

#include <cstdint>
#include <filesystem>

namespace jyppx::liuding::experiment {
class SQLiteRunStore;
}

namespace jyppx::liuding::cli::recipe_diagnostics {
// Failure observations are independent immutable artifacts. Never commit the
// failed mutable training domain or change a resumable run's lifecycle here.
// Any secondary publication error is reported without replacing the fit error.
bool publish_failure_report(
    experiment::SQLiteRunStore& store,
    const training::TrainingDiagnostics& diagnostics,
    const std::filesystem::path& root,
    const training::TrainingDiagnosticIdentity& identity,
    std::uint64_t successful_steps, std::uint64_t attached_at_unix_ms) noexcept;
}  // namespace jyppx::liuding::cli::recipe_diagnostics

#endif
