// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "failure_report.hpp"

#include "recipe_support.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>

#include <cstdio>
#include <string>
#include <utility>

namespace jyppx::liuding::cli::recipe_diagnostics {
bool publish_failure_report(
    experiment::SQLiteRunStore& store,
    const training::TrainingDiagnostics& diagnostics,
    const std::filesystem::path& root,
    const training::TrainingDiagnosticIdentity& identity,
    std::uint64_t successful_steps,
    std::uint64_t attached_at_unix_ms) noexcept {
  try {
    auto failed = identity;
    failed.attempt_id += ".step-" + std::to_string(successful_steps);
    const auto checkpoint = root / "latest.jyck";
    if (std::filesystem::exists(checkpoint))
      failed.checkpoint = recipe_support::file_digest(checkpoint);
    const auto report = training::training_diagnostics_report_json(
        training::capture_training_diagnostics_report(
            diagnostics, std::move(failed),
            training::TrainingDiagnosticOutcome::failed));
    const auto digest = recipe_support::digest(report);
    // Content identity also distinguishes different failures at the same safe
    // checkpoint. An exact repeated attempt reuses the immutable reference.
    (void)store.put_artifact(
        identity.run_id, "diagnostic.failure." + digest.hex,
        experiment::RunArtifactKind::artifact, "training.diagnostics.failure",
        recipe_support::bytes(report),
        {.logical_name = "training-diagnostics-failure.json",
         .media_type =
             "application/vnd.jyppx.training-diagnostics+json;version=1",
         .producer = "training.diagnostics/v1",
         .expected_size_bytes = report.size()},
        attached_at_unix_ms, digest);
    return true;
  } catch (...) {
    std::fputs(
        "Could not persist training failure diagnostics; the original "
        "training error is preserved.\n",
        stderr);
    return false;
  }
}
}  // namespace jyppx::liuding::cli::recipe_diagnostics
