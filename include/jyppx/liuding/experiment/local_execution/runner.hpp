// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_RUNNER_HPP
#define JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_RUNNER_HPP

/** @file runner.hpp
 * @brief Durable native local jobs sharing the run/SQLite/CAS state machine.
 */
#include <jyppx/liuding/experiment/run_store.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::experiment {

/** @brief Trusted native bootstrap for an independently owned job worker.
 * The executable must call run_local_job_worker for the appended arguments.
 * Each new attempt rechecks its SHA before recording a launch intent. The
 * caller owns the executable and runtime environment through the invocation.
 * Confirmed terminal replay does not require this program to remain present.
 */
struct LocalJobWorkerLaunch {
  std::filesystem::path executable;  ///< Absolute native worker executable.
  std::string executable_sha256;  ///< Exact lowercase SHA-256, at most 1 GiB.
  std::vector<std::string>
      arguments;  ///< Fixed prefix before worker arguments.
};

/** @brief Detached durable job description, not a native child exit code. */
struct LocalJobResult {
  std::string json;  ///< Versioned job/attempt JSON using the local CLI schema.
  int status_code = 0;  ///< 0 success/query, 1 failed, 7 fenced, 130 cancelled.
};

/** @brief Explicit persistent control; no action guesses ownership. */
enum class LocalJobControl : std::uint8_t {
  show,     ///< Read committed state without launching a worker.
  recover,  ///< Reconcile confirmed results and advance permitted attempts.
  cancel    ///< Persist an idempotent cancellation request.
};

/** @brief Validates the bounded v1 local spec and hashes its canonical JSON.
 * @param specification_json At most 64 KiB of versioned local_run_spec JSON.
 * @param inspect_files Also verify current files, placement and admission.
 * @return Lowercase SHA-256 of the canonical frozen specification.
 * @throws std::exception On invalid JSON, schema, identity, bounds or
 * preflight.
 */
std::string local_execution_spec_sha256(std::string_view specification_json,
                                        bool inspect_files = false);

/** @brief Resolves and hashes a spec's effective native environment only.
 * @param specification_json Bounded v1 local spec; environment references
 * resolve now.
 * @return Platform-specific digest with runner-owned identity fields excluded.
 * @throws std::exception On invalid spec, unavailable references or identity
 * mismatch.
 */
std::string local_execution_environment_sha256(
    std::string_view specification_json);

/** @brief Executes or replays one immutable durable job using native workers.
 * The supplied worker prefix receives --root, --id, --attempt and
 * --intent-sha256 arguments. No shell or CLI executable is required. Workers
 * own execution locks independently; unknown intent/result state stays fenced.
 * Only confirmed nonzero exits allowed by the spec can retry. Typed outputs,
 * cancellation and cleanup use the same transactions as CLI local execution.
 * @param root Caller-owned run/SQLite/CAS directory, created for valid new
 * jobs.
 * @param job_id Stable ASCII job identity, at most 128 characters.
 * @param specification_json Bounded versioned v1 local_run_spec JSON.
 * @param worker Trusted native bootstrap, used only for actual new attempts.
 * @param parent_run_id Optional existing run parent, immutable for this job.
 * @return Committed description and orchestration status, possibly fenced.
 * @throws std::exception On invalid inputs, changed frozen identity or storage
 * failure.
 */
LocalJobResult execute_local_job(
    const std::filesystem::path& root, std::string_view job_id,
    std::string_view specification_json, const LocalJobWorkerLaunch& worker,
    std::optional<std::string> parent_run_id = std::nullopt);

/** @brief Queries, cancels or reconciles a previously created durable job.
 * @param action Explicit persistent action.
 * @param root Existing run/SQLite/CAS directory.
 * @param job_id Previously created job identity.
 * @param worker Required only if recovery actually admits a new attempt.
 * @return Detached job state; unknown ownership returns status 7.
 * @throws std::exception On invalid action, missing job, storage or identity
 * failure.
 */
LocalJobResult control_local_job(LocalJobControl action,
                                 const std::filesystem::path& root,
                                 std::string_view job_id,
                                 const LocalJobWorkerLaunch& worker = {});

/** @brief Records explicit external cleanup proof for an ambiguous attempt.
 * This never infers cleanup from locks, PIDs or time, and never retries the
 * abandoned attempt. Existing confirmation schema and CAS fences apply.
 * @param root Existing run/SQLite/CAS directory.
 * @param job_id Previously created job identity.
 * @param confirmation_json Explicit bounded abandonment confirmation JSON.
 * @return Failed job description after verified confirmation/resource release.
 * @throws std::exception On invalid confirmation, active ownership or identity
 * mismatch.
 */
LocalJobResult abandon_local_job(const std::filesystem::path& root,
                                 std::string_view job_id,
                                 std::string_view confirmation_json);

/** @brief Native worker entry called by the caller's bootstrap executable.
 * Accept only arguments passed to that executable's designated worker mode.
 * Checks immutable intent/attempt ownership before any application launch.
 * The worker may survive its dispatcher and commit the confirmed result.
 * @param root Existing run/SQLite/CAS directory.
 * @param job_id Admitted job identity.
 * @param attempt_index Positive attempt index, at most the frozen bound.
 * @param intent_sha256 Exact persisted launch-intent digest.
 * @return Detached worker commit acknowledgement.
 * @throws std::exception On invalid intent, ownership, bounds or storage
 * failure.
 */
LocalJobResult run_local_job_worker(const std::filesystem::path& root,
                                    std::string_view job_id,
                                    std::uint64_t attempt_index,
                                    std::string_view intent_sha256);

/** @brief Checks one committed typed source with the execution identity fence.
 * @param store Existing durable source store; not modified by this check.
 * @param project_id Required source project.
 * @param binding_json Versioned spec binding with run/reference/SHA/size
 * fields.
 * @param kind Required artifact or checkpoint kind.
 * @param maximum_bytes Artifact limit, 1 byte through 1 GiB.
 * @throws std::exception On source, attempt, content, size or project mismatch.
 */
void validate_local_training_source_json(SQLiteRunStore& store,
                                         std::string_view project_id,
                                         std::string_view binding_json,
                                         RunArtifactKind kind,
                                         std::uint64_t maximum_bytes = 1U
                                                                       << 20U);

}  // namespace jyppx::liuding::experiment
#endif  // JYPPX_LIUDING_EXPERIMENT_LOCAL_EXECUTION_RUNNER_HPP
