// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_EXPERIMENT_RUN_HPP
#define JYPPX_LIUDING_EXPERIMENT_RUN_HPP
/**
 * @file run.hpp
 * @brief Storage-neutral run identity, lifecycle, and lineage contracts.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace jyppx::liuding::experiment {

/**
 * @brief Stable run lifecycle values persisted by local metadata backends.
 *
 * Numeric values are part of the on-disk contract and must not be renumbered.
 */
enum class RunStatus : std::uint8_t {
  created = 1,      ///< Identity allocated before execution.
  running = 2,      ///< Execution attempt is active.
  completed = 3,    ///< Execution completed successfully.
  failed = 4,       ///< Execution reported a failure.
  cancelled = 5,    ///< Execution honored a cancellation request.
  interrupted = 6,  ///< Execution ended without a terminal event.
};

/** @brief Returns the stable lowercase status name.
 * @param status Status value to render.
 * @return Stable lowercase name or `unknown`.
 */
std::string_view run_status_name(RunStatus status) noexcept;

/** @brief Returns whether a status is terminal.
 * @param status Status value to inspect.
 * @return True only for terminal lifecycle values.
 */
bool is_terminal_run_status(RunStatus status) noexcept;

/**
 * @brief Checks one lifecycle transition against the append-only run model.
 *
 * A run starts in `created`, may enter `running`, and may enter exactly one
 * terminal state.  Replaying the same transition is idempotent.
 * @param from Persisted current lifecycle value.
 * @param to Requested next lifecycle value.
 * @return True only for allowed or idempotent transitions.
 */
bool is_valid_run_transition(RunStatus from, RunStatus to) noexcept;

/** @brief Immutable lineage references attached to a run identity. */
struct RunLineage {
  std::optional<std::string> parent_run_id;  ///< Prior run or resume parent.
  std::optional<std::string>
      source_digest;  ///< Source identity or tree digest.
  std::optional<std::string> build_fingerprint;  ///< Producer build identity.
  std::optional<std::string>
      config_digest;  ///< Resolved redacted config digest.
  std::optional<std::string>
      data_manifest_digest;  ///< Dataset manifest digest.
};

/**
 * @brief Metadata contract shared by local and external RunSink adapters.
 *
 * Time values are Unix epoch milliseconds supplied by the metadata backend.
 * They are descriptive fields; run identity is carried by `run_id`.
 */
struct RunMetadata {
  std::string run_id;         ///< Stable run identity.
  std::string project_id;     ///< Stable local project identity.
  std::string experiment_id;  ///< Stable experiment identity.
  std::string display_name;   ///< Human-facing display name.
  RunStatus status =
      RunStatus::created;                ///< Current monotonic lifecycle state.
  std::uint64_t created_at_unix_ms = 0;  ///< Creation timestamp.
  std::uint64_t updated_at_unix_ms = 0;  ///< Last metadata timestamp.
  RunLineage lineage;  ///< Immutable producer and input references.
  std::map<std::string, std::string> tags;  ///< Bounded searchable metadata.
};

/** @brief Bounds for validating untrusted run metadata. */
struct RunMetadataLimits {
  std::size_t max_id_bytes = 256U;            ///< Maximum identity field bytes.
  std::size_t max_display_name_bytes = 512U;  ///< Maximum display name bytes.
  std::size_t max_tag_count = 128U;           ///< Maximum tag entries.
  std::size_t max_tag_key_bytes = 128U;       ///< Maximum tag key bytes.
  std::size_t max_tag_value_bytes = 512U;     ///< Maximum tag value bytes.
  std::size_t max_lineage_ref_bytes =
      1024U;  ///< Maximum lineage reference bytes.
};

/**
 * @brief Validates identity, tags, timestamps, lineage, and enum values.
 * @param metadata Immutable run metadata to inspect.
 * @param limits Metadata resource bounds.
 * @throws InvalidArgument On malformed fields or an unknown status value.
 * @throws ResourceExhausted When a configured metadata bound is exceeded.
 */
void validate_run_metadata(const RunMetadata& metadata,
                           const RunMetadataLimits& limits = {});

}  // namespace jyppx::liuding::experiment

#endif  // JYPPX_LIUDING_EXPERIMENT_RUN_HPP