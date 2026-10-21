// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_RUN_STORE_HPP
#define JYPPX_EXPERIMENT_RUN_STORE_HPP

/**
 * @file run_store.hpp
 * @brief Durable run metadata and SQLite coordination with the local CAS.
 */

#include <jyppx/experiment/artifact_store.hpp>
#include <jyppx/experiment/run.hpp>
#include <jyppx/experiment/run_events.hpp>
#include <jyppx/foundation/artifact.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::experiment {

/** @brief Persisted semantic role distinguishing artifacts and checkpoints. */
enum class RunArtifactKind : std::uint8_t {
  artifact = 1,  ///< Configuration, dataset, model, report, or other artifact.
  checkpoint = 2,  ///< Recoverable training checkpoint content.
};

/**
 * @brief One immutable run-to-artifact relationship.
 *
 * `reference_id` is the caller's idempotency identity within one run. Reusing
 * it with different metadata is an error. The artifact bytes are committed
 * before this relationship becomes durable.
 */
struct RunArtifactReference {
  std::string reference_id;  ///< Stable relationship identity within one run.
  RunArtifactKind kind = RunArtifactKind::artifact;  ///< Persisted link kind.
  std::string role;      ///< Semantic role such as `config`, `last`, or `best`.
  ArtifactRef artifact;  ///< Verified immutable artifact identity and metadata.
  std::uint64_t attached_at_unix_ms = 0;  ///< Relationship timestamp.
};

/** @brief Bounds and durability options captured by one local run store. */
struct SQLiteRunStoreOptions {
  std::chrono::milliseconds busy_timeout =
      std::chrono::seconds(5);  ///< Maximum SQLite writer lock wait.
  RunMetadataLimits run_metadata_limits{};  ///< Bounds for stored run fields.
  ArtifactStoreLimits artifact_limits{};    ///< Bounds for the managed CAS.
  std::uint64_t max_runs = 1000000U;  ///< Maximum persisted run identities.
  std::size_t max_references_per_run = 16384U;  ///< Maximum links per run.
  std::uint64_t max_run_artifact_bytes =
      1ULL << 42U;  ///< Sum of linked artifact sizes, including distinct roles.
  std::size_t max_reference_id_bytes =
      256U;                            ///< Maximum relationship ID bytes.
  std::size_t max_role_bytes = 128U;   ///< Maximum semantic role bytes.
  RunEventStoreLimits event_limits{};  ///< Durable envelope/index/page bounds.
  bool allow_v1_migration = true;  ///< Transactionally upgrade recognized v1.
  std::uint64_t max_backup_database_bytes = 1ULL
                                            << 36U;  ///< Snapshot DB bound.
  std::uint64_t max_backup_manifest_bytes = 128ULL << 20U;  ///< Manifest bound.
};

/** @brief Complete database and live-CAS backup/restore publication counts. */
struct RunBackupReport {
  std::uint64_t database_bytes = 0U;  ///< Verified standalone database bytes.
  std::uint64_t artifact_bytes =
      0U;  ///< Verified distinct referenced blob bytes.
  std::size_t artifact_count = 0U;  ///< Distinct referenced immutable blobs.
};

/** @brief Bounded database and optional referenced-content integrity report. */
struct RunStoreIntegrityReport {
  std::vector<std::string> issues;  ///< Database, foreign-key, or CAS failures.
  std::size_t checked_artifacts = 0U;  ///< Referenced distinct blobs checked.

  /**
   * @brief Reports whether integrity validation found no failures.
   * @return True when the issue list is empty.
   */
  bool ok() const noexcept { return issues.empty(); }
};

/** @brief Bounded run-ID keyset selection; use last returned ID to continue.
 * @note Pages order by the immutable run primary key. Metadata may advance and
 * new runs may appear between calls; this is not a historical metadata
 * snapshot.
 */
struct RunListRequest {
  std::string after_run_id;  ///< Exclusive primary key; empty starts a scan.
  std::size_t limit = 256U;  ///< Positive count, at most 1024 per call.
  std::optional<std::string> project_id;  ///< Optional exact project filter.
  std::optional<std::string>
      experiment_id;                ///< Optional exact experiment filter.
  std::optional<RunStatus> status;  ///< Optional exact current status filter.
};

/**
 * @brief Narrow durable metadata sink, independent of the training loop.
 *
 * Events and metric batches use their own versioned contracts. These methods
 * persist identity, lifecycle, lineage, and artifact relationships only.
 */
class RunSink {
 public:
  /** @brief Releases this metadata sink. */
  virtual ~RunSink() = default;

  /**
   * @brief Creates a run, or replays its identical original start request.
   * @param metadata Immutable start fields with created or running status.
   * @return Current persisted metadata, including any later lifecycle change.
   */
  virtual RunMetadata start(const RunMetadata& metadata) = 0;

  /**
   * @brief Atomically advances lifecycle state; repeated target is idempotent.
   * @param run_id Stable existing run identity.
   * @param status Requested next lifecycle status.
   * @param updated_at_unix_ms Monotonic timestamp for a new transition.
   * @return Current persisted run metadata.
   */
  virtual RunMetadata transition(std::string_view run_id, RunStatus status,
                                 std::uint64_t updated_at_unix_ms) = 0;

  /**
   * @brief Atomically attaches already committed blobs to one run.
   * @param run_id Stable existing run identity.
   * @param references Complete batch of idempotent immutable relationships.
   * @throws Error If any relationship, quota, or committed blob is invalid.
   */
  virtual void attach(std::string_view run_id,
                      std::span<const RunArtifactReference> references) = 0;
};

/**
 * @brief SQLite WAL metadata backend owning coordination with a local CAS.
 *
 * A dedicated root contains `runs.sqlite3` and `artifacts/`. SQLite uses WAL,
 * full synchronous commits, foreign keys, a schema version, and a bounded busy
 * timeout. Public operations are serialized per handle and safe to call from
 * different threads. Separate handles and processes coordinate writers using
 * SQLite's `BEGIN IMMEDIATE` lock on this one database.
 *
 * `put_artifact`, `attach`, and `collect_garbage` hold that writer lock across
 * CAS verification/publication or garbage collection and the metadata commit.
 * Use this class for all GC on its managed root. A caller may publish bytes
 * through FileArtifactStore before `attach`; a concurrent GC can then remove
 * an unreferenced blob and `attach` fails without acknowledging a relationship.
 * A failure after blob publication leaves an unreferenced object for later GC;
 * it never deletes previously confirmed relationships or their content.
 *
 * The store does not interrupt runs automatically at open, because another
 * process may still own them. Recovery requires an explicit stale cutoff.
 */
class SQLiteRunStore final : public RunSink, public RunEventSink {
 public:
  /** @brief Current local metadata schema version. */
  static constexpr std::uint32_t schema_version = 2U;

  /**
   * @brief Opens or atomically initializes a local run database and CAS.
   * @param root Dedicated local storage directory.
   * @param options Metadata, capacity, and lock wait bounds.
   * @param stop Cooperative cancellation checked before filesystem creation.
   * @param deadline Absolute open/migration deadline, including SQLite waits.
   * @throws CompatibilityError On a foreign or unsupported database schema.
   * @throws Error On an invalid root, corrupt database, or SQLite failure.
   */
  explicit SQLiteRunStore(std::filesystem::path root,
                          SQLiteRunStoreOptions options = {},
                          std::stop_token stop = {},
                          TelemetryDeadline deadline = std::nullopt);

  /** @brief Closes SQLite without deleting committed metadata or blobs. */
  ~SQLiteRunStore() override;

  /** @brief Store handles cannot be copied. */
  SQLiteRunStore(const SQLiteRunStore&) = delete;
  /** @brief Store handles cannot be copy assigned. */
  SQLiteRunStore& operator=(const SQLiteRunStore&) = delete;
  /** @brief Store handles cannot be moved while operations may be active. */
  SQLiteRunStore(SQLiteRunStore&&) = delete;
  /** @brief Store handles cannot be move assigned. */
  SQLiteRunStore& operator=(SQLiteRunStore&&) = delete;

  /**
   * @brief Persists a new identity or replays the identical original request.
   * @param metadata Validated start metadata in created or running state.
   * @return The current durable metadata for the requested identity.
   */
  RunMetadata start(const RunMetadata& metadata) override;

  /**
   * @brief Persists one permitted lifecycle transition atomically.
   * @param run_id Existing run identity.
   * @param status Next status, or the persisted status for an idempotent retry.
   * @param updated_at_unix_ms Monotonic timestamp for a new transition.
   * @return Current durable run metadata.
   */
  RunMetadata transition(std::string_view run_id, RunStatus status,
                         std::uint64_t updated_at_unix_ms) override;

  /**
   * @brief Verifies committed content and atomically records a reference batch.
   * @param run_id Existing run identity.
   * @param references Immutable idempotent relationships committed together.
   */
  void attach(std::string_view run_id,
              std::span<const RunArtifactReference> references) override;

  /**
   * @brief Publishes bytes and then commits one relationship under writer lock.
   * @param run_id Existing run identity.
   * @param reference_id Stable idempotency identity within that run.
   * @param kind Artifact or checkpoint relationship kind.
   * @param role Semantic relationship role.
   * @param bytes Exact content to publish.
   * @param descriptor Producer metadata and optional exact content size.
   * @param attached_at_unix_ms Relationship timestamp.
   * @param expected_digest Optional producer digest assertion.
   * @return The durable relationship to the immutable content.
   */
  RunArtifactReference put_artifact(
      std::string_view run_id, std::string reference_id, RunArtifactKind kind,
      std::string role, std::span<const std::byte> bytes,
      const ArtifactDescriptor& descriptor, std::uint64_t attached_at_unix_ms,
      std::optional<ArtifactDigest> expected_digest = std::nullopt);

  /**
   * @brief Finds one run without changing it.
   * @param run_id Stable run identity to query.
   * @return Detached metadata, or no value when absent.
   */
  std::optional<RunMetadata> find(std::string_view run_id) const;

  /** @brief Selects at most 1024 runs in strictly increasing run-ID order.
   * @param request Exclusive cursor, positive count, and optional exact
   * filters.
   * @param stop Cooperative cancellation, including handle mutex waiting.
   * @param deadline Absolute steady-clock deadline.
   * @return Detached metadata; continue from the last returned run ID.
   */
  std::vector<RunMetadata> list(
      const RunListRequest& request = {}, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) const;

  /**
   * @brief Reads a run's relationships in reference ID order.
   * @param run_id Existing run identity.
   * @return Detached immutable relationships, possibly empty.
   */
  std::vector<RunArtifactReference> references(std::string_view run_id) const;

  /**
   * @brief Advances the activity timestamp of a running run.
   * @param run_id Existing running run identity.
   * @param updated_at_unix_ms Monotonic activity timestamp.
   * @return Current durable run metadata.
   */
  RunMetadata heartbeat(std::string_view run_id,
                        std::uint64_t updated_at_unix_ms);

  /**
   * @brief Marks running runs at or before a stale cutoff as interrupted.
   * @param stale_at_or_before_unix_ms Maximum stale activity timestamp.
   * @param interrupted_at_unix_ms Timestamp at or after the stale cutoff.
   * @return Number of runs changed in the one transaction.
   */
  std::size_t recover_interrupted(std::uint64_t stale_at_or_before_unix_ms,
                                  std::uint64_t interrupted_at_unix_ms);

  /**
   * @brief Checks SQLite integrity, foreign keys, and optionally live CAS
   * blobs.
   * @param verify_artifacts Whether to hash every referenced distinct blob.
   * @param stop Cooperative cancellation between bounded audit steps.
   * @param deadline Absolute deadline for SQL/audit lock waiting.
   * @return Bounded diagnostic report without repairing any content.
   */
  RunStoreIntegrityReport integrity_check(
      bool verify_artifacts = true, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) const;

  /**
   * @brief Collects unreferenced CAS blobs while holding the SQLite writer
   * lock.
   * @param grace_period Minimum unreferenced object age before deletion.
   * @return Removed and retained object counts and bytes.
   */
  ArtifactGcReport collect_garbage(
      std::chrono::seconds grace_period = std::chrono::hours(24));

  /**
   * @brief Returns the canonical dedicated storage root.
   * @return Borrowed root path valid for this handle's lifetime.
   */
  const std::filesystem::path& root() const noexcept;

  /** @brief Atomically appends/replays canonical events and metric indices.
   * @param run_id Existing run correlated by every envelope.
   * @param batch_key Run-local ordered-batch idempotency identity.
   * @param envelopes Immutable neutral producer observations.
   * @param stop Cooperative cancellation, including mutex/SQLite lock waits.
   * @param deadline Absolute steady-clock deadline.
   * @return Full positional durable receipts, never partial success.
   */
  RunEventAppendResult append(
      std::string_view run_id, std::string_view batch_key,
      std::span<const TelemetryEnvelope> envelopes, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;

  /** @brief Reads an event snapshot with bounded keyset pagination.
   * @param run_id Existing run identity.
   * @param request Count, filter, cursor, and snapshot boundary.
   * @param stop Cooperative cancellation.
   * @param deadline Absolute steady-clock deadline.
   * @return Canonical decoded envelopes and a resumable exclusive cursor.
   */
  RunEventPage events(std::string_view run_id,
                      const RunEventPageRequest& request = {},
                      std::stop_token stop = {},
                      TelemetryDeadline deadline = std::nullopt) const;

  /** @brief Queries indexed points while decoding their source envelopes.
   * @param run_id Existing run identity.
   * @param request Immutable definition/step filter and snapshot cursor.
   * @param stop Cooperative cancellation.
   * @param deadline Absolute steady-clock deadline.
   * @return Full metric semantics; no implicit reaggregation or dropped NaN.
   */
  RunMetricPage metrics(std::string_view run_id,
                        const RunMetricPageRequest& request,
                        std::stop_token stop = {},
                        TelemetryDeadline deadline = std::nullopt) const;

  /**
   * @brief Returns the metadata database path.
   * @return Borrowed database path valid for this handle's lifetime.
   */
  const std::filesystem::path& database_path() const noexcept;

  /**
   * @brief Returns the managed FileArtifactStore root for verified reads.
   * @return Borrowed CAS root path valid for this handle's lifetime.
   */
  const std::filesystem::path& artifact_root() const noexcept;

  /** @brief Returns immutable event/codec capacity bounds for sink adapters.
   * @return Borrowed limits valid for this store's lifetime.
   */
  const RunEventStoreLimits& event_limits() const noexcept;

  /** @brief Publishes a complete DB/live-CAS backup into a new directory.
   * @param destination Nonexistent destination outside this store's root.
   * @param stop Cooperative cancellation between bounded copy/SQLite steps.
   * @param deadline Absolute operation deadline; failed staging is removable.
   * @return Counts after atomic directory publication; original data is
   * retained.
   * @note Holds the SQLite writer lock while creating the consistent snapshot.
   */
  RunBackupReport backup(const std::filesystem::path& destination,
                         std::stop_token stop = {},
                         TelemetryDeadline deadline = std::nullopt) const;

  /** @brief Validates a complete backup and publishes a new restored store.
   * @param backup_root Published format-v1 backup directory.
   * @param target_root Nonexistent target; existing stores are never
   * overwritten.
   * @param options Capacity/SQLite limits for validation and restored store.
   * @param stop Cooperative cancellation between bounded filesystem steps.
   * @param deadline Absolute operation deadline.
   * @return Counts after complete DB, FK, event, CAS, and manifest validation.
   */
  static RunBackupReport restore_backup(
      const std::filesystem::path& backup_root,
      const std::filesystem::path& target_root,
      SQLiteRunStoreOptions options = {}, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt);

 private:
  /** @brief Private SQLite connection and CAS coordination implementation. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Owned connection, lock, options, and CAS.
};

}  // namespace jyppx::experiment

#endif  // JYPPX_EXPERIMENT_RUN_STORE_HPP
