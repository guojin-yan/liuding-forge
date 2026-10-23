// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TRAINING_TELEMETRY_ADAPTER_HPP
#define JYPPX_LIUDING_TRAINING_TELEMETRY_ADAPTER_HPP
/** @file telemetry_adapter.hpp
 * @brief One-way callback adaptation and atomic N-1 checkpoint migration.
 * @note This transition header consumes the existing training envelope. It does
 * not expose that envelope from foundation or experiment, and is not a legacy
 * forwarding entry point for new producers.
 */

#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/training/event_sink.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::training {

/** @brief Authoritative typed semantics/context registered for a callback key.
 */
struct TrainingTelemetryMetricBinding {
  TelemetryMetricDefinition definition;  ///< Immutable registered semantics.
  TelemetryAttributes
      context;  ///< Exact context captured for this observation.
};

/** @brief Frozen inputs captured once before adapting a live callback event. */
struct TrainingTelemetryAdapterOptions {
  std::string stream_id;   ///< Current producer-attempt stream, never reused.
  std::string project_id;  ///< Optional project lineage.
  std::string attempt_id;  ///< Optional run-attempt lineage.
  std::map<std::string, TrainingTelemetryMetricBinding>
      metrics;  ///< Complete binding for each metric emitted by the source.
  TelemetryLimits limits;  ///< Explicit neutral envelope/resource bounds.
};

/** @brief Adapts one committed callback snapshot without reading a clock.
 * @param source Existing fully committed callback event/result pair.
 * @param options Immutable stream identity and registered metric semantics.
 * @param captured_time Producer times captured once before enqueue/retry.
 * @return Validated neutral lifecycle envelope with ordered callback audit.
 * @throws Error On invalid callback phase/priority/result, missing metric
 * semantics, malformed identities, or a resource bound violation.
 * @note The caller stages this result in the callback transaction. Retrying
 * sink delivery reuses the staged result, not a fresh call with new times.
 */
TelemetryEnvelope adapt_training_event(
    const EventEnvelope& source, const TrainingTelemetryAdapterOptions& options,
    const TelemetryTime& captured_time);

/** @brief Preserves the old EventSink API without inventing metric semantics.
 * @param source Snapshot staged for an existing training EventSink destination.
 * @param stream_id Stable outbox-owned producer stream.
 * @param captured_time Actual time captured once while staging this live event.
 * @param limits Neutral envelope bounds.
 * @return Neutral compatibility observation with explicitly unknown metrics.
 * @note New neutral destinations use adapt_training_event and typed bindings.
 * This function exists only for the source-compatible old EventSink path.
 */
TelemetryEnvelope adapt_training_compatibility_event(
    const EventEnvelope& source, std::string_view stream_id,
    const TelemetryTime& captured_time, const TelemetryLimits& limits = {});

/** @brief Deterministically converts an old envelope without inventing
 * metadata.
 * @param source Persisted training event-envelope schema 1.
 * @param limits Explicit migration bounds; larger legacy strings require the
 * original documented bounds to be configured rather than truncated.
 * @return Neutral event with deterministic migration stream/key, missing time,
 * source provenance, opaque legacy trace mapping, and unknown metric semantics.
 * @throws Error On unsupported/corrupt source or insufficient migration bounds.
 * @note Definitions are derived only from source bytes. An external mutable
 * registry must not change the content of the same migrated operation.
 */
TelemetryEnvelope migrate_training_envelope_v1(
    const EventEnvelope& source, const TelemetryLimits& limits = {});

/** @brief Cursor fields extracted by the bounded private old outbox reader. */
struct TrainingOutboxMigrationCursor {
  bool initialized = false;  ///< Whether a source run stream was initialized.
  bool terminal = false;     ///< Whether its final event was accepted.
  std::string run_id;  ///< Source run identity, empty before initialization.
  std::uint64_t last_enqueued_event_index = 0U;  ///< Accepted producer cursor.
  std::uint64_t last_epoch = 0U;                 ///< Accepted epoch cursor.
  std::uint64_t last_global_step = 0U;           ///< Accepted optimizer cursor.
  std::uint64_t last_delivered_event_index = 0U;  ///< Fully delivered prefix.
  std::uint64_t last_delivered_epoch = 0U;        ///< Prefix epoch cursor.
  std::uint64_t last_delivered_global_step = 0U;  ///< Prefix optimizer cursor.
};

/** @brief One old pending node, with its original ordered destination cursor.
 */
struct TrainingOutboxV1Pending {
  EventEnvelope envelope;  ///< Old detached decoded envelope.
  std::size_t next_sink_index =
      0U;  ///< First unacknowledged logical destination.
};

/** @brief Detached old-state view after strict descriptor/field decoding.
 * @note The private old codec remains the sole decoder for the old blob. This
 * view is not a second reader or an alternative public checkpoint format.
 */
struct TrainingOutboxV1MigrationInput {
  TrainingOutboxMigrationCursor cursor;    ///< Original complete control state.
  std::vector<EventSinkDescriptor> sinks;  ///< Original sorted sink topology.
  std::vector<TrainingOutboxV1Pending>
      pending;  ///< Gap-free undelivered suffix.
};

/** @brief A migrated node preserving retry position in the unchanged topology.
 */
struct TrainingOutboxMigratedPending {
  TelemetryEnvelope envelope;        ///< Neutral immutable observation.
  std::size_t next_sink_index = 0U;  ///< Preserved first unacknowledged sink.
};

/** @brief Fully validated candidate installed only after migration succeeds. */
struct TrainingOutboxMigrationCandidate {
  TrainingOutboxMigrationCursor cursor;  ///< Unmodified original control state.
  std::vector<TrainingOutboxMigratedPending> pending;  ///< Migrated suffix.
  std::uint64_t canonical_pending_bytes =
      0U;  ///< New event bytes plus node charge.
};

/** @brief Builds a bounded N-1 candidate without changing any live outbox.
 * @param source Detached schema-1 state from the private bounded old reader.
 * @param current_sinks New descriptors in exactly the old logical sink order.
 * @param allowed_type_changes Explicit old/new implementation-schema pairs.
 * Destination IDs and sink IDs can never be remapped by this list.
 * @param limits Bounds for every converted envelope.
 * @param max_pending_events New queue capacity, checked before allocation.
 * @param max_pending_bytes New canonical queue budget including node charges.
 * @return Complete candidate preserving prefix/suffix, cursors, audit,
 * terminal.
 * @throws Error On corrupt source, topology mismatch, unsupported type mapping,
 * conversion failure, or resource exhaustion; source/live state stays intact.
 */
TrainingOutboxMigrationCandidate migrate_training_outbox_v1(
    const TrainingOutboxV1MigrationInput& source,
    std::span<const TelemetrySinkDescriptor> current_sinks,
    std::span<const std::pair<std::string, std::string>> allowed_type_changes,
    const TelemetryLimits& limits = {}, std::size_t max_pending_events = 4096U,
    std::uint64_t max_pending_bytes = 64U * 1024U * 1024U);

}  // namespace jyppx::liuding::training

#endif  // JYPPX_LIUDING_TRAINING_TELEMETRY_ADAPTER_HPP