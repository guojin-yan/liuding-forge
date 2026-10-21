// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_RUN_EVENTS_HPP
#define JYPPX_EXPERIMENT_RUN_EVENTS_HPP

/** @file run_events.hpp
 * @brief Draft atomic durable event batches and bounded snapshot pagination.
 * @note Next-chapter draft; no production verification is claimed.
 */

#include <jyppx/foundation/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::experiment {

/** @brief Durable-store bounds in addition to neutral telemetry limits. */
struct RunEventStoreLimits {
  TelemetryLimits telemetry{};  ///< Shared codec/envelope/batch bounds.
  std::uint64_t max_events_per_run = 1000000U;  ///< Unique accepted envelopes.
  std::uint64_t max_metric_points_per_run = 10000000U;  ///< Indexed points.
  std::uint64_t max_event_bytes_per_run = 1ULL << 30U;  ///< Canonical bytes.
  std::uint64_t max_batches_per_run = 1000000U;  ///< Durable batch receipts.
  std::uint64_t max_streams_per_run = 4096U;     ///< Producer attempts per run.
  std::uint64_t max_metric_definitions =
      100000U;  ///< Database-wide definitions.
  std::uint64_t max_metric_points_per_batch =
      65536U;                                  ///< Preflight point bound.
  std::size_t max_page_events = 1024U;         ///< Event count bound per query.
  std::size_t max_page_metric_points = 4096U;  ///< Point count bound per query.
  std::uint64_t max_page_bytes = 16U * 1024U * 1024U;  ///< Content budget.
};

/** @brief Receipt whose store coordinate never replaces producer identity. */
struct RunEventReceipt {
  std::uint64_t run_sequence = 0U;  ///< Stable one-based per-run commit order.
  TelemetryDisposition disposition =
      TelemetryDisposition::accepted;  ///< Outcome.
};

/** @brief Complete positional acknowledgment after one atomic batch commit. */
struct RunEventAppendResult {
  std::vector<RunEventReceipt> receipts;  ///< Exactly one receipt per input.
  std::uint64_t committed_through = 0U;   ///< Original commit high watermark.
  bool batch_replay = false;  ///< Whether the exact whole-batch key was stored.
};

/** @brief Event snapshot cursor; first call captures the high watermark. */
struct RunEventPageRequest {
  std::uint64_t after_sequence =
      0U;  ///< Exclusive run-local commit coordinate.
  std::optional<std::uint64_t> snapshot_through;  ///< Reused snapshot boundary.
  std::optional<TelemetryPayloadKind> kind;  ///< Optional typed payload filter.
  std::size_t limit = 256U;                  ///< Positive bounded result count.
};

/** @brief Authoritative decoded envelope and its local pagination coordinate.
 */
struct StoredRunEvent {
  std::uint64_t run_sequence = 0U;  ///< Immutable run commit order.
  TelemetryEnvelope envelope;       ///< Foundation canonical source of truth.
};

/** @brief Count/byte bounded event page from a fixed append-only snapshot. */
struct RunEventPage {
  std::vector<StoredRunEvent> events;       ///< Increasing local commit order.
  std::uint64_t snapshot_through = 0U;      ///< Reuse for subsequent pages.
  std::optional<std::uint64_t> next_after;  ///< Absent at end of the scan.
};

/** @brief Metric cursor unaffected by step resets, missing values, or clocks.
 */
struct RunMetricCursor {
  std::uint64_t run_sequence = 0U;  ///< Envelope commit coordinate.
  std::uint64_t point_index = 0U;   ///< Zero-based index inside that envelope.
};

/** @brief Indexed metric selection with stable keyset pagination. */
struct RunMetricPageRequest {
  std::string metric_id;  ///< Required semantic metric identity.
  std::optional<std::uint32_t> definition_version;  ///< All versions if absent.
  std::optional<TelemetryStepKind> step_kind;  ///< Optional coordinate filter.
  RunMetricCursor after{};  ///< Exclusive cursor; zero sequence begins a scan.
  std::optional<std::uint64_t> snapshot_through;  ///< Fixed high watermark.
  std::size_t limit = 256U;  ///< Positive bounded point count.
};

/** @brief Metric projected from its authoritative source envelope. */
struct StoredRunMetricPoint {
  RunMetricCursor cursor{};          ///< Immutable envelope/point ordering.
  TelemetryIdentity event_identity;  ///< Original producer/retry coordinates.
  TelemetryTime time;  ///< Original clocks, including absent clocks.
  TelemetryCorrelation correlation;        ///< Original trace/run coordinates.
  TelemetryAttributes attributes;          ///< Original envelope annotations.
  std::optional<std::string> resource_id;  ///< Present for resource samples.
  TelemetryMetricPoint point;  ///< Full definition, context, and value status.
};

/** @brief Point page retaining nonfinite/missing semantics and context. */
struct RunMetricPage {
  std::vector<StoredRunMetricPoint> points;  ///< Increasing commit/index order.
  std::uint64_t snapshot_through = 0U;  ///< Fixed high watermark for this scan.
  std::optional<RunMetricCursor> next_after;  ///< Absent at end of scan.
};

/** @brief Event sink independent of metadata RunSink and the training loop.
 *
 * The whole-batch key scopes to one run and identifies ordered immutable input.
 * It does not replace foundation TelemetryIdentity. Streams start at one and
 * remain contiguous; store-assigned run sequences define pagination only.
 */
class RunEventSink {
 public:
  /** @brief Allows destruction through this narrow event contract. */
  virtual ~RunEventSink() = default;
  /** @brief Atomically commits or exactly replays one bounded batch.
   * @param run_id Existing correlated run identity.
   * @param batch_key Stable whole-batch idempotency identity within the run.
   * @param envelopes Immutable producer envelopes in delivery order.
   * @param stop Cooperative cancellation, including lock waiting.
   * @param deadline Absolute steady-clock deadline.
   * @return Complete positional receipts after synchronous durable commit.
   * @throws CompatibilityError On gaps or conflicting identities/definitions.
   * @throws InvalidArgument On new evidence after a run/stream terminal.
   * @throws Error On malformed input, quota, cancellation/deadline, or I/O.
   */
  virtual RunEventAppendResult append(
      std::string_view run_id, std::string_view batch_key,
      std::span<const TelemetryEnvelope> envelopes, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) = 0;
};

}  // namespace jyppx::experiment

#endif  // JYPPX_EXPERIMENT_RUN_EVENTS_HPP
