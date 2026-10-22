// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_EXPERIMENT_RUN_TELEMETRY_SINK_HPP
#define JYPPX_EXPERIMENT_RUN_TELEMETRY_SINK_HPP

/** @file run_telemetry_sink.hpp
 * @brief Draft synchronous evidence adapter for one existing SQLite run.
 */

#include <jyppx/experiment/run_store.hpp>
#include <jyppx/foundation/telemetry.hpp>

#include <memory>
#include <span>
#include <stop_token>
#include <string>

namespace jyppx::experiment {

/** @brief Shared neutral sink backed by atomic SQLite evidence commits.
 *
 * The owner serializes calls, as required by TelemetrySink. No worker queue or
 * training dependency is introduced. Whole-batch keys derive from the ordered
 * producer operation identities, while the store checks immutable content.
 * Counters count unique newly accepted envelopes through this adapter instance.
 * Destruction never flushes or changes run/stream lifecycle state.
 */
class SQLiteRunTelemetrySink final : public TelemetrySink {
 public:
  /** @brief Binds a redacted logical sink destination to one existing run.
   * @param store Shared open SQLite store; must be nonnull.
   * @param run_id Existing run to which all input must correlate.
   * @param sink_id Stable unique logical sink ID in the fanout.
   * @param destination_id Stable redacted destination ID, without a path.
   */
  SQLiteRunTelemetrySink(std::shared_ptr<SQLiteRunStore> store,
                         std::string run_id, std::string sink_id,
                         std::string destination_id);
  /** @brief Releases ownership without fallible flushing. */
  ~SQLiteRunTelemetrySink() override = default;
  /** @brief Avoids duplicating sink counter and close state. */
  SQLiteRunTelemetrySink(const SQLiteRunTelemetrySink&) = delete;
  /** @brief Avoids assigning sink state across logical destinations. */
  SQLiteRunTelemetrySink& operator=(const SQLiteRunTelemetrySink&) = delete;
  /** @brief Returns the stable evidence destination.
   * @return Immutable sink identity and destination metadata.
   */
  const TelemetrySinkDescriptor& descriptor() const noexcept override;
  /** @brief Durably commits a whole batch or exactly replays its retry.
   * @param envelopes Ordered telemetry envelopes to commit.
   * @param stop Cooperative cancellation request.
   * @param deadline Optional absolute operation deadline.
   * @return Durable receipt and accepted-envelope counters.
   */
  TelemetryEmitResult emit(std::span<const TelemetryEnvelope> envelopes,
                           std::stop_token stop = {},
                           TelemetryDeadline deadline = std::nullopt) override;
  /** @brief Returns counters; successful emit has no deferred work to drain.
   * @param stop Cooperative cancellation request.
   * @param deadline Optional absolute operation deadline.
   * @return Flush status and counters.
   */
  TelemetryFlushResult flush(
      std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;
  /** @brief Idempotently records closure and returns final counters.
   * @param stop Cooperative cancellation request.
   * @param deadline Optional absolute operation deadline.
   * @return Final flush status and counters.
   */
  TelemetryFlushResult close(
      std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;

 private:
  std::shared_ptr<SQLiteRunStore> store_;  ///< Connection lifetime ownership.
  std::string run_id_;                     ///< Bound existing run.
  TelemetrySinkDescriptor descriptor_;     ///< Stable redacted destination.
  TelemetryFlushResult counters_{};  ///< Per-instance durable accepted count.
  bool closed_ = false;              ///< Only set after a successful close.
};

}  // namespace jyppx::experiment

#endif  // JYPPX_EXPERIMENT_RUN_TELEMETRY_SINK_HPP
