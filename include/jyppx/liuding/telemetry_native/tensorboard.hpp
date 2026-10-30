// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_TELEMETRY_NATIVE_TENSORBOARD_HPP
#define JYPPX_LIUDING_TELEMETRY_NATIVE_TENSORBOARD_HPP
/** @file tensorboard.hpp
 * @brief Native C++ TensorBoard event observer without a Python runtime.
 */

#include <jyppx/liuding/foundation/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace jyppx::liuding::experiment {

/** @brief Exclusive output ownership, logical identity and resource ceilings.
 */
struct TensorBoardTelemetryOptions {
  std::filesystem::path directory;  ///< New directory; its parent must exist.
  std::string sink_id = "telemetry.tensorboard-native";  ///< Fanout identity.
  std::string destination_id = "local.tensorboard-native";  ///< Redacted ID.
  TelemetryLimits limits;  ///< Existing neutral envelope and batch bounds.
  std::uint64_t max_file_bytes = 1ULL << 30U;  ///< Includes TFRecord framing.
  std::size_t max_record_bytes = 4U * 1024U * 1024U;  ///< One Event protobuf.
  std::size_t max_batch_encoded_bytes = 32U * 1024U * 1024U;  ///< Staged bytes.
  std::size_t max_seen = 100000U;  ///< Unique accepted/unsupported identities.
  std::uint64_t max_seen_bytes = 32U * 1024U * 1024U;  ///< Logical index bytes.
};

/** @brief Synchronous native TensorBoard Event/Summary TFRecord writer.
 *
 * The owner serializes calls and reserves a private new output directory. The
 * writer supports lifecycle text, scalar metric batches and resource samples.
 * The mapping, series tags and step domains match the existing official Python
 * TensorBoard adapter. Missing wall time, unsupported payloads, unsigned steps
 * beyond int64, and standalone batches containing missing values receive an
 * explicit unsupported receipt. A lifecycle preserves missing metrics in its
 * complete text projection and writes no scalar for those missing values.
 *
 * All inputs, identities, capacities and encoded records are preflighted before
 * any batch write. Retry identity/content checks apply within this handle only;
 * a new handle always creates a new directory. No resume, crash recovery,
 * exactly-once external effect or filesystem fsync durability is claimed. Keep
 * the durable run sink as the source of truth and replay into a fresh
 * directory. An I/O failure or cancellation/deadline after writing starts
 * poisons the handle. Explicit close reports close errors; destruction only
 * releases the native file handle and never performs a fallible flush.
 *
 * Writes have no hidden application queue. Successful emit/flush means the OS
 * accepted the bytes, like EventFileWriter.flush; it is not a disk sync. Stop
 * and deadline are checked around bounded writes, but a synchronous OS file
 * call may return after its deadline. A late return is detected and poisons the
 * handle instead of reporting success. Index byte limits describe logical
 * fields, not allocator overhead or an RSS guarantee.
 */
class TensorBoardTelemetrySink final : public TelemetrySink {
 public:
  /** @brief Exclusively creates the output directory and initial event file.
   * @param options Positive capacities, identity and a new private directory.
   * @throws Error On invalid bounds, an existing path, or filesystem failure.
   */
  explicit TensorBoardTelemetrySink(TensorBoardTelemetryOptions options);
  /** @brief Releases handles; use close to observe finalization errors. */
  ~TensorBoardTelemetrySink() override;
  /** @brief File ownership cannot be copied. */
  TensorBoardTelemetrySink(const TensorBoardTelemetrySink&) = delete;
  /** @brief File ownership cannot be reassigned. */
  TensorBoardTelemetrySink& operator=(const TensorBoardTelemetrySink&) = delete;
  /** @copydoc TelemetrySink::descriptor */
  const TelemetrySinkDescriptor& descriptor() const noexcept override;
  /** @copydoc TelemetrySink::emit */
  TelemetryEmitResult emit(std::span<const TelemetryEnvelope> envelopes,
                           std::stop_token stop = {},
                           TelemetryDeadline deadline = std::nullopt) override;
  /** @copydoc TelemetrySink::flush */
  TelemetryFlushResult flush(
      std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;
  /** @copydoc TelemetrySink::close */
  TelemetryFlushResult close(
      std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;
  /** @brief Returns the owned file path for offline TensorBoard consumers.
   * @return A borrowed path valid for this sink's lifetime.
   */
  const std::filesystem::path& event_file() const noexcept;

 private:
  /** @brief Owns the exclusive file, indexes, capacities and failure state. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Exclusive native file and bounded indexes.
};

}  // namespace jyppx::liuding::experiment

#endif  // JYPPX_LIUDING_TELEMETRY_NATIVE_TENSORBOARD_HPP
