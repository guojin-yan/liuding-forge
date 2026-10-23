// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_EXPERIMENT_TELEMETRY_EXPORTERS_HPP
#define JYPPX_LIUDING_EXPERIMENT_TELEMETRY_EXPORTERS_HPP
/** @file telemetry_exporters.hpp
 * @brief Draft neutral JSONL and optional official-library sidecar exporters.
 * @note Exporters are observers, not the durable run-event source of truth.
 * Neither this API nor its implementation includes a training header.
 */

#include <jyppx/liuding/foundation/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>

namespace jyppx::liuding::experiment {

/** @brief Shared bounded JSON projection used by JSONL and sidecar transport.
 * @param envelope Fully validated neutral observation.
 * @param limits Field/collection/canonical byte bounds.
 * @return Compact JSON with symbolic statuses and full definition provenance.
 * @note Canonical foundation bytes remain persistence/hash truth. This JSON is
 * a projection; NaN/Inf are status fields with a null numerical value.
 */
std::string telemetry_projection_json(const TelemetryEnvelope& envelope,
                                      const TelemetryLimits& limits = {});

/** @brief Limits and identity for a single-owner local JSONL observer. */
struct JsonlTelemetryOptions {
  std::filesystem::path path;  ///< Dedicated regular-file output path.
  std::string sink_id = "telemetry.jsonl";  ///< Stable fanout identity.
  std::string destination_id =
      "local.jsonl";       ///< Redacted destination identity.
  TelemetryLimits limits;  ///< Underlying observation bounds.
  std::uint64_t max_file_bytes = 1ULL << 30U;  ///< Recovery/append file bound.
  std::size_t max_records = 1000000U;  ///< Bounded recovery/dedup index count.
  std::size_t max_line_bytes =
      4U * 1024U * 1024U;  ///< Encoded record line bound.
};

/** @brief Complete-record recovery observations from opening a JSONL file. */
struct JsonlTelemetryRecovery {
  std::size_t complete_records = 0U;        ///< Verified full records retained.
  std::uint64_t truncated_tail_bytes = 0U;  ///< Incomplete final line removed.
};

/** @brief Single-owner JSONL observer with verified restart/replay recovery.
 *
 * Records contain foundation canonical bytes as lowercase hex, their SHA-256,
 * and a verified JSON projection. Opening rejects corrupt complete records and
 * removes only an incomplete final line. Callers must grant exclusive ownership
 * of the path; concurrent processes/writers are not supported. The underlying
 * stream is flushed on successful emit; filesystem fsync durability is not
 * claimed. Use SQLite's evidence sink for acknowledged durable run facts.
 *
 * A failed write poisons this handle; callers reopen and recover before retry.
 * The class owns its stream and dedup index; no destructor flush is promised.
 */
class JsonlTelemetrySink final : public TelemetrySink {
 public:
  /** @brief Recovers a bounded existing file and opens it for append.
   * @param options Identity, file path, and explicit capacity bounds.
   * @throws Error On corrupt complete records, nonregular paths, or limits/I/O.
   */
  explicit JsonlTelemetrySink(JsonlTelemetryOptions options);
  /** @brief Releases handles; use explicit close to observe I/O failures. */
  ~JsonlTelemetrySink() override;
  /** @brief Active stream ownership cannot be copied. */
  JsonlTelemetrySink(const JsonlTelemetrySink&) = delete;
  /** @brief Active stream ownership cannot be reassigned. */
  JsonlTelemetrySink& operator=(const JsonlTelemetrySink&) = delete;
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
  /** @brief Returns opening-time verified/truncated record counts.
   * @return Detached immutable recovery observations.
   */
  JsonlTelemetryRecovery recovery() const noexcept;

 private:
  /** @brief Owns the bounded stream/recovery index and poison/close state. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Owning private state.
};

/** @brief Process-local JSON request/reply channel supplied by a tool runner.
 *
 * The runner starts a locked official-library Python sidecar using an argv
 * vector, an isolated working directory, no shell, bounded stdin/stdout/stderr,
 * and process-level timeout/termination. It supplies the backend/output config
 * at process creation, never from an event payload. This seam prevents the
 * library from inventing a second cross-platform subprocess implementation.
 */
class TelemetrySidecarTransport {
 public:
  /** @brief Enables destruction through the process transport interface. */
  virtual ~TelemetrySidecarTransport() = default;
  /** @brief Exchanges exactly one bounded request/reply JSON line.
   * @param request_json Complete protocol-v1 UTF-8 request without a newline.
   * @param stop Cooperative stop request propagated to the child process.
   * @param deadline Absolute operation bound propagated to the process runner.
   * @return One bounded reply line, without interleaved child diagnostics.
   * @throws Error On process/protocol/I/O/cancel/deadline failure.
   */
  virtual std::string transact(std::string_view request_json,
                               std::stop_token stop,
                               TelemetryDeadline deadline) = 0;
};

/** @brief Exact official-library adapter selected when launching the sidecar.
 */
enum class OfficialTelemetryBackend : std::uint8_t {
  tensorboard = 1,  ///< Official TensorBoard EventFileWriter/generated schemas.
  opentelemetry =
      2,  ///< Official OTel generated protobuf to OTLP/HTTP collector.
};

/** @brief Bounded sidecar protocol configuration, without credentials. */
struct OfficialTelemetryOptions {
  OfficialTelemetryBackend backend =
      OfficialTelemetryBackend::tensorboard;      ///< SDK.
  std::string sink_id = "telemetry.tensorboard";  ///< Stable logical sink ID.
  std::string destination_id =
      "local.tensorboard";  ///< Redacted destination ID.
  TelemetryLimits limits;   ///< Neutral input bounds.
  std::size_t max_request_bytes =
      16U * 1024U * 1024U;                     ///< Protocol line bytes.
  std::size_t max_reply_bytes = 128U * 1024U;  ///< Protocol receipt bytes.
};

/** @brief Optional observer backed by a locked official-library Python process.
 *
 * The transport owns process supervision. This class owns a shared transport
 * reference, validates all neutral inputs and complete receipts, and keeps
 * accepted observations immutable across request retries. Backend errors or
 * unsupported observations remain explicit; core does not depend on Python.
 */
class OfficialTelemetrySink final : public TelemetrySink {
 public:
  /** @brief Binds a runner-owned official sidecar to one logical destination.
   * @param transport Nonnull serial request/reply transport for this backend.
   * @param options Backend, identity, and input/protocol bounds.
   */
  OfficialTelemetrySink(std::shared_ptr<TelemetrySidecarTransport> transport,
                        OfficialTelemetryOptions options = {});
  /** @brief Releases references; explicit close performs fallible finalization.
   */
  ~OfficialTelemetrySink() override;
  /** @brief Transport protocol state cannot be copied. */
  OfficialTelemetrySink(const OfficialTelemetrySink&) = delete;
  /** @brief Transport protocol state cannot be replaced. */
  OfficialTelemetrySink& operator=(const OfficialTelemetrySink&) = delete;
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

 private:
  /** @brief Holds serialized protocol ownership, limits, and close counters. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Owning implementation.
};

}  // namespace jyppx::liuding::experiment

#endif  // JYPPX_LIUDING_EXPERIMENT_TELEMETRY_EXPORTERS_HPP