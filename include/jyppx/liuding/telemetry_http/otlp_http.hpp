// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TELEMETRY_HTTP_OTLP_HTTP_HPP
#define JYPPX_LIUDING_TELEMETRY_HTTP_OTLP_HTTP_HPP

#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace jyppx::liuding::experiment {

/** @brief Explicit collector authority; events cannot select destinations.
 * @note Host is an ASCII DNS name or unbracketed IP literal, without userinfo,
 * paths, scope IDs, query or fragment. Cleartext HTTP is restricted to the
 * literal loopback addresses 127.0.0.1 and ::1. DNS localhost requires TLS.
 */
struct OtlpHttpEndpoint {
  std::string host;  ///< Required; never inferred from an environment variable.
  std::uint16_t port = 4318U;  ///< Explicit nonzero collector port.
  bool tls = true;  ///< System certificate-chain/hostname validation stays on.
};

/** @brief Immutable endpoint, mapping and bounded process-local retry state. */
struct OtlpHttpOptions {
  OtlpHttpEndpoint endpoint;
  OtlpProjectionOptions mapping;
  TelemetrySinkDescriptor descriptor{
      .type_id = "jyppx.telemetry-sink.opentelemetry-http/v1",
      .sink_id = "telemetry.opentelemetry-http",
      .destination_id = "configured.opentelemetry-http",
      .role = TelemetrySinkRole::observer};
  std::chrono::milliseconds request_timeout{30000};  ///< In [1, 30000] ms.
  std::size_t max_seen = 100000U;  ///< May only narrow the default.
  std::uint64_t max_seen_bytes = 32U * 1024U * 1024U;
  std::size_t max_batch_prepared_bytes = 32U * 1024U * 1024U;
};

/** @brief Synchronous C++ OTLP/HTTP observer with no hidden application queue.
 *
 * The whole batch is validated and encoded before network I/O. An envelope is
 * remembered only after all of its requests return HTTP 200 with protobuf
 * content type, no unsupported content coding, and valid official protobuf
 * responses. This is protocol response acknowledgement,
 * not collector durable acknowledgement. A metrics/audit pair is one envelope;
 * retry after a partial pair can duplicate the first request. Keep the durable
 * run store as the source of truth. Seen identities are process-local, bounded
 * and never evicted; exhaustion rejects the batch before sending anything.
 *
 * No environment proxies, redirects, cookies, automatic authentication,
 * credentials, custom headers or application retries are used. HTTPS uses the
 * system trust store. Calls are serialized by the owner. Destruction performs
 * no network I/O. Flush has no pending network work.
 *
 * The caller waits synchronously over asynchronous system I/O. Stop requests
 * wake an idle request wait; the earlier caller deadline/request timeout bounds
 * that wait. Cancellation closes the request without accepting a late reply.
 * On Windows a private 16 KiB read/write chunk remains owned by native
 * callbacks until the final handle-close notification; caller buffers never
 * escape. System cleanup, DNS and OS scheduling do not provide an absolute
 * wall-clock guarantee. This API does not provide process-level termination.
 * Limits bound encoded and logical fields, not total RSS or TLS stack memory.
 */
class OtlpHttpTelemetrySink final : public TelemetrySink {
 public:
  /** @brief Validates options without connecting to the collector. */
  explicit OtlpHttpTelemetrySink(OtlpHttpOptions options);
  /** @brief Releases native state without a fallible flush. */
  ~OtlpHttpTelemetrySink() override;
  OtlpHttpTelemetrySink(const OtlpHttpTelemetrySink&) = delete;
  OtlpHttpTelemetrySink& operator=(const OtlpHttpTelemetrySink&) = delete;

  const TelemetrySinkDescriptor& descriptor() const noexcept override;
  TelemetryEmitResult emit(std::span<const TelemetryEnvelope> envelopes,
                           std::stop_token stop = {},
                           TelemetryDeadline deadline = std::nullopt) override;
  TelemetryFlushResult flush(
      std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;
  TelemetryFlushResult close(
      std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jyppx::liuding::experiment
#endif  // JYPPX_LIUDING_TELEMETRY_HTTP_OTLP_HTTP_HPP
