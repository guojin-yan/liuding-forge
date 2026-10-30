// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_PRIVATE_OTLP_PREFLIGHT_HPP
#define JYPPX_LIUDING_PRIVATE_OTLP_PREFLIGHT_HPP

#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

#include <map>
#include <utility>

namespace jyppx::liuding::experiment::detail {

// Internal seam only, not a public callback API. Senders return only a complete
// bounded HTTP 200 body and suppress proxies/redirect/auth/cookies. The native
// synchronous HTTP adapter implements TelemetrySink cooperative stop/deadline
// checks, including cleanup before success. It does not satisfy the separate
// process-isolated hard-termination gate required to replace the old CLI
// sidecar.
class OtlpRequestSender {
 public:
  virtual ~OtlpRequestSender() = default;
  virtual std::string send(const PreparedOtlpRequest& request,
                           std::size_t maximum_reply_bytes,
                           std::stop_token stop,
                           TelemetryDeadline deadline) = 0;
};

struct OtlpPreflightOptions {
  OtlpProjectionOptions mapping;
  TelemetrySinkDescriptor descriptor{
      .type_id = "jyppx.telemetry-sink.opentelemetry-native/v1",
      .sink_id = "telemetry.opentelemetry-native",
      .destination_id = "configured.opentelemetry-native",
      .role = TelemetrySinkRole::observer};
  std::size_t max_seen = 100000U;
  std::uint64_t max_seen_bytes = 32U * 1024U * 1024U;
  std::size_t max_batch_prepared_bytes = 32U * 1024U * 1024U;
};

// Synchronous ownership/counter state, deliberately not a background queue.
// Validate and prepare the complete batch before invoking the sender. Only a
// whole metrics+audit pair commits its event identity. A later failure retains
// acknowledged prefix identities; retrying a partly sent pair can duplicate
// its metric and never asserts exactly-once collector effects.
class OtlpDeliveryState final {
 public:
  explicit OtlpDeliveryState(OtlpPreflightOptions options,
                             OtlpRequestSender& sender);
  TelemetryEmitResult emit(std::span<const TelemetryEnvelope> envelopes,
                           std::stop_token stop = {},
                           TelemetryDeadline deadline = std::nullopt);
  TelemetryFlushResult flush(std::stop_token stop = {},
                             TelemetryDeadline deadline = std::nullopt) const;
  TelemetryFlushResult close(std::stop_token stop = {},
                             TelemetryDeadline deadline = std::nullopt);

 private:
  struct Seen {
    std::string content_digest;
    std::string projection_fingerprint;
    TelemetryDisposition disposition;
  };
  using Key = std::pair<std::string, std::string>;
  using Index = std::map<Key, Seen>;
  OtlpPreflightOptions options_;
  OtlpRequestSender& sender_;
  Index seen_;
  std::uint64_t seen_bytes_ = 0U;
  TelemetryFlushResult counters_;
  bool closed_ = false;
};

}  // namespace jyppx::liuding::experiment::detail
#endif
