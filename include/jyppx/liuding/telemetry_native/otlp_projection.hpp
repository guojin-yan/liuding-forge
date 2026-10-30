// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TELEMETRY_NATIVE_OTLP_PROJECTION_HPP
#define JYPPX_LIUDING_TELEMETRY_NATIVE_OTLP_PROJECTION_HPP
/** @file otlp_projection.hpp
 * @brief C++ OTLP protobuf preparation without network side effects.
 */

#include <jyppx/liuding/foundation/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::experiment {

/** @brief Official OTLP/HTTP signal family. */
enum class OtlpSignal { metrics, logs, traces };

/** @brief Explicit anchor relating one producer monotonic clock to Unix time.
 */
struct OtlpClockAnchor {
  std::uint64_t monotonic_ns = 0U;  ///< Captured producer monotonic time.
  std::uint64_t unix_ns = 0U;  ///< Corresponding captured Unix nanoseconds.
};

/** @brief Source identity, explicit clocks and bounded protocol preparation. */
struct OtlpProjectionOptions {
  std::string
      resource_instance_id;  ///< Explicit instance; never inferred from events.
  std::map<std::string, OtlpClockAnchor>
      clock_mappings;      ///< At most 64 captured anchors.
  TelemetryLimits limits;  ///< May narrow the default neutral envelope bounds.
  std::size_t max_request_bytes =
      16U * 1024U * 1024U;  ///< One official request.
  std::size_t max_prepared_bytes =
      32U * 1024U * 1024U;  ///< Sum of one event's requests.
  std::size_t max_reply_bytes = 1024U * 1024U;  ///< One official response.
};

/** @brief One complete, owned official protobuf request; no queued I/O. */
struct PreparedOtlpRequest {
  OtlpSignal signal;  ///< Selects the corresponding OTLP/HTTP path.
  std::string body;   ///< Binary application/x-protobuf request bytes.
};

/** @brief Prepares official metrics/logs/traces and checks collector protobuf.
 *
 * This codec performs no network I/O and does not acknowledge delivery. Empty
 * encode output means explicitly unsupported. Metric/resource observations and
 * metric-bearing lifecycle events produce a metrics request followed by a full
 * audit-log request, both prepared before return. Other lifecycle/log/artifact
 * events produce logs; spans require a matching explicit clock anchor.
 *
 * Native IDs derive from complete opaque IDs, missing data remains missing,
 * metric resources omit changing run identities, and numeric points keep their
 * declared definitions/context/units. A downstream sender must enforce the full
 * URL/TLS/deadline/cancellation/status/partial-success contract. A successful
 * response check is only the protobuf acknowledgement check for one request;
 * it is not proof of HTTP status, complete event delivery, retries or
 * durability.
 */
class OtlpProjectionEncoder final {
 public:
  /** @brief Validates immutable identity, clock and capacity options. */
  explicit OtlpProjectionEncoder(OtlpProjectionOptions options);
  /** @brief Prepares all requests for one immutable neutral envelope. */
  std::vector<PreparedOtlpRequest> encode(
      const TelemetryEnvelope& envelope, std::stop_token stop = {},
      TelemetryDeadline deadline = std::nullopt) const;
  /** @brief Rejects malformed/oversized protobuf and any collector partial
   * success. */
  void validate_response(OtlpSignal signal,
                         std::span<const std::byte> bytes) const;
  /** @brief Returns the fixed official path for a signal, without an endpoint.
   */
  static std::string_view signal_path(OtlpSignal signal);

 private:
  OtlpProjectionOptions options_;  ///< Immutable mapping configuration.
};

}  // namespace jyppx::liuding::experiment
#endif
