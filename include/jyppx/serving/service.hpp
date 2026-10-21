// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
#ifndef JYPPX_SERVING_SERVICE_HPP
#define JYPPX_SERVING_SERVICE_HPP
/** @file service.hpp @brief Unary transport-neutral protocol adapter. */
#include <jyppx/serving/batcher.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace jyppx::serving {
/** @brief Stable public status categories, without backend exception text. */
enum class ServiceStatus : std::uint8_t {
  ok = 0,                ///< A validated independent result was encoded.
  invalid_argument = 1,  ///< Malformed request or schema/numeric violation.
  unauthenticated =
      2,  ///< Verified project/principal authentication is absent.
  permission_denied = 3,   ///< Host authorization callback denied this scope.
  resource_exhausted = 4,  ///< Admission, byte or tensor budget was exhausted.
  cancelled = 5,           ///< Caller cooperatively cancelled the request.
  deadline_exceeded = 6,   ///< Caller cooperative deadline elapsed.
  unavailable = 7,  ///< Runtime is unready or cannot execute this capability.
  internal = 8      ///< An unexpected error occurred; details remain private.
};
/** @brief Returns fixed public status text without backend exception details.
 */
std::string_view service_status_name(ServiceStatus status) noexcept;
/** @brief Host-authenticated security scope; never decoded from model input. */
struct TransportContext {
  RequestContext correlation;  ///< Redacted host-verified identity/correlation.
  bool authenticated = false;  ///< Host verified the caller credentials.
  Deadline deadline;           ///< Cooperative unary request deadline.
};
/** @brief Explicit trusted protocol/schema ownership at the host boundary.
 * @note Codecs must bound allocations during decoding/encoding. Encoded size is
 * checked before decode and after encode; this does not constrain codec RSS.
 * The host owns TLS/authentication and must derive context from verified
 * credentials, without passing raw tokens into RequestContext or telemetry.
 */
struct ProtocolAdapter {
  std::string protocol_id;  ///< Explicit host codec/protocol identity.
  std::uint32_t schema_version =
      1U;  ///< Positive host protocol schema version.
  std::function<nn::Invocation(std::span<const std::byte>)>
      decode;  ///< Trusted bounded request decoder.
  std::function<std::vector<std::byte>(const nn::InvocationResult&)>
      encode;  ///< Trusted bounded result encoder.
  std::function<bool(const RequestContext&)>
      authorize;  ///< Required host authorization decision.
};
/** @brief Positive encoded byte ceilings at the protocol boundary. */
struct ServiceOptions {
  std::size_t max_request_bytes = 1U << 20U;  ///< Checked before decoder entry.
  std::size_t max_response_bytes = 1U
                                   << 20U;  ///< Checked after encoder return.
};
/** @brief Bounded unary reply with fixed public status text and correlation. */
struct ServiceReply {
  ServiceStatus status = ServiceStatus::internal;  ///< Stable public outcome.
  std::string request_id;  ///< Validated original request correlation.
  std::string
      message;  ///< Fixed public status text, never backend diagnostics.
  std::vector<std::byte> body;  ///< Owned encoded result, empty on failure.
  TimingBreakdown timing;       ///< Local decode/inference/encode measurements.
};
/** @brief Readiness is false during draining/degraded/closed lifecycle. */
struct ServiceHealth {
  bool live = true;  ///< Adapter object remains available for health requests.
  bool ready = false;  ///< Predictor is ready and scheduler accepts admission.
  bool streaming = false;  ///< Always false for this unary Preview.
  ModelRuntimeInfo
      model;  ///< Owning predictor identity and lifecycle snapshot.
};
/** @brief Unary protocol bridge; network listeners are host-owned.
 * @note The adapter retains its predictor. A supplied scheduler must own the
 * same predictor; this is checked, not inferred from a model-version string.
 * handle is synchronous. Member cancellation returns independently while a
 * scheduled batch may continue for peers; native code/kernels are cooperative.
 */
class InferenceServiceAdapter final {
 public:
  /** @brief Binds trusted codecs and an optional scheduler for the same owner.
   */
  InferenceServiceAdapter(std::shared_ptr<Predictor> predictor,
                          ProtocolAdapter protocol, ServiceOptions options = {},
                          std::shared_ptr<DynamicBatcher> batcher = {});
  /** @brief Authenticates/authorizes, decodes, predicts and encodes one call.
   */
  ServiceReply handle(std::span<const std::byte> body,
                      const TransportContext& context,
                      std::stop_token stop = {}) const;
  /** @brief Returns synchronized runtime and scheduler readiness. */
  ServiceHealth health() const;
  /** @brief Drains scheduler before closing predictor; false retains ownership.
   */
  bool close(Deadline deadline = {}, std::stop_token stop = {});

 private:
  std::shared_ptr<Predictor> predictor_;  ///< Retained serialized model owner.
  ProtocolAdapter
      protocol_;  ///< Immutable trusted host codec/authorization hooks.
  ServiceOptions options_;  ///< Immutable encoded-size limits.
  std::shared_ptr<DynamicBatcher>
      batcher_;  ///< Optional shared bounded scheduler.
};
}  // namespace jyppx::serving
#endif
