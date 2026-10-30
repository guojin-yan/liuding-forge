// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "otlp_preflight.hpp"

#include "otlp_mapping.hpp"

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <utility>

namespace jyppx::liuding::experiment::detail {
namespace {
std::uint64_t index_bytes(
    const std::pair<std::string, std::string>& key) noexcept {
  return key.first.size() + key.second.size() + std::uint64_t{128U};
}
}  // namespace

OtlpDeliveryState::OtlpDeliveryState(OtlpPreflightOptions options,
                                     OtlpRequestSender& sender)
    : options_(std::move(options)), sender_(sender) {
  validate_otlp_mapping_options(options_.mapping);
  validate_telemetry_sink_descriptor(options_.descriptor,
                                     options_.mapping.limits);
  if (options_.descriptor.role != TelemetrySinkRole::observer ||
      options_.max_seen == 0U || options_.max_seen > 100000U ||
      options_.max_seen_bytes == 0U ||
      options_.max_seen_bytes > 32U * 1024U * 1024U ||
      options_.max_batch_prepared_bytes == 0U ||
      options_.max_batch_prepared_bytes > 32U * 1024U * 1024U)
    throw InvalidArgument("Invalid native OTLP preflight options");
}

TelemetryEmitResult OtlpDeliveryState::emit(
    std::span<const TelemetryEnvelope> envelopes, std::stop_token stop,
    TelemetryDeadline deadline) {
  check_telemetry_operation(stop, deadline);
  if (closed_) throw NotReady("Native OTLP delivery state is closed");
  validate_telemetry_batch(envelopes, options_.mapping.limits);
  struct Prepared {
    Key key;
    std::vector<PreparedOtlpRequest> requests;
  };
  Index staged;
  std::vector<Prepared> prepared;
  prepared.reserve(envelopes.size());
  TelemetryEmitResult result;
  result.dispositions.reserve(envelopes.size());
  std::uint64_t staged_bytes = 0U;
  std::size_t prepared_bytes = 0U;
  auto mapping = options_.mapping;
  for (const auto& envelope : envelopes) {
    check_telemetry_operation(stop, deadline);
    Key key{envelope.identity.stream_id, envelope.identity.idempotency_key};
    const auto content =
        telemetry_content_digest(envelope, options_.mapping.limits).hex;
    const auto projection =
        telemetry_projection_json(envelope, options_.mapping.limits);
    const auto fingerprint =
        sha256_hex({reinterpret_cast<const std::byte*>(projection.data()),
                    projection.size()});
    const auto existing = seen_.find(key);
    const auto pending = staged.find(key);
    const auto* known = existing != seen_.end()   ? &existing->second
                        : pending != staged.end() ? &pending->second
                                                  : nullptr;
    if (known) {
      if (known->content_digest != content ||
          known->projection_fingerprint != fingerprint)
        throw CompatibilityError("Conflicting native OTLP operation identity");
      result.dispositions.push_back(known->disposition ==
                                            TelemetryDisposition::accepted
                                        ? TelemetryDisposition::duplicate
                                        : known->disposition);
      continue;
    }
    const auto bytes = index_bytes(key);
    if (seen_.size() + staged.size() + 1U > options_.max_seen ||
        bytes > options_.max_seen_bytes - seen_bytes_ - staged_bytes)
      throw ResourceExhausted("Native OTLP replay index capacity exceeded");
    const auto remaining = options_.max_batch_prepared_bytes - prepared_bytes;
    if (remaining == 0U)
      throw ResourceExhausted("Native OTLP prepared batch capacity exceeded");
    mapping.max_prepared_bytes =
        (std::min)(options_.mapping.max_prepared_bytes, remaining);
    mapping.max_request_bytes = (std::min)(options_.mapping.max_request_bytes,
                                           mapping.max_prepared_bytes);
    auto requests = prepare_otlp_requests(envelope, mapping, stop, deadline);
    for (const auto& request : requests) prepared_bytes += request.body.size();
    const auto disposition = requests.empty()
                                 ? TelemetryDisposition::unsupported
                                 : TelemetryDisposition::accepted;
    staged.emplace(key, Seen{content, fingerprint, disposition});
    staged_bytes += bytes;
    prepared.push_back({std::move(key), std::move(requests)});
    result.dispositions.push_back(disposition);
  }
  validate_telemetry_emit_result(options_.descriptor, envelopes, result,
                                 options_.mapping.limits);
  check_telemetry_operation(stop, deadline);
  for (const auto& item : prepared) {
    check_telemetry_operation(stop, deadline);
    for (const auto& request : item.requests) {
      auto response = sender_.send(request, options_.mapping.max_reply_bytes,
                                   stop, deadline);
      check_telemetry_operation(stop, deadline);
      validate_otlp_reply(request.signal,
                          {reinterpret_cast<const std::byte*>(response.data()),
                           response.size()},
                          options_.mapping.max_reply_bytes);
    }
    check_telemetry_operation(stop, deadline);
    auto node = staged.extract(item.key);
    const auto disposition = node.mapped().disposition;
    seen_bytes_ += index_bytes(node.key());
    seen_.insert(std::move(
        node));  // Preallocated node; no index allocation after external I/O.
    if (disposition == TelemetryDisposition::accepted)
      ++counters_.exported_events;
    else
      ++counters_.unsupported_events;
  }
  return result;
}

TelemetryFlushResult OtlpDeliveryState::flush(
    std::stop_token stop, TelemetryDeadline deadline) const {
  if (!closed_) check_telemetry_operation(stop, deadline);
  return counters_;  // Sender is synchronous; there is no hidden queue to
                     // drain.
}

TelemetryFlushResult OtlpDeliveryState::close(std::stop_token stop,
                                              TelemetryDeadline deadline) {
  if (!closed_) {
    check_telemetry_operation(stop, deadline);
    closed_ = true;
  }
  return counters_;
}

}  // namespace jyppx::liuding::experiment::detail
