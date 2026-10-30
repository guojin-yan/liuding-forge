// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "detail/otlp_mapping.hpp"

#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

#include <utility>

namespace jyppx::liuding::experiment {

OtlpProjectionEncoder::OtlpProjectionEncoder(OtlpProjectionOptions options)
    : options_(std::move(options)) {
  detail::validate_otlp_mapping_options(options_);
}

std::vector<PreparedOtlpRequest> OtlpProjectionEncoder::encode(
    const TelemetryEnvelope& envelope, std::stop_token stop,
    TelemetryDeadline deadline) const {
  return detail::prepare_otlp_requests(envelope, options_, stop, deadline);
}

void OtlpProjectionEncoder::validate_response(
    OtlpSignal signal, std::span<const std::byte> bytes) const {
  detail::validate_otlp_reply(signal, bytes, options_.max_reply_bytes);
}

std::string_view OtlpProjectionEncoder::signal_path(OtlpSignal signal) {
  return detail::otlp_signal_path(signal);
}

}  // namespace jyppx::liuding::experiment
