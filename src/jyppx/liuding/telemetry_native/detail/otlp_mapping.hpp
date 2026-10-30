// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_PRIVATE_OTLP_MAPPING_HPP
#define JYPPX_LIUDING_PRIVATE_OTLP_MAPPING_HPP

#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::experiment::detail {

// Private preparation module, not an available network exporter. The real
// cancellable mature HTTP SDK implementation is a separate qualification gate.
using OtlpMappingOptions = OtlpProjectionOptions;

// Empty means explicitly unsupported. A metric event returns metrics then
// its complete audit log, both encoded before the caller can send either.
void validate_otlp_mapping_options(const OtlpMappingOptions& options);
std::vector<PreparedOtlpRequest> prepare_otlp_requests(
    const TelemetryEnvelope& envelope, const OtlpMappingOptions& options,
    std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt);

// Official generated response decoder. Unknown fields follow official nanopb
// semantics; malformed protobuf or nonzero rejected counts/nonempty error text
// fail with redacted errors. HTTP status is checked by the future SDK adapter.
void validate_otlp_reply(OtlpSignal signal, std::span<const std::byte> bytes,
                         std::size_t maximum);

std::string_view otlp_signal_path(OtlpSignal signal);

}  // namespace jyppx::liuding::experiment::detail
#endif
