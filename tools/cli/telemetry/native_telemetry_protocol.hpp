// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_TOOLS_CLI_NATIVE_TELEMETRY_PROTOCOL_HPP
#define JYPPX_TOOLS_CLI_NATIVE_TELEMETRY_PROTOCOL_HPP

#include <jyppx/liuding/foundation/telemetry.hpp>

#include <cstddef>
#include <nlohmann/json.hpp>
#include <string_view>

namespace jyppx::liuding::cli {

// Private CLI wire implementation; no installed header or protocol extension.
nlohmann::json parse_native_telemetry_json(std::string_view text,
                                           std::size_t maximum_bytes);
TelemetryEnvelope decode_native_telemetry_projection(
    const nlohmann::json& value, std::string_view content_digest,
    const TelemetryLimits& limits = {});

// Own one persistent sink externally. Failed emits preserve its accepted
// prefix; request IDs are checked, never cached as whole-batch acknowledgments.
class NativeTelemetryProtocol final {
 public:
  explicit NativeTelemetryProtocol(TelemetrySink& sink,
                                   TelemetryLimits limits = {},
                                   std::string_view backend = "opentelemetry");
  nlohmann::json exchange(const nlohmann::json& request,
                          TelemetryDeadline deadline = std::nullopt);

 private:
  TelemetrySink& sink_;
  TelemetryLimits limits_;
  // Validated literal; never borrowed from configuration or a request.
  std::string_view backend_;
  TelemetryFlushResult counters_;
  bool closed_ = false;
};

}  // namespace jyppx::liuding::cli
#endif
