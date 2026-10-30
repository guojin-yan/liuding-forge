// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      std::cerr << "Usage: liuding-forge-example-otlp-http HOST PORT "
                   "http|https INSTANCE_ID\n";
      return 2;
    }
    const std::string_view port_text(argv[2]);
    unsigned int port = 0U;
    const auto parsed = std::from_chars(
        port_text.data(), port_text.data() + port_text.size(), port);
    const std::string_view scheme(argv[3]);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != port_text.data() + port_text.size() || port == 0U ||
        port > 65535U || (scheme != "http" && scheme != "https"))
      throw jyppx::liuding::InvalidArgument(
          "Invalid explicit port or transport");
    namespace forge = jyppx::liuding;
    namespace experiment = forge::experiment;
    experiment::OtlpHttpOptions options;
    options.endpoint = {.host = argv[1],
                        .port = static_cast<std::uint16_t>(port),
                        .tls = scheme == "https"};
    options.mapping.resource_instance_id = argv[4];
    options.descriptor.destination_id = "example.collector";
    experiment::OtlpHttpTelemetrySink sink(std::move(options));

    const std::array predictions{0.0, 0.5, 1.0, 1.5};
    const std::array targets{0.0, 0.75, 0.75, 1.5};
    double absolute_error = 0.0;
    for (std::size_t i = 0U; i < predictions.size(); ++i)
      absolute_error += std::abs(predictions[i] - targets[i]);
    const auto mean_absolute_error =
        absolute_error / static_cast<double>(predictions.size());
    forge::TelemetryMetricPoint point;
    point.definition.metric_id = "example.validation.mean_absolute_error";
    point.step.value = 1U;
    point.observation = forge::make_telemetry_metric_value(mean_absolute_error);
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    const std::array<forge::TelemetryEnvelope, 1> batch{
        {{.identity = {.stream_id = "example.otlp-http",
                       .event_id = "loss.1",
                       .idempotency_key = "loss.1",
                       .sequence = 1U},
          .time = {.wall_time_unix_ns = now},
          .payload = forge::TelemetryMetricBatch{{point}}}}};
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::cout << "Computed validation MAE from 4 samples: "
              << mean_absolute_error << '\n';
    const auto first = sink.emit(batch, {}, deadline);
    const auto repeated = sink.emit(batch, {}, deadline);
    const auto counters = sink.close({}, deadline);
    if (first.dispositions !=
            std::vector{forge::TelemetryDisposition::accepted} ||
        repeated.dispositions !=
            std::vector{forge::TelemetryDisposition::duplicate} ||
        counters.exported_events != 1U)
      throw forge::CompatibilityError("Unexpected synchronous OTLP receipts");
    std::cout
        << "Protocol responses accepted for 1 envelope (metrics + audit log); "
           "exact replay was deduplicated in this process.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
