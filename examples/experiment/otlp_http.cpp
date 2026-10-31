// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 5 && argc != 6) {
      std::cerr << "Usage: liuding-forge-example-otlp-http HOST PORT "
                   "http|https INSTANCE_ID [NEW_RUN_ROOT]\n";
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
    const auto monotonic_now = [] {
      return static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
    };
    const auto start = monotonic_now();
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    const std::array predictions{0.0, 0.5, 1.0, 1.5};
    const std::array targets{0.0, 0.75, 0.75, 1.5};
    double absolute_error = 0.0;
    for (std::size_t i = 0U; i < predictions.size(); ++i)
      absolute_error += std::abs(predictions[i] - targets[i]);
    const auto mean_absolute_error =
        absolute_error / static_cast<double>(predictions.size());
    const auto end = monotonic_now();
    experiment::OtlpHttpOptions options;
    options.endpoint = {.host = argv[1],
                        .port = static_cast<std::uint16_t>(port),
                        .tls = scheme == "https"};
    options.mapping.resource_instance_id = argv[4];
    options.mapping.clock_mappings.emplace(
        "example.clock",
        experiment::OtlpClockAnchor{start, static_cast<std::uint64_t>(wall)});
    options.descriptor.destination_id = "example.collector";
    experiment::OtlpHttpTelemetrySink sink(std::move(options));
    forge::TelemetryMetricPoint point;
    point.definition.metric_id = "example.validation.mean_absolute_error";
    point.step.value = 1U;
    point.observation = forge::make_telemetry_metric_value(mean_absolute_error);
    const forge::TelemetryCorrelation correlation{
        .project_id = "example",
        .run_id = "run.otlp-example",
        .attempt_id = "attempt.1",
        .trace_id = "example.validation",
        .span_id = "example.mae"};
    const forge::TelemetryTime time{.wall_time_unix_ns = wall,
                                    .monotonic_time_ns = end,
                                    .clock_id = "example.clock"};
    const std::array<forge::TelemetryEnvelope, 3> batch{
        {{.identity = {.stream_id = "example.otlp-http",
                       .event_id = "loss.1",
                       .idempotency_key = "loss.1",
                       .sequence = 1U},
          .time = time,
          .correlation = correlation,
          .payload = forge::TelemetryMetricBatch{{point}}},
         {.identity = {.stream_id = "example.otlp-http",
                       .event_id = "log.2",
                       .idempotency_key = "log.2",
                       .sequence = 2U},
          .time = time,
          .correlation = correlation,
          .payload = forge::TelemetryLog{forge::TelemetryLogSeverity::info,
                                         "Computed MAE from four samples"}},
         {.identity = {.stream_id = "example.otlp-http",
                       .event_id = "span.3",
                       .idempotency_key = "span.3",
                       .sequence = 3U},
          .time = time,
          .correlation = correlation,
          .payload =
              forge::TelemetrySpan{.name = "compute.mae",
                                   .start_monotonic_ns = start,
                                   .end_monotonic_ns = end,
                                   .status = forge::TelemetrySpanStatus::ok}}}};
    if (argc == 6) {
      const auto root = std::filesystem::u8path(argv[5]);
      if (!std::filesystem::create_directory(root))
        throw forge::InvalidArgument("Choose a new dedicated run directory");
      experiment::SQLiteRunStore store(root);
      const auto milliseconds = static_cast<std::uint64_t>(wall) / 1000000U;
      store.start({.run_id = correlation.run_id,
                   .project_id = correlation.project_id,
                   .experiment_id = "otlp-example",
                   .display_name = "Four-sample MAE",
                   .status = experiment::RunStatus::running,
                   .created_at_unix_ms = milliseconds,
                   .updated_at_unix_ms = milliseconds});
      store.append(correlation.run_id, "computed-observations", batch);
      store.transition(correlation.run_id, experiment::RunStatus::completed,
                       milliseconds);
      std::cout
          << "Committed 3 envelopes to run.otlp-example before network I/O.\n";
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    std::cout << "Computed validation MAE from 4 samples: "
              << mean_absolute_error << '\n';
    const auto first = sink.emit(batch, {}, deadline);
    const auto repeated = sink.emit(batch, {}, deadline);
    const auto counters = sink.close({}, deadline);
    if (first.dispositions !=
            std::vector(3U, forge::TelemetryDisposition::accepted) ||
        repeated.dispositions !=
            std::vector(3U, forge::TelemetryDisposition::duplicate) ||
        counters.exported_events != 3U)
      throw forge::CompatibilityError("Unexpected synchronous OTLP receipts");
    std::cout << "Protocol responses accepted for 3 envelopes (metrics, audit "
                 "log, log, span); "
                 "exact replay was deduplicated in this process.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
