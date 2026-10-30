// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <iostream>
#include <stop_token>
#include <string_view>
#include <thread>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace ex = forge::experiment;
void require(bool condition) {
  if (!condition)
    throw forge::CompatibilityError("OTLP HTTP harness assertion");
}
forge::TelemetryEnvelope log_event(std::uint64_t sequence) {
  return {.identity = {.stream_id = "http.integration",
                       .event_id = std::to_string(sequence),
                       .idempotency_key = std::to_string(sequence),
                       .sequence = sequence},
          .time = {.wall_time_unix_ns = 1700000000000000000LL},
          .payload = forge::TelemetryLog{forge::TelemetryLogSeverity::info,
                                         "transport lesson"}};
}
template <typename Expected, typename Operation>
void fails(Operation&& operation) {
  bool matched = false;
  try {
    operation();
  } catch (const Expected&) {
    matched = true;
  }
  require(matched);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    const std::string_view port_text(argv[1]), mode(argv[2]);
    unsigned int port = 0U;
    const auto parsed = std::from_chars(
        port_text.data(), port_text.data() + port_text.size(), port);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != port_text.data() + port_text.size() || port == 0U ||
        port > 65535U)
      return 2;
    ex::OtlpHttpOptions options;
    options.endpoint = {.host = "127.0.0.1",
                        .port = static_cast<std::uint16_t>(port),
                        .tls = mode == "tls-reject"};
    options.mapping.resource_instance_id = "http.integration.instance";
    options.mapping.max_reply_bytes = 64U;
    options.request_timeout =
        std::chrono::milliseconds(mode == "slow-response" ? 100 : 2000);
    if (mode == "max-seen") options.max_seen = 1U;
    if (mode == "traces")
      options.mapping.clock_mappings = {
          {"producer", {10U, 1700000000000000000ULL}}};
    ex::OtlpHttpTelemetrySink sink(options);
    std::vector<forge::TelemetryEnvelope> batch{log_event(1U)};
    if (mode == "pair-retry") {
      forge::TelemetryMetricPoint point;
      point.definition.metric_id = "http.integration.loss";
      point.observation = forge::make_telemetry_metric_value(0.125);
      batch[0].payload = forge::TelemetryMetricBatch{{point}};
    } else if (mode == "traces") {
      batch[0].correlation = {.trace_id = "opaque-trace",
                              .span_id = "opaque-span"};
      // A completed span is observed at or after its end in the same clock.
      batch[0].time.monotonic_time_ns = 20U;
      batch[0].time.clock_id = "producer";
      batch[0].payload =
          forge::TelemetrySpan{.name = "transport-span",
                               .start_monotonic_ns = 10U,
                               .end_monotonic_ns = 20U,
                               .status = forge::TelemetrySpanStatus::ok};
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(8);
    if (mode == "invalid-batch" || mode == "max-seen") {
      batch.push_back(log_event(2U));
      if (mode == "invalid-batch") batch[1].identity.stream_id.clear();
      if (mode == "invalid-batch")
        fails<forge::InvalidArgument>([&] { sink.emit(batch, {}, deadline); });
      else
        fails<forge::ResourceExhausted>(
            [&] { sink.emit(batch, {}, deadline); });
      require(sink.close().exported_events == 0U);
    } else if (mode == "cancelled" || mode == "deadline") {
      std::stop_source stop;
      if (mode == "cancelled") {
        stop.request_stop();
        fails<forge::Cancelled>(
            [&] { sink.emit(batch, stop.get_token(), deadline); });
      } else {
        fails<forge::DeadlineExceeded>([&] {
          sink.emit(batch, {},
                    std::chrono::steady_clock::now() - std::chrono::seconds(1));
        });
      }
      require(sink.close().exported_events == 0U);
    } else if (mode == "stop-during-send") {
      std::stop_source stop;
      std::jthread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        stop.request_stop();
      });
      fails<forge::Cancelled>(
          [&] { sink.emit(batch, stop.get_token(), deadline); });
      require(sink.close().exported_events == 0U);
    } else if (mode == "slow-response") {
      fails<forge::DeadlineExceeded>([&] { sink.emit(batch, {}, deadline); });
      require(sink.close().exported_events == 0U);
    } else if (mode == "tls-reject") {
      fails<forge::Error>([&] { sink.emit(batch, {}, deadline); });
      require(sink.close().exported_events == 0U);
    } else {
      if (mode == "prefix-retry") batch.push_back(log_event(2U));
      const bool retry = mode == "reply-retry" || mode == "pair-retry" ||
                         mode == "prefix-retry";
      if (retry) {
        fails<forge::Error>([&] { sink.emit(batch, {}, deadline); });
        require(sink.flush().exported_events ==
                (mode == "prefix-retry" ? 1U : 0U));
      } else {
        require(mode == "logs" || mode == "traces");
      }
      const auto result = sink.emit(batch, {}, deadline);
      for (std::size_t i = 0U; i < batch.size(); ++i)
        require(result.dispositions[i] ==
                (mode == "prefix-retry" && i == 0U
                     ? forge::TelemetryDisposition::duplicate
                     : forge::TelemetryDisposition::accepted));
      const auto repeated = sink.emit(batch, {}, deadline);
      for (const auto value : repeated.dispositions)
        require(value == forge::TelemetryDisposition::duplicate);
      require(sink.close().exported_events == batch.size());
    }
    std::cout << "native-http-contract-ok\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
