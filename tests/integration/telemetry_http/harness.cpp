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
#include <string>
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
    const bool queue_full = mode.starts_with("queue-full-");
    const auto signal = queue_full ? mode.substr(11U) : mode;
    unsigned int port = 0U;
    const auto parsed = std::from_chars(
        port_text.data(), port_text.data() + port_text.size(), port);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != port_text.data() + port_text.size() || port == 0U ||
        port > 65535U)
      return 2;
    ex::OtlpHttpOptions options;
    options.endpoint = {.host = mode == "ipv6-logs" ? "::1" : "127.0.0.1",
                        .port = static_cast<std::uint16_t>(port),
                        .tls = mode == "tls-reject" ||
                               mode == "tls-certificate-reject" ||
                               mode == "tls-logs"};
    options.mapping.resource_instance_id = "http.integration.instance";
    options.mapping.max_reply_bytes = 64U;
    options.request_timeout =
        std::chrono::milliseconds(mode == "slow-response" ? 100 : 2000);
    if (mode == "max-seen") options.max_seen = 1U;
    if (signal == "traces")
      options.mapping.clock_mappings = {
          {"producer", {10U, 1700000000000000000ULL}}};
    ex::OtlpHttpTelemetrySink sink(options);
    std::vector<forge::TelemetryEnvelope> batch{log_event(1U)};
    if (queue_full) {
      require(signal == "logs" || signal == "metrics" || signal == "traces");
      for (std::uint64_t sequence = 2U; sequence <= 16U; ++sequence)
        batch.push_back(log_event(sequence));
    }
    for (auto& event : batch) {
      if (mode == "pair-retry" || signal == "metrics") {
        forge::TelemetryMetricPoint point;
        point.definition.metric_id = "http.integration.loss";
        point.observation = forge::make_telemetry_metric_value(
            0.125 * static_cast<double>(event.identity.sequence));
        event.payload = forge::TelemetryMetricBatch{{point}};
      } else if (signal == "traces") {
        const auto suffix =
            queue_full ? "-" + std::to_string(event.identity.sequence) : "";
        event.correlation = {.trace_id = "opaque-trace" + suffix,
                             .span_id = "opaque-span" + suffix};
        // A completed span is observed at or after its end in the same clock.
        event.time.monotonic_time_ns = 20U;
        event.time.clock_id = "producer";
        event.payload =
            forge::TelemetrySpan{.name = "transport-span",
                                 .start_monotonic_ns = 10U,
                                 .end_monotonic_ns = 20U,
                                 .status = forge::TelemetrySpanStatus::ok};
      }
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(queue_full ? 40 : 8);
    if (queue_full) {
      std::string rejected_status;
      try {
        sink.emit(batch, {}, deadline);
      } catch (const forge::IoError& error) {
        if (error.message() == "OTLP collector returned HTTP status 503")
          rejected_status = "503";
        else if (error.message() == "OTLP collector returned HTTP status 429")
          rejected_status = "429";
        else
          throw;
      }
      require(!rejected_status.empty());
      const auto prefix = sink.flush().exported_events;
      require(prefix > 0U && prefix < batch.size());
      std::cout << "native-http-queue-full " << prefix << ' ' << batch.size()
                << ' ' << rejected_status << std::endl;
      // The supervising fixture drains the real Collector before retrying.
      // Its process deadline also bounds this synchronization read.
      std::string command;
      require(static_cast<bool>(std::getline(std::cin, command)) &&
              command == "retry");
      const auto result = sink.emit(batch, {}, deadline);
      require(result.dispositions.size() == batch.size());
      for (std::size_t i = 0U; i < batch.size(); ++i)
        require(result.dispositions[i] ==
                (i < prefix ? forge::TelemetryDisposition::duplicate
                            : forge::TelemetryDisposition::accepted));
      const auto repeated = sink.emit(batch, {}, deadline);
      require(repeated.dispositions.size() == batch.size());
      for (const auto value : repeated.dispositions)
        require(value == forge::TelemetryDisposition::duplicate);
      require(sink.close().exported_events == batch.size());
    } else if (mode == "invalid-batch" || mode == "max-seen") {
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
    } else if (mode == "tls-certificate-reject") {
      for (int attempt = 0; attempt < 2; ++attempt) {
        bool certificate_rejected = false;
        try {
          sink.emit(batch, {}, deadline);
        } catch (const forge::IoError& error) {
          require(error.message() ==
                  "OTLP HTTPS certificate validation failed");
          require(error.details().component == "telemetry_http");
          require(error.details().operation == "verify_peer_certificate");
          require(!error.details().actual.empty());
          std::cout << "native-http-certificate-rejected: "
                    << error.details().actual << '\n';
          certificate_rejected = true;
        }
        // Retry the same identity through the same sink: neither attempt may
        // advance the process-local acknowledgement or become a duplicate.
        require(certificate_rejected);
        require(sink.flush().exported_events == 0U);
      }
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
        require(mode == "logs" || mode == "metrics" || mode == "traces" ||
                mode == "ipv6-logs" || mode == "tls-logs");
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
