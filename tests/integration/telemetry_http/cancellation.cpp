// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_http/otlp_http.hpp>

#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stop_token>
#include <string_view>
#include <thread>

namespace {
namespace forge = jyppx::liuding;
namespace ex = forge::experiment;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;

void require(bool condition, const char* message) {
  if (!condition) throw forge::CompatibilityError(message);
}

std::size_t handle_count() {
#ifdef _WIN32
  DWORD count = 0U;
  if (!GetProcessHandleCount(GetCurrentProcess(), &count))
    throw forge::IoError("Cannot observe native process handles");
  return count;
#else
  std::size_t count = 0U;
  for (const auto& entry :
       std::filesystem::directory_iterator("/proc/self/fd")) {
    static_cast<void>(entry);
    ++count;
  }
  return count;
#endif
}

std::size_t settled_handles(std::optional<std::size_t> expected) {
  const auto deadline = Clock::now() + Ms(1000);
  auto previous = handle_count();
  auto stable_since = Clock::now();
  for (;;) {
    std::this_thread::sleep_for(Ms(5));
    const auto count = handle_count();
    if (count != previous) stable_since = Clock::now();
    if ((!expected || count == *expected) &&
        Clock::now() - stable_since >= Ms(100))
      return count;
    previous = count;
    if (Clock::now() >= deadline)
      throw forge::CompatibilityError(
          "Native HTTP handles did not return exactly; expected " +
          std::to_string(expected.value_or(previous)) + ", observed " +
          std::to_string(count));
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) return 2;
    const std::string_view port_text(argv[1]), mode(argv[2]);
    unsigned port = 0U;
    const auto parsed = std::from_chars(
        port_text.data(), port_text.data() + port_text.size(), port);
    require(parsed.ec == std::errc{} &&
                parsed.ptr == port_text.data() + port_text.size() &&
                port > 0U && port <= 65535U,
            "Invalid cancellation fixture port");
    const bool cancelled = mode.starts_with("stop-");
    require(cancelled || mode.starts_with("deadline-"), "Invalid control mode");
    const bool tls = mode.ends_with("tls");
    require(tls || mode.ends_with("headers") || mode.ends_with("body"),
            "Invalid stalled response phase");
    const unsigned measured_rounds = cancelled ? 8U : 2U;
    constexpr unsigned max_warmup_rounds = 32U;
    unsigned warmup_rounds = 0U, measured = 0U, stable_warmups = 0U;
    std::optional<std::size_t> previous_warmup;
    std::optional<std::size_t> baseline;
    for (unsigned round = 0U; measured < measured_rounds; ++round) {
      ex::OtlpHttpOptions options;
      options.endpoint = {.host = "127.0.0.1",
                          .port = static_cast<std::uint16_t>(port),
                          .tls = tls};
      options.mapping.resource_instance_id = "http.cancellation.instance";
      options.request_timeout = Ms(30000);
      ex::OtlpHttpTelemetrySink sink(std::move(options));
      const std::array batch{forge::TelemetryEnvelope{
          .identity = {.stream_id = "http.cancellation",
                       .event_id = std::to_string(round),
                       .idempotency_key = std::to_string(round),
                       .sequence = round + 1U},
          .time = {.wall_time_unix_ns = 1700000000000000000LL},
          .payload = forge::TelemetryLog{forge::TelemetryLogSeverity::info,
                                         "stalled native request"}}};
      std::stop_source stop;
      std::optional<Clock::time_point> stopped_at;
      std::jthread canceller;
      if (cancelled) {
        canceller = std::jthread([&] {
          // The independent checker writes only after its server has reached
          // the declared stalled stage, never on an arbitrary startup timer.
          if (std::cin.get() == 'x') {
            stopped_at = Clock::now();
            stop.request_stop();
          }
        });
      }
      const auto started = Clock::now();
      const auto deadline = started + (cancelled ? Ms(30000) : Ms(1000));
      bool expected_error = false;
      try {
        sink.emit(batch, stop.get_token(), deadline);
      } catch (const forge::Cancelled&) {
        expected_error = cancelled;
      } catch (const forge::DeadlineExceeded&) {
        expected_error = !cancelled;
      }
      const auto ended = Clock::now();
      if (canceller.joinable()) canceller.join();
      require(expected_error,
              "Stalled request did not throw the expected error");
      const auto elapsed = std::chrono::duration_cast<Ms>(
          ended - (cancelled ? stopped_at.value() : started));
      require(elapsed < (cancelled ? Ms(1000) : Ms(2500)),
              "Stalled native request outlived its wait budget");
      require(sink.flush().exported_events == 0U,
              "Cancelled/uncompleted request fabricated an acknowledgement");
      if (!tls) {
        const auto retried = sink.emit(batch);
        require(retried.dispositions.size() == 1U &&
                    retried.dispositions.front() ==
                        forge::TelemetryDisposition::accepted,
                "An unacknowledged request could not be retried");
        require(sink.emit(batch).dispositions.front() ==
                    forge::TelemetryDisposition::duplicate,
                "Successful retry did not retain replay identity");
      }
      require(sink.close().exported_events == (tls ? 0U : 1U),
              "Cancelled/retried observer counters disagree");
      const bool warmup = !baseline;
      const auto handles = settled_handles(baseline);
      if (warmup) {
        ++warmup_rounds;
        stable_warmups = previous_warmup == handles ? stable_warmups + 1U : 1U;
        previous_warmup = handles;
#ifdef _WIN32
        // WinHTTP lazily creates process-wide workers after its first two
        // requests. Require three identical observations before freezing the
        // baseline; never move it during the strict measured rounds.
        const bool initialized = warmup_rounds >= 2U && stable_warmups >= 3U;
#else
        const bool initialized = warmup_rounds >= 2U;
#endif
        if (initialized) baseline = handles;
        require(baseline || warmup_rounds < max_warmup_rounds,
                "Native HTTP runtime did not stabilize within warmup bound");
      } else {
        ++measured;
      }
      std::cout << "{\"event\":\"round\",\"index\":" << round
                << ",\"warmup\":" << (warmup ? "true" : "false")
                << ",\"elapsed_ms\":" << elapsed.count()
                << ",\"handles\":" << handles
                << ",\"exported_events\":" << (tls ? 0U : 1U) << "}\n"
                << std::flush;
    }
    std::cout << "{\"passed\":true,\"rounds\":"
              << warmup_rounds + measured_rounds
              << ",\"warmup_rounds\":" << warmup_rounds
              << ",\"measured_rounds\":" << measured_rounds
              << ",\"baseline_handles\":" << baseline.value() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
