// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
  namespace forge = jyppx::liuding;
  const auto root =
      std::filesystem::temp_directory_path() /
      ("liuding-native-telemetry-consumer-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    if (!std::filesystem::create_directory(root)) return 2;
    forge::experiment::TensorBoardTelemetrySink observer(
        {.directory = root / "events"});
    forge::TelemetryEnvelope event{
        .identity = {.stream_id = "installed",
                     .event_id = "first",
                     .idempotency_key = "first",
                     .sequence = 1U},
        .time = {.wall_time_unix_ns = 1700000000000000000LL,
                 .monotonic_time_ns = 10U,
                 .clock_id = "installed.clock"},
        .correlation = {.project_id = "example",
                        .run_id = "installed",
                        .attempt_id = "one"},
        .payload = forge::TelemetryLifecycle{.name = "installed.start"}};
    const std::array batch{event};
    const auto first = observer.emit(batch);
    const auto duplicate = observer.emit(batch);
    const auto result = observer.close();
    if (first.dispositions !=
            std::vector{forge::TelemetryDisposition::accepted} ||
        duplicate.dispositions !=
            std::vector{forge::TelemetryDisposition::duplicate} ||
        result.exported_events != 1U ||
        !std::filesystem::is_regular_file(observer.event_file()))
      return 3;
    forge::experiment::OtlpProjectionEncoder encoder(
        {.resource_instance_id = "installed.example"});
    const auto requests = encoder.encode(event);
    if (requests.size() != 1U || requests.front().body.empty() ||
        requests.front().signal != forge::experiment::OtlpSignal::logs ||
        encoder.signal_path(requests.front().signal) != "/v1/logs")
      return 5;
    encoder.validate_response(requests.front().signal, {});
    std::filesystem::remove_all(root);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    return 4;
  }
}
