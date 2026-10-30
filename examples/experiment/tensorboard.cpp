// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace experiment = forge::experiment;

forge::TelemetryEnvelope Envelope(std::uint64_t sequence) {
  forge::TelemetryEnvelope event;
  event.identity = {.stream_id = "example.tensorboard/attempt.1",
                    .event_id = "event." + std::to_string(sequence),
                    .idempotency_key = "operation." + std::to_string(sequence),
                    .sequence = sequence};
  // Fixed captured clocks make this tiny demonstration independently
  // replayable.
  event.time = {.wall_time_unix_ns = 1700000000123456789LL,
                .monotonic_time_ns = sequence * 1000U,
                .clock_id = "example.clock"};
  event.correlation = {.project_id = "example.tensorboard",
                       .run_id = "run.1",
                       .attempt_id = "attempt.1",
                       .trace_id = "example.trace"};
  event.attributes = {{"example", "六丁炼算：原生 TensorBoard"}};
  return event;
}

forge::TelemetryMetricPoint Point(std::string name, double number,
                                  std::uint64_t step = 1U) {
  forge::TelemetryMetricPoint point;
  point.definition.metric_id = std::move(name);
  point.definition.aggregation = forge::TelemetryAggregation::mean;
  point.definition.direction = forge::TelemetryMetricDirection::minimize;
  point.definition.context_keys = {"dataset", "split"};
  point.context = {{"dataset", "array-regression-v1"}, {"split", "validation"}};
  point.step = {.kind = forge::TelemetryStepKind::global_step, .value = step};
  point.observation = forge::make_telemetry_metric_value(number);
  point.sample_count = 4U;
  return point;
}

std::vector<forge::TelemetryEnvelope> Demo() {
  const std::array target{1.0, 2.0, 3.0, 4.0};
  const std::array prediction{1.25, 1.75, 3.5, 3.5};
  double absolute_error = 0.0;
  for (std::size_t index = 0U; index < target.size(); ++index)
    absolute_error += std::abs(prediction[index] - target[index]);
  absolute_error /= static_cast<double>(target.size());

  std::vector<forge::TelemetryEnvelope> events;
  auto started = Envelope(1U);
  started.payload = forge::TelemetryLifecycle{.name = "example.started"};
  events.push_back(std::move(started));
  auto measured = Envelope(2U);
  measured.payload = forge::TelemetryMetricBatch{
      {Point("validation.mean_absolute_error", absolute_error)}};
  events.push_back(std::move(measured));
  auto resource = Envelope(3U);
  auto bytes =
      Point("example.buffer_bytes", static_cast<double>(sizeof(prediction)));
  bytes.definition.unit = forge::TelemetryMetricUnit::bytes;
  bytes.definition.aggregation = forge::TelemetryAggregation::last;
  bytes.definition.direction = forge::TelemetryMetricDirection::none;
  bytes.step.kind = forge::TelemetryStepKind::sample;
  bytes.step.value = prediction.size();
  resource.payload =
      forge::TelemetryResourceSample{"example.cpu-buffer", {bytes}};
  events.push_back(std::move(resource));
  auto completed = Envelope(4U);
  auto optional = Point("validation.optional_measurement", 0.0);
  optional.observation =
      forge::make_telemetry_metric_value(std::nullopt, "not measured");
  completed.payload = forge::TelemetryLifecycle{
      .name = "example.completed",
      .terminal = true,
      .metrics = {Point("validation.mean_absolute_error", absolute_error),
                  optional},
      .decisions = {
          {"example.policy", "finish", "all four samples evaluated", true}}};
  events.push_back(std::move(completed));
  return events;
}

void AddConformanceCases(std::vector<forge::TelemetryEnvelope>& events) {
  std::uint64_t sequence = 1U;
  const auto add = [&](forge::TelemetryPayload payload, std::int64_t wall) {
    auto event = Envelope(sequence++);
    event.identity.stream_id = "example.conformance";
    event.time.wall_time_unix_ns = wall;
    event.payload = std::move(payload);
    events.push_back(std::move(event));
  };
  const std::array numbers{0.0,
                           -0.0,
                           0.12345678901234567,
                           1.0e-7,
                           1.0e20,
                           (std::numeric_limits<double>::denorm_min)(),
                           (std::numeric_limits<double>::max)(),
                           std::numeric_limits<double>::quiet_NaN(),
                           std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()};
  for (std::size_t index = 0U; index < numbers.size(); ++index) {
    auto point = Point("精度.二进制64", numbers[index], index);
    // Typed live observations permit finite -0.0. The convenience factory and
    // canonical recovery intentionally normalize it to +0.0 instead.
    if (numbers[index] == 0.0 && std::signbit(numbers[index]))
      point.observation.value = numbers[index];
    point.definition.parameters = {{"encoding", "\"quoted\"\nline\ttab"},
                                   {"unicode", "中文/😀"}};
    point.context["dataset"] = "数据集/😀";
    add(forge::TelemetryMetricBatch{{point}}, 1700000000123456789LL);
  }
  add(forge::TelemetryMetricBatch{{Point("zero.factory", -0.0)}},
      1700000000123456789LL);
  auto canonical_zero = Envelope(sequence++);
  canonical_zero.identity.stream_id = "example.conformance";
  auto zero = Point("zero.canonical_round_trip", 0.0);
  zero.observation.value = -0.0;
  canonical_zero.payload = forge::TelemetryMetricBatch{{zero}};
  const auto encoded_zero = forge::encode_telemetry_envelope(canonical_zero);
  events.push_back(forge::decode_telemetry_envelope(encoded_zero));
  for (const auto wall : std::array<std::int64_t, 7>{
           1, -1, 999999999, 1000000001, -1700000000123456789LL,
           (std::numeric_limits<std::int64_t>::min)(),
           (std::numeric_limits<std::int64_t>::max)()})
    add(forge::TelemetryMetricBatch{{Point("clocks.exact", 1.0)}}, wall);
  auto epoch = Point("validation.mean_absolute_error", 0.5);
  epoch.step.kind = forge::TelemetryStepKind::epoch;
  add(forge::TelemetryMetricBatch{{epoch}}, 1700000000123456789LL);
  auto missing = Point("missing", 0.0);
  missing.observation =
      forge::make_telemetry_metric_value(std::nullopt, "no sample");
  add(forge::TelemetryMetricBatch{{missing}}, 1700000000123456789LL);
  auto maximum = Point(
      "maximum.step", 1.0,
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()));
  add(forge::TelemetryMetricBatch{{maximum}}, 1700000000123456789LL);
  ++maximum.step.value;
  add(forge::TelemetryMetricBatch{{maximum}}, 1700000000123456789LL);
  add(forge::TelemetryLog{forge::TelemetryLogSeverity::info,
                          "unsupported explicitly"},
      1700000000123456789LL);
  auto no_clock = Envelope(sequence++);
  no_clock.identity.stream_id = "example.conformance";
  no_clock.time.wall_time_unix_ns.reset();
  no_clock.payload = forge::TelemetryMetricBatch{{Point("missing.wall", 1.0)}};
  events.push_back(std::move(no_clock));
  auto large_sequence = Envelope(
      static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()));
  large_sequence.identity.stream_id = "example.large-sequence";
  large_sequence.payload = forge::TelemetryLifecycle{.name = "int64.maximum"};
  events.push_back(large_sequence);
  large_sequence.identity.event_id = "beyond.int64";
  large_sequence.identity.idempotency_key = "beyond.int64";
  ++large_sequence.identity.sequence;
  events.push_back(std::move(large_sequence));
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    if (argc < 2 || argc > 3)
      throw forge::InvalidArgument(
          "Usage: liuding-forge-example-native-tensorboard "
          "NEW_DIRECTORY [--conformance]");
    bool conformance = false;
    if (argc == 3) {
      if (std::filesystem::path(argv[2]) != "--conformance")
        throw forge::InvalidArgument("Unknown example option");
      conformance = true;
    }
    experiment::TensorBoardTelemetrySink observer(
        {.directory = std::filesystem::path(argv[1])});
    // Store canonical semantics and the original live JSON projection together.
    // Recovery from canonical bytes normalizes signed zero to positive zero.
    experiment::JsonlTelemetrySink source(
        {.path = observer.event_file().parent_path() / "source-events.jsonl"});
    auto events = Demo();
    if (conformance) AddConformanceCases(events);
    source.emit(events);
    const auto first = observer.emit(events);
    const auto replay = observer.emit(events);
    for (std::size_t index = 0U; index < first.dispositions.size(); ++index) {
      const auto expected =
          first.dispositions[index] == forge::TelemetryDisposition::accepted
              ? forge::TelemetryDisposition::duplicate
              : first.dispositions[index];
      if (replay.dispositions[index] != expected)
        throw forge::InternalError("Replay changed the observation receipt");
    }
    source.close();
    const auto result = observer.close();
    if (observer.close().exported_events != result.exported_events)
      throw forge::InternalError("Repeated close changed the final counters");
    std::cout << "Native TensorBoard export: " << result.exported_events
              << " accepted, " << result.unsupported_events
              << " unsupported. Immutable replay added no events.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
