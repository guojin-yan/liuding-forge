// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/telemetry_native/otlp_projection.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace forge = jyppx::liuding;
namespace experiment = forge::experiment;
using Json = nlohmann::json;

forge::TelemetryEnvelope Event(std::uint64_t sequence) {
  forge::TelemetryEnvelope event;
  event.identity = {.stream_id = "example.otlp",
                    .event_id = std::to_string(sequence),
                    .idempotency_key = std::to_string(sequence),
                    .sequence = sequence};
  event.time = {.wall_time_unix_ns = 1700000000123456789LL,
                .monotonic_time_ns = 1000U + sequence,
                .clock_id = "captured.clock"};
  event.correlation = {.project_id = "offline-learning",
                       .run_id = "run.1",
                       .attempt_id = "attempt.1",
                       .trace_id = "训练/😀",
                       .span_id = "phase.evaluate",
                       .parent_span_id = "phase.run"};
  event.attributes = {{"example", "OTLP：保留原始定义和时钟"}};
  return event;
}

forge::TelemetryMetricPoint Point(double value, std::uint64_t step) {
  forge::TelemetryMetricPoint point;
  point.definition.metric_id = "validation.mean_absolute_error";
  point.definition.aggregation = forge::TelemetryAggregation::mean;
  point.definition.direction = forge::TelemetryMetricDirection::minimize;
  point.definition.context_keys = {"dataset", "split"};
  point.context = {{"dataset", "数组/😀"}, {"split", "validation"}};
  point.step = {.kind = forge::TelemetryStepKind::global_step, .value = step};
  point.observation = forge::make_telemetry_metric_value(value);
  point.sample_count = 4U;
  return point;
}

std::vector<forge::TelemetryEnvelope> Events() {
  std::vector<forge::TelemetryEnvelope> events;
  auto add = [&](forge::TelemetryPayload payload) -> forge::TelemetryEnvelope& {
    auto event = Event(events.size() + 1U);
    event.payload = std::move(payload);
    events.push_back(std::move(event));
    return events.back();
  };
  add(forge::TelemetryLifecycle{.name = "evaluation.started"});
  const std::array prediction{1.25, 1.75, 3.5, 3.5};
  const std::array target{1.0, 2.0, 3.0, 4.0};
  double error = 0.0;
  for (std::size_t index = 0U; index < target.size(); ++index)
    error += std::abs(prediction[index] - target[index]);
  auto measured = Point(error / static_cast<double>(target.size()), 7U);
  auto missing = Point(0.0, 7U);
  missing.definition.metric_id = "validation.optional";
  missing.observation =
      forge::make_telemetry_metric_value(std::nullopt, "not measured");
  add(forge::TelemetryMetricBatch{{measured, missing}});
  auto resource = Point(static_cast<double>(sizeof(prediction)), 4U);
  resource.definition.metric_id = "example.prediction_buffer_bytes";
  resource.definition.unit = forge::TelemetryMetricUnit::bytes;
  resource.definition.aggregation = forge::TelemetryAggregation::last;
  resource.definition.direction = forge::TelemetryMetricDirection::none;
  add(forge::TelemetryResourceSample{"example.buffer", {resource}});
  add(forge::TelemetrySpan{.name = "evaluation",
                           .start_monotonic_ns = 1000U,
                           .end_monotonic_ns = 1004U,
                           .status = forge::TelemetrySpanStatus::ok});
  auto& no_wall = add(forge::TelemetryLog{forge::TelemetryLogSeverity::info,
                                          "wall clock unavailable"});
  no_wall.time.wall_time_unix_ns.reset();
  auto& negative_wall = add(forge::TelemetryMetricBatch{{measured}});
  negative_wall.time.wall_time_unix_ns = -1;
  add(forge::TelemetryLifecycle{.name = "evaluation.completed",
                                .terminal = true,
                                .metrics = {measured, missing}});
  for (const auto number :
       std::array{0.0, -0.0, 1.0e-7, 1.0e20, 0.12345678901234567,
                  (std::numeric_limits<double>::denorm_min)(),
                  (std::numeric_limits<double>::max)(),
                  std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()})
    add(forge::TelemetryMetricBatch{
        {Point(number, (std::numeric_limits<std::uint64_t>::max)())}});
  // The factory above normalizes both zero signs. Direct typed live
  // observations may carry -0; canonical persistence restores +0.
  auto live_zero = Point(0.0, 9U);
  live_zero.definition.metric_id = "example.signed_zero";
  live_zero.observation = {.status = forge::TelemetryValueStatus::finite,
                           .value = -0.0};
  auto& live_event = add(forge::TelemetryMetricBatch{{live_zero}});
  live_event.attributes["zero_path"] = "typed-live";
  const auto canonical_bytes = forge::encode_telemetry_envelope(live_event);
  const auto canonical = forge::decode_telemetry_envelope(canonical_bytes);
  auto& recovered_event = add(canonical.payload);
  recovered_event.attributes["zero_path"] = "canonical-recovered";
  auto& no_anchor = add(forge::TelemetrySpan{
      .name = "unmapped.clock",
      .start_monotonic_ns = 1U,
      .end_monotonic_ns = 2U,
      .status = forge::TelemetrySpanStatus::cancelled,
      .reason = "explicitly unsupported without an anchor"});
  no_anchor.time.clock_id = "unmapped.clock";
  return events;
}

std::string Read(const std::filesystem::path& path, std::size_t maximum) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) > maximum)
    throw forge::ResourceExhausted(
        "Example input is not a bounded regular file");
  std::ifstream input(path, std::ios::binary);
  std::string result(static_cast<std::size_t>(std::filesystem::file_size(path)),
                     '\0');
  if (!input.read(result.data(), static_cast<std::streamsize>(result.size())))
    throw forge::IoError("Cannot read example input");
  return result;
}

void Write(const std::filesystem::path& path, std::string_view bytes) {
  std::ofstream output(path,
                       std::ios::binary | std::ios::out | std::ios::trunc);
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
}

void VerifyReplies(const std::filesystem::path& directory) {
  experiment::OtlpProjectionEncoder encoder(
      {.resource_instance_id = "offline-example"});
  const auto manifest = Json::parse(Read(directory / "replies.json", 65536U));
  if (manifest.at("schema_version") != 1 || manifest.at("cases").size() > 64U)
    throw forge::InvalidArgument("Invalid official reply fixture manifest");
  for (const auto& item : manifest.at("cases")) {
    const auto name = item.at("file").get<std::string>();
    if (std::filesystem::path(name).filename() != name)
      throw forge::InvalidArgument("Reply fixture must use a local filename");
    const auto signal_name = item.at("signal").get<std::string>();
    const auto signal =
        signal_name == "metrics" ? experiment::OtlpSignal::metrics
        : signal_name == "logs"  ? experiment::OtlpSignal::logs
        : signal_name == "traces"
            ? experiment::OtlpSignal::traces
            : throw forge::InvalidArgument("Unknown fixture signal");
    const auto body = Read(directory / name, 1024U * 1024U + 1U);
    bool accepted = false;
    try {
      encoder.validate_response(
          signal,
          {reinterpret_cast<const std::byte*>(body.data()), body.size()});
      accepted = true;
    } catch (const forge::Error&) {
    }
    if (accepted != item.at("accepted").get<bool>())
      throw forge::InternalError(
          "Native/official OTLP response decision differs");
  }
}

void Project(const std::filesystem::path& directory) {
  experiment::OtlpProjectionOptions options;
  options.resource_instance_id = "offline-example";
  options.clock_mappings = {
      {"captured.clock", {1000U, 1700000000123456789ULL}}};
  const experiment::OtlpProjectionEncoder encoder(options);
  if (!std::filesystem::create_directory(directory))
    throw forge::IoError("Example requires a new private output directory");
  experiment::JsonlTelemetrySink source(
      {.path = directory / "source-events.jsonl"});
  const auto events = Events();
  source.emit(events);
  source.close();
  Json index{
      {"schema_version", 1},
      {"resource_instance_id", options.resource_instance_id},
      {"clock_mappings",
       {{"captured.clock",
         {{"monotonic_ns", 1000U}, {"unix_ns", 1700000000123456789ULL}}}}},
      {"events", Json::array()}};
  std::size_t requests = 0U;
  std::size_t unsupported = 0U;
  for (const auto& event : events) {
    const auto encoded = encoder.encode(event);
    Json item{{"sequence", event.identity.sequence},
              {"unsupported", encoded.empty()},
              {"requests", Json::array()}};
    if (encoded.empty()) ++unsupported;
    for (std::size_t part = 0U; part < encoded.size(); ++part) {
      const auto name = "event-" + std::to_string(event.identity.sequence) +
                        "-" + std::to_string(part) + ".pb";
      const auto signal = encoder.signal_path(encoded[part].signal).substr(4U);
      Write(directory / name, encoded[part].body);
      item["requests"].push_back({{"signal", signal},
                                  {"file", name},
                                  {"bytes", encoded[part].body.size()}});
      ++requests;
    }
    index["events"].push_back(std::move(item));
  }
  Write(directory / "projection-index.json", index.dump(2) + "\n");
  std::cout << events.size() << " events, " << requests << " requests, "
            << unsupported
            << " unsupported. Prepared for offline inspection.\n";
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    if (argc == 3 && std::filesystem::path(argv[1]) == "--verify-replies") {
      VerifyReplies(std::filesystem::path(argv[2]));
      return 0;
    }
    if (argc != 2)
      throw forge::InvalidArgument(
          "Usage: liuding-forge-example-otlp-projection NEW_DIRECTORY");
    Project(std::filesystem::path(argv[1]));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
