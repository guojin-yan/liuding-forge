// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"
#include "sidecar_process.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#ifdef JYPPX_LIUDING_CLI_NATIVE_TELEMETRY
#include <jyppx/liuding/telemetry_native/tensorboard.hpp>
#endif
#include <jyppx/liuding/foundation/sha256.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::cli {
namespace {
using namespace experiment;

constexpr std::uint64_t kMaximumCoordinate =
    static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)());
constexpr std::uint64_t kMaximumPageBytes = 16U * 1024U * 1024U;
constexpr std::uint64_t kMaximumResponseBytes = 64U * 1024U * 1024U;

std::uint64_t positive(const Arguments& arguments, std::string_view name,
                       std::uint64_t fallback, std::uint64_t maximum) {
  const auto value = arguments.number(name, fallback, maximum);
  if (value == 0U)
    usage("run.telemetry",
          "Page sizes, capacities, and timeouts must be positive.");
  return value;
}

TelemetryDeadline operation_deadline(const Arguments& arguments) {
  return std::chrono::steady_clock::now() +
         std::chrono::milliseconds(
             positive(arguments, "timeout-ms", 60000U, 600000U));
}

std::optional<std::uint64_t> snapshot(const Arguments& arguments,
                                      std::uint64_t after) {
  if (arguments.optional("snapshot")) {
    return arguments.number("snapshot", 0U, kMaximumCoordinate);
  }
  if (after != 0U)
    usage("run.snapshot",
          "Resume with both the returned --snapshot and exclusive cursor.");
  return std::nullopt;
}

SQLiteRunStoreOptions store_options(const Arguments& arguments) {
  SQLiteRunStoreOptions options;
  options.event_limits.max_page_bytes =
      positive(arguments, "page-bytes", kMaximumPageBytes, kMaximumPageBytes);
  if (options.event_limits.max_page_bytes <
      options.event_limits.telemetry.max_envelope_bytes) {
    usage("run.page-bytes",
          "Use a page byte budget between 1048576 and 16777216.");
  }
  return options;
}

std::filesystem::path existing_root(const Arguments& arguments) {
  const auto root = std::filesystem::u8path(arguments.require("root"));
  std::error_code error;
  if (!std::filesystem::is_regular_file(root / "runs.sqlite3", error) ||
      error) {
    throw IoError("Run storage database does not exist",
                  "Use an existing local storage root; telemetry reads do not "
                  "create one.");
  }
  return root;
}

TelemetryPayloadKind payload_kind(std::string_view value) {
  if (value == "lifecycle") return TelemetryPayloadKind::lifecycle;
  if (value == "metric_batch") return TelemetryPayloadKind::metric_batch;
  if (value == "log") return TelemetryPayloadKind::log;
  if (value == "span") return TelemetryPayloadKind::span;
  if (value == "resource_sample") return TelemetryPayloadKind::resource_sample;
  if (value == "artifact_reference")
    return TelemetryPayloadKind::artifact_reference;
  usage("run.kind",
        "Choose "
        "lifecycle/metric_batch/log/span/resource_sample/artifact_reference.");
}

TelemetryStepKind step_kind(std::string_view value) {
  if (value == "micro_step") return TelemetryStepKind::micro_step;
  if (value == "global_step") return TelemetryStepKind::global_step;
  if (value == "epoch") return TelemetryStepKind::epoch;
  if (value == "sample") return TelemetryStepKind::sample;
  if (value == "elapsed_nanoseconds")
    return TelemetryStepKind::elapsed_nanoseconds;
  usage("run.step-kind",
        "Choose micro_step/global_step/epoch/sample/elapsed_nanoseconds.");
}

std::string_view status_name(TelemetryValueStatus status) {
  switch (status) {
    case TelemetryValueStatus::finite:
      return "finite";
    case TelemetryValueStatus::not_a_number:
      return "not_a_number";
    case TelemetryValueStatus::positive_infinity:
      return "positive_infinity";
    case TelemetryValueStatus::negative_infinity:
      return "negative_infinity";
    case TelemetryValueStatus::missing:
      return "missing";
  }
  throw SerializationError("Unknown persisted metric status");
}

std::string canonical_hex(std::span<const std::byte> bytes) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2U);
  for (const auto byte : bytes) {
    const auto value = std::to_integer<unsigned>(byte);
    result.push_back(digits[value >> 4U]);
    result.push_back(digits[value & 15U]);
  }
  return result;
}

Json event_record(const StoredRunEvent& stored, const TelemetryLimits& limits) {
  const auto bytes = encode_telemetry_envelope(stored.envelope, limits);
  return {{"schema_version", 1U},
          {"run_sequence", stored.run_sequence},
          {"content_digest", sha256_hex(bytes)},
          {"canonical_hex", canonical_hex(bytes)},
          {"event",
           Json::parse(telemetry_projection_json(stored.envelope, limits))}};
}

void response_budget(const Json& value, std::uint64_t& used) {
  const auto bytes = value.dump().size();
  if (bytes > kMaximumResponseBytes - used) {
    throw ResourceExhausted(
        "Telemetry page JSON exceeds 64 MiB",
        "Resume the same snapshot with a smaller --limit or --page-bytes.");
  }
  used += bytes;
}

Json cursor_json(const RunMetricCursor& cursor) {
  return {{"run_sequence", cursor.run_sequence},
          {"point_index", cursor.point_index}};
}

Result events_command(const Arguments& arguments, SQLiteRunStore& store,
                      std::string_view run_id, TelemetryDeadline deadline) {
  RunEventPageRequest request;
  request.after_sequence = arguments.number("after", 0U, kMaximumCoordinate);
  request.snapshot_through = snapshot(arguments, request.after_sequence);
  request.limit =
      static_cast<std::size_t>(positive(arguments, "limit", 256U, 1024U));
  if (const auto kind = arguments.optional("kind"))
    request.kind = payload_kind(*kind);
  const auto page = store.events(run_id, request, {}, deadline);
  Json records = Json::array();
  std::uint64_t bytes = 0U;
  for (const auto& stored : page.events) {
    check_telemetry_operation({}, deadline);
    auto record = event_record(stored, store.event_limits().telemetry);
    response_budget(record, bytes);
    records.push_back(std::move(record));
  }
  check_telemetry_operation({}, deadline);
  return {Json{{"run_id", run_id},
               {"source", "sqlite-canonical"},
               {"snapshot_through", page.snapshot_through},
               {"after_sequence", request.after_sequence},
               {"kind", arguments.optional("kind")
                            ? Json(*arguments.optional("kind"))
                            : Json(nullptr)},
               {"events", std::move(records)},
               {"next_after",
                page.next_after ? Json(*page.next_after) : Json(nullptr)}},
          0};
}

const Json& projected_points(const Json& event) {
  const auto& payload = event.at("payload");
  if (payload.contains("points")) return payload.at("points");
  if (payload.contains("metrics")) return payload.at("metrics");
  if (payload.contains("measurements")) return payload.at("measurements");
  throw SerializationError("Metric cursor refers to an event without metrics");
}

Result metrics_command(const Arguments& arguments, SQLiteRunStore& store,
                       std::string_view run_id, TelemetryDeadline deadline) {
  RunMetricPageRequest request;
  request.metric_id = arguments.require("metric");
  request.after.run_sequence =
      arguments.number("after-sequence", 0U, kMaximumCoordinate);
  request.after.point_index =
      arguments.number("after-index", 0U, kMaximumCoordinate);
  request.snapshot_through = snapshot(arguments, request.after.run_sequence);
  request.limit =
      static_cast<std::size_t>(positive(arguments, "limit", 256U, 4096U));
  if (arguments.optional("definition-version")) {
    request.definition_version = static_cast<std::uint32_t>(
        positive(arguments, "definition-version", 1U,
                 (std::numeric_limits<std::uint32_t>::max)()));
  }
  if (const auto kind = arguments.optional("step-kind"))
    request.step_kind = step_kind(*kind);
  const auto page = store.metrics(run_id, request, {}, deadline);
  Json records = Json::array();
  Json sources = Json::array();
  Json source;
  std::uint64_t source_sequence = 0U;
  std::uint64_t bytes = 0U;
  for (const auto& stored : page.points) {
    check_telemetry_operation({}, deadline);
    if (source_sequence != stored.cursor.run_sequence) {
      const auto source_page =
          store.events(run_id,
                       {.after_sequence = stored.cursor.run_sequence - 1U,
                        .snapshot_through = page.snapshot_through,
                        .kind = {},
                        .limit = 1U},
                       {}, deadline);
      if (source_page.events.size() != 1U ||
          source_page.events.front().run_sequence !=
              stored.cursor.run_sequence) {
        throw SerializationError(
            "Metric source event is missing from its fixed snapshot");
      }
      source = event_record(source_page.events.front(),
                            store.event_limits().telemetry);
      response_budget(source, bytes);
      sources.push_back(source);
      source_sequence = stored.cursor.run_sequence;
    }
    const auto& event = source.at("event");
    const auto& values = projected_points(event);
    if (stored.cursor.point_index >= values.size())
      throw SerializationError("Metric point is outside its source envelope");
    Json record{
        {"cursor", cursor_json(stored.cursor)},
        {"source_content_digest", source.at("content_digest")},
        {"event_schema_version", event.at("schema_version")},
        {"event_identity", event.at("identity")},
        {"time", event.at("time")},
        {"correlation", event.at("correlation")},
        {"reliability", event.at("reliability")},
        {"attributes", event.at("attributes")},
        {"resource_id",
         stored.resource_id ? Json(*stored.resource_id) : Json(nullptr)},
        {"status_name", status_name(stored.point.observation.status)},
        {"point",
         values.at(static_cast<std::size_t>(stored.cursor.point_index))}};
    response_budget(record, bytes);
    records.push_back(std::move(record));
  }
  check_telemetry_operation({}, deadline);
  return {Json{{"run_id", run_id},
               {"source", "sqlite-canonical"},
               {"metric_id", request.metric_id},
               {"snapshot_through", page.snapshot_through},
               {"after", cursor_json(request.after)},
               {"points", std::move(records)},
               {"source_events", std::move(sources)},
               {"next_after", page.next_after ? cursor_json(*page.next_after)
                                              : Json(nullptr)}},
          0};
}

std::uint64_t config_number(const Json& config, const char* name,
                            std::uint64_t fallback, std::uint64_t maximum) {
  const auto value = config.contains(name)
                         ? unsigned_number(config.at(name), maximum)
                         : fallback;
  if (value == 0U)
    usage("run.sidecar-config",
          "Sidecar limits and timeouts must be positive.");
  return value;
}

SidecarProcessOptions sidecar_options(const Arguments& arguments,
                                      std::string_view backend) {
  const auto config = parse_json(
      read_text(std::filesystem::u8path(arguments.require("sidecar-config")),
                256U * 1024U));
  fields(config,
         {"schema_version", "backend", "executable", "arguments",
          "working_directory"},
         {"environment", "inherit_environment", "operation_timeout_ms",
          "startup_timeout_ms", "cleanup_timeout_ms", "max_request_bytes",
          "max_reply_bytes", "max_stderr_bytes"});
  if (unsigned_number(config.at("schema_version"), 1U) != 1U ||
      config.at("backend").get<std::string>() != backend) {
    throw CompatibilityError(
        "Sidecar configuration schema or backend does not match the export");
  }
  if (!config.at("arguments").is_array() ||
      config.at("arguments").size() > 256U ||
      (config.contains("environment") &&
       (!config.at("environment").is_object() ||
        config.at("environment").size() > 128U))) {
    throw ResourceExhausted(
        "Sidecar configuration exceeds argv/environment bounds");
  }
  SidecarProcessOptions options;
  options.executable =
      std::filesystem::u8path(config.at("executable").get<std::string>());
  options.arguments = config.at("arguments").get<std::vector<std::string>>();
  options.working_directory = std::filesystem::u8path(
      config.at("working_directory").get<std::string>());
  if (config.contains("environment"))
    options.environment =
        config.at("environment").get<std::map<std::string, std::string>>();
  options.inherit_environment = config.value("inherit_environment", false);
  options.operation_timeout = std::chrono::milliseconds(
      config_number(config, "operation_timeout_ms", 10000U, 600000U));
  options.startup_timeout = std::chrono::milliseconds(
      config_number(config, "startup_timeout_ms", 10000U, 600000U));
  options.cleanup_timeout = std::chrono::milliseconds(
      config_number(config, "cleanup_timeout_ms", 2000U, 10000U));
  options.max_request_bytes = static_cast<std::size_t>(config_number(
      config, "max_request_bytes", 16U * 1024U * 1024U, 16U * 1024U * 1024U));
  options.max_reply_bytes = static_cast<std::size_t>(
      config_number(config, "max_reply_bytes", 128U * 1024U, 128U * 1024U));
  options.max_stderr_bytes = static_cast<std::size_t>(
      config_number(config, "max_stderr_bytes", 64U * 1024U, 64U * 1024U));
  if (options.max_request_bytes < 1024U)
    usage("run.sidecar-config", "Allow at least 1024 request bytes.");
  return options;
}

bool boolean_option(const Arguments& arguments, std::string_view name) {
  const auto value = arguments.optional(name).value_or("false");
  if (value == "true") return true;
  if (value == "false") return false;
  usage("run.export", "Boolean options require true or false.");
}

std::filesystem::path telemetry_output_path(const Arguments& arguments,
                                            const SQLiteRunStore& store) {
  const auto path = std::filesystem::u8path(arguments.require("output"));
  std::error_code error;
  const auto absolute = std::filesystem::absolute(path, error);
  if (error) throw IoError("Could not resolve the telemetry output path");
  const auto resolved = std::filesystem::weakly_canonical(absolute, error);
  if (error) throw IoError("Could not resolve the telemetry output path");
  const auto root = store.database_path().parent_path();
  for (auto ancestor = resolved; !ancestor.empty();) {
    error.clear();
    if (std::filesystem::equivalent(ancestor, root, error) && !error) {
      usage("run.output",
            "Choose a dedicated output outside the managed SQLite/CAS root.");
    }
    if (error && error != std::errc::no_such_file_or_directory) {
      throw IoError("Could not inspect telemetry output ancestry");
    }
    const auto parent = ancestor.parent_path();
    if (parent == ancestor) break;
    ancestor = parent;
  }
  return path;
}

std::filesystem::path jsonl_output(const Arguments& arguments,
                                   const SQLiteRunStore& store) {
  const auto path = telemetry_output_path(arguments, store);
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error && error != std::errc::no_such_file_or_directory)
    throw IoError("Could not inspect the telemetry output path");
  if (!error && std::filesystem::exists(status)) {
    if (!boolean_option(arguments, "resume"))
      usage("run.output",
            "Use a new file, or --resume true for an exclusively owned JSONL "
            "export.");
    if (!std::filesystem::is_regular_file(status))
      throw IoError("Telemetry output must be a regular file");
    if (std::filesystem::hard_link_count(path, error) != 1U || error) {
      usage("run.output",
            "A resumed JSONL export must have exactly one filesystem link.");
    }
    // Do not let recovery truncate an arbitrary nonempty, unterminated file.
    // Full record/digest/projection verification belongs to JsonlTelemetrySink.
    std::ifstream input(path, std::ios::binary);
    if (!input) throw IoError("Could not inspect the existing JSONL export");
    char character = 0;
    std::size_t inspected = 0U;
    bool complete_record = false;
    while (input.get(character)) {
      ++inspected;
      if (character == '\n') {
        complete_record = true;
        break;
      }
      if (inspected > 4U * 1024U * 1024U)
        throw ResourceExhausted("Existing JSONL first record exceeds 4 MiB");
    }
    if (input.bad()) throw IoError("Could not read the existing JSONL export");
    if (inspected != 0U && !complete_record)
      throw SerializationError(
          "Existing output has no complete JSONL record; preserve it for "
          "diagnosis");
  } else {
    static_cast<void>(boolean_option(arguments, "resume"));
  }
  return path;
}

struct ExportReceipts {
  std::uint64_t accepted = 0U;
  std::uint64_t duplicate = 0U;
  std::uint64_t sampled = 0U;
  std::uint64_t unsupported = 0U;
};

void export_batch(TelemetrySink& sink, std::vector<TelemetryEnvelope>& batch,
                  ExportReceipts& counts, TelemetryDeadline deadline,
                  const TelemetryLimits& limits) {
  if (batch.empty()) return;
  const auto result = sink.emit(batch, {}, deadline);
  validate_telemetry_emit_result(sink.descriptor(), batch, result, limits);
  for (const auto disposition : result.dispositions) {
    switch (disposition) {
      case TelemetryDisposition::accepted:
        ++counts.accepted;
        break;
      case TelemetryDisposition::duplicate:
        ++counts.duplicate;
        break;
      case TelemetryDisposition::sampled_out:
        ++counts.sampled;
        break;
      case TelemetryDisposition::unsupported:
        ++counts.unsupported;
        break;
    }
  }
  batch.clear();
}

Json counters_json(const TelemetryFlushResult& counters) {
  return {{"exported_events", counters.exported_events},
          {"sampled_out_events", counters.sampled_out_events},
          {"unsupported_events", counters.unsupported_events}};
}

Result export_command(const Arguments& arguments, SQLiteRunStore& store,
                      std::string_view run_id, TelemetryDeadline deadline) {
  const auto backend = arguments.require("backend");
  const bool jsonl = backend == "jsonl";
  const bool native_tensorboard = backend == "tensorboard-native";
  const bool local = jsonl || native_tensorboard;
  if (!local && backend != "tensorboard" && backend != "opentelemetry")
    usage("run.backend",
          "Choose jsonl/tensorboard-native/tensorboard/opentelemetry.");
  if ((local && arguments.optional("sidecar-config")) ||
      (!local &&
       (arguments.optional("output") || arguments.optional("resume"))) ||
      (native_tensorboard && arguments.optional("resume")))
    usage(
        "run.export",
        "JSONL uses --output/--resume; tensorboard-native uses a new --output "
        "directory; sidecar backends use --sidecar-config.");
  RunEventPageRequest request;
  request.after_sequence = arguments.number("after", 0U, kMaximumCoordinate);
  const auto initial_after = request.after_sequence;
  request.snapshot_through = snapshot(arguments, request.after_sequence);
  request.limit =
      static_cast<std::size_t>(positive(arguments, "page-size", 256U, 1024U));
  const auto max_events = positive(arguments, "max-events", 1000000U, 1000000U);
  auto page = store.events(run_id, request, {}, deadline);
  request.snapshot_through = page.snapshot_through;
  if (page.snapshot_through - request.after_sequence > max_events) {
    throw ResourceExhausted("Export snapshot exceeds --max-events",
                            "Choose an explicit smaller snapshot or increase "
                            "the bounded capacity.");
  }
  std::shared_ptr<SidecarProcessTransport> process;
  std::unique_ptr<TelemetrySink> sink;
  std::size_t request_bytes = 16U * 1024U * 1024U;
  Json recovery = nullptr;
  if (jsonl) {
    JsonlTelemetryOptions options;
    options.path = jsonl_output(arguments, store);
    options.limits = store.event_limits().telemetry;
    auto observer = std::make_unique<JsonlTelemetrySink>(std::move(options));
    const auto recovered = observer->recovery();
    recovery = {{"complete_records", recovered.complete_records},
                {"truncated_tail_bytes", recovered.truncated_tail_bytes}};
    sink = std::move(observer);
  } else if (native_tensorboard) {
#ifdef JYPPX_LIUDING_CLI_NATIVE_TELEMETRY
    TensorBoardTelemetryOptions options;
    options.directory = telemetry_output_path(arguments, store);
    options.sink_id = "cli.export.tensorboard-native";
    options.destination_id = "local.tensorboard-native";
    options.limits = store.event_limits().telemetry;
    options.max_seen = static_cast<std::size_t>(max_events);
    sink = std::make_unique<TensorBoardTelemetrySink>(std::move(options));
#else
    throw NotImplemented("Native TensorBoard is disabled in this build");
#endif
  } else {
    auto options = sidecar_options(arguments, backend);
    request_bytes = options.max_request_bytes;
    const auto reply_bytes = options.max_reply_bytes;
    check_telemetry_operation({}, deadline);
    process = std::make_shared<SidecarProcessTransport>(
        std::move(options), std::stop_token{}, deadline);
    OfficialTelemetryOptions official;
    official.backend = backend == "tensorboard"
                           ? OfficialTelemetryBackend::tensorboard
                           : OfficialTelemetryBackend::opentelemetry;
    official.sink_id = "cli.export." + backend;
    official.destination_id = "configured." + backend;
    official.limits = store.event_limits().telemetry;
    official.max_request_bytes = request_bytes;
    official.max_reply_bytes = reply_bytes;
    sink =
        std::make_unique<OfficialTelemetrySink>(process, std::move(official));
  }
  const auto initial = sink->flush({}, deadline);
  ExportReceipts counts;
  std::uint64_t read_events = 0U;
  std::uint64_t last_after = request.after_sequence;
  const auto& limits = store.event_limits().telemetry;
  while (true) {
    std::vector<TelemetryEnvelope> batch;
    std::uint64_t canonical_bytes = 0U;
    std::size_t projected_bytes = 256U;
    for (auto& stored : page.events) {
      check_telemetry_operation({}, deadline);
      if (++read_events > max_events)
        throw ResourceExhausted("Export event capacity exceeded");
      const auto canonical_size =
          encode_telemetry_envelope(stored.envelope, limits).size();
      const auto projected_size =
          telemetry_projection_json(stored.envelope, limits).size() + 128U;
      if (projected_size > request_bytes - 256U)
        throw ResourceExhausted(
            "One event exceeds the configured export request capacity");
      if (!batch.empty() &&
          (batch.size() == limits.max_batch_events ||
           canonical_size > limits.max_batch_bytes - canonical_bytes ||
           projected_size > request_bytes - projected_bytes)) {
        export_batch(*sink, batch, counts, deadline, limits);
        canonical_bytes = 0U;
        projected_bytes = 256U;
      }
      canonical_bytes += canonical_size;
      projected_bytes += projected_size;
      last_after = stored.run_sequence;
      batch.push_back(std::move(stored.envelope));
    }
    export_batch(*sink, batch, counts, deadline, limits);
    if (!page.next_after) break;
    if (*page.next_after <= request.after_sequence || page.events.empty()) {
      throw SerializationError("Export snapshot cursor did not advance");
    }
    request.after_sequence = *page.next_after;
    page = store.events(run_id, request, {}, deadline);
    if (page.snapshot_through != *request.snapshot_through)
      throw SerializationError("Export snapshot boundary changed");
  }
  if (read_events != *request.snapshot_through - initial_after) {
    throw SerializationError(
        "Export snapshot contains missing commit coordinates");
  }
  const auto final = sink->close({}, deadline);
  const auto delta_matches = [](std::uint64_t before, std::uint64_t after,
                                std::uint64_t count) {
    return after >= before && after - before == count;
  };
  if (!delta_matches(initial.exported_events, final.exported_events,
                     counts.accepted) ||
      !delta_matches(initial.sampled_out_events, final.sampled_out_events,
                     counts.sampled) ||
      !delta_matches(initial.unsupported_events, final.unsupported_events,
                     counts.unsupported)) {
    throw SerializationError(
        "Export receipt counts disagree with final observer counters");
  }
  check_telemetry_operation({}, deadline);
  Json transport_stats = nullptr;
  if (process) {
    const auto stats = process->stats();
    transport_stats = {{"completed_requests", stats.completed_requests},
                       {"stdout_bytes", stats.stdout_bytes},
                       {"discarded_stderr_bytes", stats.stderr_bytes}};
    process->terminate();
  }
  const bool represented = counts.unsupported == 0U && counts.sampled == 0U;
  return {Json{{"run_id", run_id},
               {"source", "sqlite-canonical"},
               {"backend", backend},
               {"snapshot_through", *request.snapshot_through},
               {"last_after", last_after},
               {"scan_completed", true},
               {"fully_represented", represented},
               {"read_events", read_events},
               {"accepted_events", counts.accepted},
               {"duplicate_events", counts.duplicate},
               {"sampled_out_events", counts.sampled},
               {"unsupported_events", counts.unsupported},
               {"delivery_guarantee", "at_least_once"},
               {"destination_counters", counters_json(final)},
               {"jsonl_recovery", recovery},
               {"transport", transport_stats}},
          represented ? 0 : 1};
}
}  // namespace

Result run_telemetry_command(const Arguments& arguments) {
  if (arguments.words().size() != 2U || arguments.words().front() != "run") {
    usage("run.telemetry",
          "Use run events/metrics/export with a storage root and run ID.");
  }
  const auto& action = arguments.words()[1];
  if (action == "events") {
    arguments.allow({"root", "id", "after", "snapshot", "kind", "limit",
                     "page-bytes", "timeout-ms"});
  } else if (action == "metrics") {
    arguments.allow({"root", "id", "metric", "after-sequence", "after-index",
                     "snapshot", "definition-version", "step-kind", "limit",
                     "page-bytes", "timeout-ms"});
  } else if (action == "export") {
    arguments.allow({"root", "id", "backend", "output", "resume",
                     "sidecar-config", "after", "snapshot", "page-size",
                     "page-bytes", "max-events", "timeout-ms"});
  } else {
    usage("run.telemetry", "Choose events/metrics/export.");
  }
  const auto root = existing_root(arguments);
  const auto run_id = arguments.require("id");
  const auto deadline = operation_deadline(arguments);
  SQLiteRunStore store(root, store_options(arguments), {}, deadline);
  if (action == "events")
    return events_command(arguments, store, run_id, deadline);
  if (action == "metrics")
    return metrics_command(arguments, store, run_id, deadline);
  return export_command(arguments, store, run_id, deadline);
}
}  // namespace jyppx::liuding::cli
