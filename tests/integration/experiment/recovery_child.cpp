// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// This test-only executable exposes a supervisor fence around the real SQLite
// sink. The Python driver kills it while emit() is still on the stack.

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/experiment/run_telemetry_sink.hpp>
#include <jyppx/liuding/experiment/telemetry_exporters.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>
#include <jyppx/liuding/foundation/telemetry.hpp>
#include <jyppx/liuding/training/checkpoint_coordinator.hpp>
#include <jyppx/liuding/training/event_sink.hpp>
#include <jyppx/liuding/training/file_checkpoint_store.hpp>
#include <jyppx/liuding/training/telemetry_adapter.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
namespace experiment = jyppx::liuding::experiment;
namespace training = jyppx::liuding::training;
using Json = nlohmann::json;

constexpr auto kRunId = "run.process-recovery";
constexpr auto kMetricId = "training.validation.loss";
constexpr std::size_t kMaximumReportBytes = 256U * 1024U;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

std::string utf8_path(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::uint64_t process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(::_getpid());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

void report(const Json& value) {
  const auto line = value.dump();
  require(line.size() <= kMaximumReportBytes, "recovery report exceeded bound");
  std::cout << line << '\n' << std::flush;
  require(static_cast<bool>(std::cout), "could not publish recovery report");
}

std::string hex(std::span<const std::byte> bytes) {
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  result.reserve(bytes.size() * 2U);
  for (const auto byte : bytes) {
    const auto value = std::to_integer<unsigned>(byte);
    result.push_back(alphabet[value >> 4U]);
    result.push_back(alphabet[value & 15U]);
  }
  return result;
}

Json observation(const jyppx::liuding::TelemetryEnvelope& envelope) {
  const auto canonical = jyppx::liuding::encode_telemetry_envelope(envelope);
  return {{"canonical_hex", hex(canonical)},
          {"canonical_bytes", canonical.size()},
          {"canonical_sha256", jyppx::liuding::sha256_hex(canonical)},
          {"content_digest", jyppx::liuding::telemetry_content_digest(envelope).hex},
          {"envelope",
           Json::parse(experiment::telemetry_projection_json(envelope))}};
}

Json checkpoint_evidence(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  require(size > 0U && size <= 1024U * 1024U,
          "fixture checkpoint must be nonempty and bounded");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "could not open saved checkpoint");
  input.read(reinterpret_cast<char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  require(static_cast<bool>(input), "could not read complete saved checkpoint");
  return {{"path", utf8_path(path)},
          {"bytes", size},
          {"sha256", jyppx::liuding::sha256_hex(bytes)}};
}

Json outbox_state(const training::EventOutbox& outbox) {
  return {{"stream_id", outbox.telemetry_stream_id()},
          {"pending_events", outbox.pending_events()},
          {"pending_bytes", outbox.pending_bytes()},
          {"last_enqueued", outbox.last_enqueued_event_index()},
          {"last_delivered", outbox.last_delivered_event_index()},
          {"closed", outbox.closed()}};
}

Json store_state(const experiment::SQLiteRunStore& store) {
  const auto events = store.events(kRunId);
  const auto metrics = store.metrics(kRunId, {.metric_id = kMetricId});
  require(!events.next_after && !metrics.next_after,
          "bounded recovery fixture unexpectedly requires another page");
  auto stored_events = Json::array();
  for (const auto& event : events.events) {
    stored_events.push_back({{"run_sequence", event.run_sequence},
                             {"observation", observation(event.envelope)}});
  }
  auto stored_metrics = Json::array();
  for (const auto& metric : metrics.points) {
    // Reuse the production JSON projection for all indexed semantic fields.
    jyppx::liuding::TelemetryEnvelope projected;
    projected.identity = metric.event_identity;
    projected.time = metric.time;
    projected.correlation = metric.correlation;
    projected.attributes = metric.attributes;
    projected.payload = jyppx::liuding::TelemetryMetricBatch{.points = {metric.point}};
    const auto json =
        Json::parse(experiment::telemetry_projection_json(projected));
    stored_metrics.push_back(
        {{"run_sequence", metric.cursor.run_sequence},
         {"point_index", metric.cursor.point_index},
         {"identity", json.at("identity")},
         {"time", json.at("time")},
         {"correlation", json.at("correlation")},
         {"attributes", json.at("attributes")},
         {"point", json.at("payload").at("points").at(0U)},
         {"resource_id",
          metric.resource_id ? Json(*metric.resource_id) : Json(nullptr)}});
  }
  const auto integrity = store.integrity_check();
  const auto metadata = store.find(kRunId);
  require(metadata.has_value(), "recovery fixture run disappeared");
  return {{"database_path", utf8_path(store.database_path())},
          {"run_status", experiment::run_status_name(metadata->status)},
          {"event_snapshot_through", events.snapshot_through},
          {"metric_snapshot_through", metrics.snapshot_through},
          {"events", std::move(stored_events)},
          {"metrics", std::move(stored_metrics)},
          {"integrity_ok", integrity.ok()},
          {"integrity_issues", integrity.issues}};
}

Json dispositions(const jyppx::liuding::TelemetryEmitResult& result) {
  auto values = Json::array();
  for (const auto disposition : result.dispositions)
    values.push_back(static_cast<unsigned>(disposition));
  return values;
}

enum class Fence { none, before_acceptance, after_commit_before_ack };

class ObservedSink final : public jyppx::liuding::TelemetrySink {
 public:
  ObservedSink(std::shared_ptr<experiment::SQLiteRunTelemetrySink> sink,
               Fence fence)
      : sink_(std::move(sink)), fence_(fence) {}

  const jyppx::liuding::TelemetrySinkDescriptor& descriptor() const noexcept override {
    return sink_->descriptor();
  }

  jyppx::liuding::TelemetryEmitResult emit(
      std::span<const jyppx::liuding::TelemetryEnvelope> envelopes, std::stop_token stop,
      jyppx::liuding::TelemetryDeadline deadline) override {
    require(envelopes.size() == 1U, "expected exactly one outbox observation");
    const auto captured = observation(envelopes.front());
    if (fence_ == Fence::before_acceptance) fence_callback(captured, {});
    auto result = sink_->emit(envelopes, stop, deadline);
    emissions.push_back(
        {{"observation", captured}, {"dispositions", dispositions(result)}});
    if (fence_ == Fence::after_commit_before_ack) {
      require(result.dispositions.size() == 1U &&
                  result.dispositions.front() ==
                      jyppx::liuding::TelemetryDisposition::accepted,
              "ACK-loss fence requires a real newly committed acceptance");
      fence_callback(captured, result);
    }
    return result;
  }

  jyppx::liuding::TelemetryFlushResult flush(
      std::stop_token stop, jyppx::liuding::TelemetryDeadline deadline) override {
    return sink_->flush(stop, deadline);
  }

  jyppx::liuding::TelemetryFlushResult close(
      std::stop_token stop, jyppx::liuding::TelemetryDeadline deadline) override {
    return sink_->close(stop, deadline);
  }

  std::function<void(const Json&, const jyppx::liuding::TelemetryEmitResult&)>
      fence_callback;
  Json emissions = Json::array();

 private:
  std::shared_ptr<experiment::SQLiteRunTelemetrySink> sink_;
  Fence fence_;
};

// Normal stack unwinding leaves evidence; a real forced kill bypasses it.
class CleanupMarker final {
 public:
  explicit CleanupMarker(std::filesystem::path path) : path_(std::move(path)) {}
  ~CleanupMarker() {
    std::ofstream marker(path_, std::ios::binary);
    marker << "producer stack unwound\n";
  }
  CleanupMarker(const CleanupMarker&) = delete;
  CleanupMarker& operator=(const CleanupMarker&) = delete;

 private:
  std::filesystem::path path_;
};

training::TrainingTelemetryAdapterOptions adapter_options() {
  training::TrainingTelemetryAdapterOptions options;
  options.project_id = "project.process-recovery";
  options.attempt_id = "attempt.original";
  options.metrics["validation.loss"] = {
      .definition = {.metric_id = kMetricId,
                     .unit = jyppx::liuding::TelemetryMetricUnit::dimensionless,
                     .aggregation = jyppx::liuding::TelemetryAggregation::mean,
                     .direction = jyppx::liuding::TelemetryMetricDirection::minimize,
                     .context_keys = {"split"}},
      .context = {{"split", "validation"}}};
  return options;
}

training::CallbackEvent callback_event(bool first) {
  return {.schema_version = 1U,
          .kind = first ? training::CallbackEventKind::run_started
                        : training::CallbackEventKind::epoch_finished,
          .phase = first ? training::CallbackPhase::preparing
                         : training::CallbackPhase::training,
          .event_index = first ? 1U : 2U,
          .run_id = kRunId,
          .trace_id = "trace.process-recovery",
          .epoch = first ? 0U : 1U,
          .global_step = first ? 0U : 1U,
          .metrics = {{"validation.loss", first ? 0.25 : 0.125}}};
}

training::CallbackDispatchResult callback_decisions() {
  training::CallbackCommand command{
      .kind = training::CallbackCommandKind::request_checkpoint,
      .callback_id = "callback.recovery-audit",
      .reason =
          "Retain the original committed decision across process recovery"};
  return {.commands = {command}, .effective_command = command};
}

int execute(std::string_view mode, const std::filesystem::path& root) {
  const bool preparing =
      mode == "prepare-pending" || mode == "prepare-ack-loss";
  require(preparing || mode == "inspect" || mode == "recover" ||
              mode == "advance" || mode == "verify-final",
          "unknown recovery-child mode");
  require(std::filesystem::is_directory(root),
          "fixture root must already exist");
  const auto pending_path = root / "pending.jyck";
  if (preparing) {
    require(!std::filesystem::exists(root / "store") &&
                !std::filesystem::exists(pending_path),
            "producer fixture requires a fresh root");
  } else {
    require(std::filesystem::is_regular_file(root / "store" / "runs.sqlite3") &&
                std::filesystem::is_regular_file(pending_path),
            "recovery fixture must have been prepared by the producer");
  }
  std::unique_ptr<CleanupMarker> cleanup;
  if (preparing)
    cleanup = std::make_unique<CleanupMarker>(root / "producer-cleanup.marker");
  auto store = std::make_shared<experiment::SQLiteRunStore>(root / "store");
  if (preparing) {
    store->start({.run_id = kRunId,
                  .project_id = "project.process-recovery",
                  .experiment_id = "experiment.process-recovery",
                  .display_name = "Producer process recovery",
                  .status = experiment::RunStatus::running,
                  .created_at_unix_ms = 100U,
                  .updated_at_unix_ms = 100U});
  }
  Json result{{"schema_version", 1U},
              {"mode", mode},
              {"process_id", process_id()},
              {"root", utf8_path(root)}};
  if (mode == "inspect") {
    result["phase"] = "completed";
    result["store"] = store_state(*store);
    report(result);
    return 0;
  }
  auto real_sink = std::make_shared<experiment::SQLiteRunTelemetrySink>(
      store, kRunId, "sink.sqlite", "local.process-recovery");
  const auto fence = mode == "prepare-pending" ? Fence::before_acceptance
                     : mode == "prepare-ack-loss"
                         ? Fence::after_commit_before_ack
                         : Fence::none;
  auto observed_sink = std::make_shared<ObservedSink>(real_sink, fence);
  auto outbox = std::make_shared<training::EventOutbox>(
      std::vector<std::shared_ptr<jyppx::liuding::TelemetrySink>>{observed_sink},
      adapter_options());
  training::CheckpointParticipantRegistry registry;
  registry.register_participant(outbox);
  training::CheckpointCoordinator coordinator(registry);
  if (preparing) {
    outbox->enqueue(callback_event(true), callback_decisions());
    training::FileCheckpointStore checkpoint(pending_path);
    coordinator.save(checkpoint);
    observed_sink->fence_callback =
        [&](const Json& captured, const jyppx::liuding::TelemetryEmitResult& receipt) {
          result["phase"] = "ready-for-forced-kill";
          result["checkpoint"] = checkpoint_evidence(pending_path);
          result["outbox"] = outbox_state(*outbox);
          result["observation"] = captured;
          result["internal_dispositions"] = dispositions(receipt);
          result["store"] = store_state(*store);
          report(result);
          // The supervisor keeps stdin open but never sends data. EOF or a byte
          // fails the fixture and unwinds the marker, rather than imitating a
          // kill.
          char input = 0;
          std::cin.get(input);
          throw std::runtime_error(
              "supervisor fence must end only by forced kill");
        };
    outbox->drain_one();
    throw std::runtime_error("producer unexpectedly escaped its kill fence");
  }

  result["created_stream_id"] = outbox->telemetry_stream_id();
  const auto source_path = mode == "recover"   ? pending_path
                           : mode == "advance" ? root / "delivered.jyck"
                                               : root / "advanced.jyck";
  result["source_checkpoint"] = checkpoint_evidence(source_path);
  training::FileCheckpointStore source(source_path);
  coordinator.restore(source, jyppx::liuding::RestorePolicy::exact_only);
  result["restored"] = outbox_state(*outbox);
  if (mode == "recover") {
    require(outbox->pending_events() == 1U &&
                outbox->last_delivered_event_index() == 0U,
            "original checkpoint must retain its unacknowledged event");
  } else {
    require(outbox->pending_events() == 0U,
            "delivered checkpoint unexpectedly retained an event");
  }
  if (mode == "advance") {
    require(outbox->last_enqueued_event_index() == 1U &&
                outbox->last_delivered_event_index() == 1U,
            "advance requires the recovered first event");
    outbox->enqueue(callback_event(false), callback_decisions());
    result["after_enqueue"] = outbox_state(*outbox);
  }
  if (mode == "verify-final") {
    require(outbox->last_enqueued_event_index() == 2U &&
                outbox->last_delivered_event_index() == 2U,
            "final checkpoint must contain both delivered cursors");
  }
  outbox->drain();
  result["after"] = outbox_state(*outbox);
  if (mode != "verify-final") {
    const auto saved_path =
        root / (mode == "recover" ? "delivered.jyck" : "advanced.jyck");
    training::FileCheckpointStore saved(saved_path);
    coordinator.save(saved);
    result["saved_checkpoint"] = checkpoint_evidence(saved_path);
  }
  outbox->close();
  const auto counters = real_sink->flush();
  result["closed"] = outbox->closed();
  result["emissions"] = observed_sink->emissions;
  result["exported_events"] = counters.exported_events;
  result["sampled_out_events"] = counters.sampled_out_events;
  result["unsupported_events"] = counters.unsupported_events;
  result["store"] = store_state(*store);
  result["phase"] = "completed";
  report(result);
  return 0;
}

int guarded_execute(std::string_view mode, const std::filesystem::path& root) {
  try {
    return execute(mode, std::filesystem::absolute(root));
  } catch (const std::exception& error) {
    std::cerr << "recovery-child: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  if (argc != 3) return 2;
  return guarded_execute(utf8_path(std::filesystem::path(argv[1])),
                         std::filesystem::path(argv[2]));
}
#else
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  return guarded_execute(argv[1], std::filesystem::path(argv[2]));
}
#endif
