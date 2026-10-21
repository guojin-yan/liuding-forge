// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0.
#include "resource_support.hpp"

#include <jyppx/experiment/artifact_store.hpp>
#include <jyppx/experiment/run_store.hpp>
#include <jyppx/experiment/telemetry_exporters.hpp>
#include <jyppx/training/checkpoint_coordinator.hpp>
#include <jyppx/training/file_checkpoint_store.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace {
using namespace jyppx::benchmarks;
namespace ex = jyppx::experiment;
namespace tr = jyppx::training;

std::vector<std::byte> payload(std::size_t size, unsigned salt) {
  std::vector<std::byte> result(size);
  for (std::size_t i = 0; i < size; ++i)
    result[i] = static_cast<std::byte>((i * 37U + i / 251U + salt) % 256U);
  return result;
}

void storage(const Arguments& args) {
  constexpr std::size_t count = 8, size = 256U << 10U;
  auto sample = report(args,
                       {{"objects", count},
                        {"bytes_per_object", size},
                        {"generator", "(i*37+i/251+object)%256"},
                        {"durability", "SQLite FULL plus CAS publication"}},
                       count);
  std::unique_ptr<ex::SQLiteRunStore> store;
  auto& metrics = sample["metrics"];
  metrics["load_seconds"] = metric(
      seconds([&] {
        store =
            std::make_unique<ex::SQLiteRunStore>(args.directory / "run-store");
      }),
      "seconds", "lower", false);
  const auto run = store->start({.run_id = "resource-run",
                                 .project_id = "resource-project",
                                 .experiment_id = "resource-experiment",
                                 .display_name = "fixed resource benchmark",
                                 .status = ex::RunStatus::running,
                                 .created_at_unix_ms = 100,
                                 .updated_at_unix_ms = 100});
  require(run.run_id == "resource-run", "run start identity mismatch");
  std::vector<std::vector<std::byte>> objects;
  for (std::size_t index = 0; index < count; ++index)
    objects.push_back(payload(size, index));
  std::vector<ex::RunArtifactReference> references;
  const auto publish = seconds([&] {
    for (std::size_t index = 0; index < count; ++index) {
      references.push_back(store->put_artifact(
          run.run_id, "object-" + std::to_string(index),
          ex::RunArtifactKind::artifact, "resource", objects[index],
          {.logical_name = "fixed-object",
           .producer = "jyppx.resource-benchmark",
           .expected_size_bytes = size},
          100 + index));
    }
  });
  metrics["publish_seconds"] = metric(publish, "seconds");
  metrics["throughput_bytes_per_second"] =
      metric(count * size / publish, "bytes/second", "higher");
  require(store->references(run.run_id).size() == count &&
              store->integrity_check().ok(),
          "SQLite/CAS integrity failed");
  store.reset();
  std::uint64_t read_bytes = 0;
  metrics["reopen_read_seconds"] = metric(
      seconds([&] {
        ex::SQLiteRunStore reopened(args.directory / "run-store");
        require(reopened.find(run.run_id).has_value(), "durable run missing");
        const auto stored = reopened.references(run.run_id);
        require(stored.size() == count, "durable references missing");
        ex::FileArtifactStore cas(reopened.artifact_root());
        for (std::size_t index = 0; index < count; ++index) {
          const auto& reference = references[index];
          auto reader = cas.open(reference.artifact.digest);
          require(reader->size_bytes() == size, "artifact length mismatch");
          std::vector<std::byte> buffer(size);
          std::size_t offset = 0;
          while (offset < size) {
            const auto got = reader->read(std::span(buffer).subspan(offset));
            require(got > 0, "unexpected artifact EOF");
            offset += got;
          }
          require(buffer == objects[index], "CAS byte verification failed");
          read_bytes += offset;
        }
      }),
      "seconds");
  require(read_bytes == count * size, "artifact read workload incomplete");
  metrics["artifact_payload_bytes"] = metric(count * size, "bytes");
  metrics["output_bytes"] = metric(directory_bytes(args.directory), "bytes");
  sample["execution"]["path"] =
      "FileArtifactStore plus SQLiteRunStore; local filesystem";
  sample["validation"]["output_kind"] = "exact binary roundtrip";
  finish(sample, references.size());
}

class BlobParticipant final : public jyppx::CheckpointParticipant {
 public:
  std::vector<std::byte> bytes;
  explicit BlobParticipant(std::vector<std::byte> data)
      : bytes(std::move(data)) {}
  std::string participant_id() const override { return "resource-blob"; }
  jyppx::StateManifest describe_state() const override {
    return {.participant_id = participant_id(),
            .schema_version = 1,
            .restore_level = jyppx::RestoreLevel::exact,
            .fields = {
                {.name = "payload", .max_bytes = 1U << 20U, .required = true}}};
  }
  void save_state(jyppx::StateWriter& writer) const override {
    writer.write_blob("payload", bytes);
  }
  void load_state(const jyppx::StateReader& reader,
                  jyppx::RestorePolicy policy) override {
    require(
        jyppx::restore_policy_accepts(policy, reader.manifest().restore_level),
        "restore policy mismatch");
    const auto saved = reader.read_blob("payload", 1U << 20U);
    require(saved.size() == (1U << 20U), "checkpoint payload truncated");
    bytes.assign(saved.begin(), saved.end());
  }
};

void checkpoint(const Arguments& args) {
  constexpr std::size_t count = 4, size = 1U << 20U;
  auto sample = report(args,
                       {{"checkpoints", count},
                        {"payload_bytes", size},
                        {"generator", "(i*37+i/251+checkpoint)%256"},
                        {"synchronize", true}},
                       count);
  auto& metrics = sample["metrics"];
  auto participant = std::make_shared<BlobParticipant>(payload(size, 0));
  tr::CheckpointParticipantRegistry registry;
  registry.register_participant(participant);
  tr::CheckpointCoordinator coordinator(registry);
  double save_time = 0, restore_time = 0;
  for (std::size_t index = 0; index < count; ++index) {
    participant->bytes = payload(size, index);
    const auto expected = participant->bytes;
    const auto path =
        args.directory / ("checkpoint-" + std::to_string(index) + ".bin");
    save_time += seconds([&] {
      tr::FileCheckpointStore store(path, {.synchronize = true});
      const auto descriptor = coordinator.save(store);
      require(descriptor.restore_level == jyppx::RestoreLevel::exact,
              "non-exact checkpoint");
    });
    participant->bytes.assign(size, std::byte{0});
    restore_time += seconds([&] {
      tr::FileCheckpointStore reopened(path);
      static_cast<void>(coordinator.restore(reopened));
    });
    require(participant->bytes == expected,
            "restored checkpoint content mismatch");
  }
  metrics["checkpoint_pause_seconds"] =
      metric(save_time / count, "seconds", "lower", true,
             "mean coordinated save including serialization, hashes, "
             "synchronized publication");
  metrics["restore_seconds"] = metric(restore_time / count, "seconds");
  metrics["throughput_bytes_per_second"] =
      metric(count * size / save_time, "bytes/second", "higher");
  metrics["output_bytes"] = metric(directory_bytes(args.directory), "bytes");
  sample["execution"]["path"] =
      "CheckpointCoordinator plus FileCheckpointStore";
  sample["validation"]["output_kind"] = "exact binary roundtrip";
  finish(sample, count);
}

void telemetry(const Arguments& args) {
  constexpr std::size_t count = 256, encode_rounds = 16;
  auto sample = report(args,
                       {{"events", count},
                        {"encode_rounds", encode_rounds},
                        {"message", "fixed finite local resource event"},
                        {"sink", "JSONL observer; stream flush, no fsync"}},
                       count);
  std::vector<jyppx::TelemetryEnvelope> events;
  for (std::size_t index = 1; index <= count; ++index) {
    events.push_back(
        {.identity = {.stream_id = "resource/attempt.1",
                      .event_id = "event." + std::to_string(index),
                      .idempotency_key = "operation." + std::to_string(index),
                      .sequence = index},
         .time = {.wall_time_unix_ns = 1700000000123456789LL},
         .correlation = {.project_id = "resource-project",
                         .run_id = "resource-run"},
         .reliability = jyppx::TelemetryReliability::required,
         .payload = jyppx::TelemetryLog{jyppx::TelemetryLogSeverity::info,
                                        "fixed finite local resource event"}});
  }
  auto& metrics = sample["metrics"];
  std::size_t encoded_bytes = 0;
  metrics["serialization_seconds"] = metric(
      seconds([&] {
        for (std::size_t round = 0; round < encode_rounds; ++round)
          for (const auto& event : events)
            encoded_bytes += jyppx::encode_telemetry_envelope(event).size();
      }),
      "seconds");
  const ex::JsonlTelemetryOptions options{.path =
                                              args.directory / "events.jsonl"};
  std::unique_ptr<ex::JsonlTelemetrySink> sink;
  metrics["load_seconds"] =
      metric(seconds([&] {
               sink = std::make_unique<ex::JsonlTelemetrySink>(options);
             }),
             "seconds", "lower", false);
  const auto emit = seconds([&] {
    const auto receipts = sink->emit(events);
    require(receipts.dispositions.size() == count &&
                std::all_of(
                    receipts.dispositions.begin(), receipts.dispositions.end(),
                    [](auto value) {
                      return value == jyppx::TelemetryDisposition::accepted;
                    }),
            "JSONL export did not accept all events");
    require(sink->close().exported_events == count,
            "JSONL close count mismatch");
  });
  sink.reset();
  metrics["emit_flush_seconds"] = metric(emit, "seconds");
  metrics["throughput_events_per_second"] =
      metric(count / emit, "events/second", "higher");
  metrics["reopen_replay_seconds"] = metric(
      seconds([&] {
        ex::JsonlTelemetrySink reopened(options);
        require(reopened.recovery().complete_records == count &&
                    reopened.recovery().truncated_tail_bytes == 0,
                "JSONL recovery mismatch");
        const auto receipts = reopened.emit(events);
        require(receipts.dispositions.size() == count &&
                    std::all_of(receipts.dispositions.begin(),
                                receipts.dispositions.end(),
                                [](auto value) {
                                  return value ==
                                         jyppx::TelemetryDisposition::duplicate;
                                }),
                "JSONL replay was not deduplicated");
        require(reopened.close().exported_events == count,
                "JSONL replay changed persisted event count");
      }),
      "seconds");
  metrics["canonical_bytes"] = metric(encoded_bytes / encode_rounds, "bytes");
  metrics["output_bytes"] =
      metric(std::filesystem::file_size(options.path), "bytes");
  sample["execution"]["path"] =
      "foundation canonical envelope codec plus JsonlTelemetrySink";
  sample["validation"]["output_kind"] =
      "verified canonical bytes and deduplicated receipts";
  finish(sample, count);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = arguments(argc, argv);
    require(args.device == "cpu",
            "stable persistence profiles execute on CPU only");
    if (args.profile == "storage")
      storage(args);
    else if (args.profile == "checkpoint")
      checkpoint(args);
    else if (args.profile == "telemetry")
      telemetry(args);
    else
      throw std::runtime_error("unknown stable resource profile");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "resource benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
