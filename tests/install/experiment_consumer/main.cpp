// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/experiment/config.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

namespace {
struct TempRoot {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("jyppx-experiment-consumer-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  ~TempRoot() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

bool storage_consumer() {
  const TempRoot root;
  const std::string payload = "installed-checkpoint";
  const auto bytes = std::as_bytes(std::span(payload.data(), payload.size()));
  {
    jyppx::liuding::experiment::SQLiteRunStore store(root.path);
    store.start({.run_id = "consumer.run",
                 .project_id = "consumer.project",
                 .experiment_id = "consumer.experiment",
                 .display_name = "Installed storage",
                 .status = jyppx::liuding::experiment::RunStatus::running,
                 .created_at_unix_ms = 1U,
                 .updated_at_unix_ms = 1U,
                 .lineage = {},
                 .tags = {}});
    const auto reference = store.put_artifact(
        "consumer.run", "checkpoint.1",
        jyppx::liuding::experiment::RunArtifactKind::checkpoint, "last", bytes,
        {.logical_name = "last.bin",
         .producer = "install-consumer",
         .expected_size_bytes = bytes.size()},
        2U);
    if (reference.artifact.digest.hex != jyppx::liuding::sha256_hex(bytes))
      return false;
    const auto published = root.path / "published.bin";
    std::ofstream file(published, std::ios::binary);
    file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    file.close();
    if (!file) return false;
    jyppx::liuding::experiment::PublishedFileEvidenceRequest request;
    request.run_id = "consumer.run";
    request.reference_id = "published.1";
    request.path = published;
    request.descriptor = {.logical_name = "published.bin",
                          .producer = "install-consumer",
                          .expected_size_bytes = bytes.size()};
    request.attached_at_unix_ms = 3U;
    request.expected_digest = {.hex = jyppx::liuding::sha256_hex(bytes)};
    request.event_batch_key = "published.event";
    request.event.identity = {.stream_id = "consumer.output",
                              .event_id = "published.1",
                              .idempotency_key = "published.1",
                              .sequence = 1U};
    request.event.correlation.project_id = "consumer.project";
    request.event.correlation.run_id = "consumer.run";
    request.event.payload =
        jyppx::liuding::TelemetryArtifactReference{.role = "output.file"};
    const auto evidence = store.record_published_file_evidence(request);
    if (evidence.reference_replay || evidence.event.receipts.size() != 1U)
      return false;
    if (!std::filesystem::remove(published)) return false;
  }
  jyppx::liuding::experiment::SQLiteRunStore reopened(root.path);
  return reopened.find("consumer.run").has_value() &&
         reopened.find_reference("consumer.run", "published.1").has_value() &&
         reopened.references("consumer.run").size() == 3U &&
         reopened.events("consumer.run").events.size() == 1U &&
         reopened.integrity_check().ok();
}
}  // namespace

int main() {
  c10::impl::GenericDict defaults(c10::StringType::get(), c10::AnyType::get());
  defaults.insert("epochs", std::int64_t{2});
  defaults.insert("token", "secret://env/consumer/token");
  jyppx::liuding::experiment::config::ConfigSchema schema{
      .schema_id = "consumer.config",
      .version = 1,
      .root = jyppx::liuding::ValueSchema::dictionary(
          {{"epochs",
            jyppx::liuding::ValueSchema{
                .kind = jyppx::liuding::ValueKind::integer}},
           {"token",
            jyppx::liuding::ValueSchema{
                .kind = jyppx::liuding::ValueKind::string}}}),
      .defaults = c10::IValue(defaults),
      .environment_bindings = {},
      .secret_paths = {"/token"},
  };
  jyppx::liuding::experiment::config::ConfigResolver resolver(schema);
  resolver.add_cli_override("/epochs", "3");
  const auto resolved = resolver.resolve();
  const auto capability =
      jyppx::liuding::experiment::config::make_config_capability(schema);
  return resolved.integer("/epochs") == 3 &&
                 resolved.explain("/token").redacted &&
                 capability.scope_id == "consumer.config" &&
                 resolved.canonical_json().find("secret://") ==
                     std::string::npos &&
                 storage_consumer()
             ? 0
             : 1;
}
