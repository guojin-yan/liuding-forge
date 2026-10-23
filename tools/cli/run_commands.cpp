// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>

#include <chrono>
#include <limits>

namespace jyppx::liuding::cli {

Result run_telemetry_command(const Arguments& arguments);

namespace {
using namespace experiment;

std::uint64_t now_ms() {
  const auto value = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
  if (value < 0) throw InvalidArgument("System clock precedes Unix epoch");
  return static_cast<std::uint64_t>(value);
}

RunStatus status_value(std::string_view status) {
  for (const auto candidate :
       {RunStatus::created, RunStatus::running, RunStatus::completed,
        RunStatus::failed, RunStatus::cancelled, RunStatus::interrupted}) {
    if (run_status_name(candidate) == status) return candidate;
  }
  usage("run.status",
        "Use created/running/completed/failed/cancelled/interrupted.");
}

Json metadata_json(const RunMetadata& metadata) {
  Json lineage = Json::object();
  const auto add = [&](const char* name,
                       const std::optional<std::string>& value) {
    lineage[name] = value ? Json(*value) : Json(nullptr);
  };
  add("parent_run_id", metadata.lineage.parent_run_id);
  add("source_digest", metadata.lineage.source_digest);
  add("build_fingerprint", metadata.lineage.build_fingerprint);
  add("config_digest", metadata.lineage.config_digest);
  add("data_manifest_digest", metadata.lineage.data_manifest_digest);
  return Json{{"run_id", metadata.run_id},
              {"project_id", metadata.project_id},
              {"experiment_id", metadata.experiment_id},
              {"display_name", metadata.display_name},
              {"status", run_status_name(metadata.status)},
              {"created_at_unix_ms", metadata.created_at_unix_ms},
              {"updated_at_unix_ms", metadata.updated_at_unix_ms},
              {"lineage", lineage},
              {"tags", metadata.tags}};
}

RunMetadata read_metadata(const Json& value) {
  fields(value,
         {"schema_version", "run_id", "project_id", "experiment_id",
          "display_name", "created_at_unix_ms"},
         {"status", "updated_at_unix_ms", "lineage", "tags"});
  if (unsigned_number(value.at("schema_version"), 1U) != 1U)
    throw CompatibilityError("Unsupported run metadata input schema");
  RunMetadata metadata;
  metadata.run_id = value.at("run_id").get<std::string>();
  metadata.project_id = value.at("project_id").get<std::string>();
  metadata.experiment_id = value.at("experiment_id").get<std::string>();
  metadata.display_name = value.at("display_name").get<std::string>();
  metadata.status = status_value(value.value("status", std::string("created")));
  metadata.created_at_unix_ms =
      unsigned_number(value.at("created_at_unix_ms"),
                      std::numeric_limits<std::uint64_t>::max());
  metadata.updated_at_unix_ms =
      value.contains("updated_at_unix_ms")
          ? unsigned_number(value.at("updated_at_unix_ms"),
                            std::numeric_limits<std::uint64_t>::max())
          : metadata.created_at_unix_ms;
  if (value.contains("tags"))
    metadata.tags = value.at("tags").get<std::map<std::string, std::string>>();
  if (value.contains("lineage")) {
    const auto& lineage = value.at("lineage");
    fields(lineage, {},
           {"parent_run_id", "source_digest", "build_fingerprint",
            "config_digest", "data_manifest_digest"});
    const auto get = [&](const char* name) -> std::optional<std::string> {
      return lineage.contains(name) && !lineage.at(name).is_null()
                 ? std::optional(lineage.at(name).get<std::string>())
                 : std::nullopt;
    };
    metadata.lineage = {.parent_run_id = get("parent_run_id"),
                        .source_digest = get("source_digest"),
                        .build_fingerprint = get("build_fingerprint"),
                        .config_digest = get("config_digest"),
                        .data_manifest_digest = get("data_manifest_digest")};
  }
  validate_run_metadata(metadata);
  return metadata;
}

Json reference_json(const RunArtifactReference& reference) {
  return Json{
      {"reference_id", reference.reference_id},
      {"kind", reference.kind == RunArtifactKind::checkpoint ? "checkpoint"
                                                             : "artifact"},
      {"role", reference.role},
      {"attached_at_unix_ms", reference.attached_at_unix_ms},
      {"artifact",
       {{"sha256", reference.artifact.digest.hex},
        {"size_bytes", reference.artifact.size_bytes},
        {"logical_name", reference.artifact.logical_name},
        {"media_type", reference.artifact.media_type},
        {"producer", reference.artifact.producer}}}};
}

Json backup_json(const RunBackupReport& report) {
  return Json{{"database_bytes", report.database_bytes},
              {"artifact_bytes", report.artifact_bytes},
              {"artifact_count", report.artifact_count}};
}
}  // namespace

Result run_command(const Arguments& arguments) {
  using namespace experiment;
  const auto action = arguments.words()[1];
  if (action == "events" || action == "metrics" || action == "export") {
    return run_telemetry_command(arguments);
  }
  if (action == "restore") {
    arguments.allow({"backup", "root"});
    return {backup_json(SQLiteRunStore::restore_backup(
                std::filesystem::u8path(arguments.require("backup")),
                std::filesystem::u8path(arguments.require("root")))),
            0};
  }
  const auto root = std::filesystem::u8path(arguments.require("root"));
  if (action != "start") {
    if (!std::filesystem::is_regular_file(std::filesystem::path(root) /
                                          "runs.sqlite3")) {
      throw IoError("Run storage database does not exist",
                    "Use a valid storage root or run start first.");
    }
  }
  if (action == "start") {
    arguments.allow({"root", "metadata"});
    const auto metadata = read_metadata(parse_json(
        read_text(std::filesystem::u8path(arguments.require("metadata")))));
    SQLiteRunStore store(root);
    return {metadata_json(store.start(metadata)), 0};
  }
  if (action == "list") {
    arguments.allow(
        {"root", "after", "limit", "project", "experiment", "status"});
    RunListRequest request;
    request.after_run_id = arguments.optional("after").value_or("");
    request.limit =
        static_cast<std::size_t>(arguments.number("limit", 256U, 1024U));
    if (request.limit == 0U)
      usage("run.list", "Supply a positive --limit, at most 1024.");
    request.project_id = arguments.optional("project");
    request.experiment_id = arguments.optional("experiment");
    if (const auto status = arguments.optional("status"))
      request.status = status_value(*status);
    SQLiteRunStore store(root);
    const auto page = store.list(request);
    Json runs = Json::array();
    for (const auto& metadata : page) runs.push_back(metadata_json(metadata));
    return {Json{{"runs", runs},
                 {"next_after", page.size() == request.limit
                                    ? Json(page.back().run_id)
                                    : Json(nullptr)}},
            0};
  }
  if (action == "show") {
    arguments.allow({"root", "id"});
    SQLiteRunStore store(root);
    const auto id = arguments.require("id");
    const auto metadata = store.find(id);
    if (!metadata) throw InvalidArgument("Run ID does not exist");
    Json references = Json::array();
    for (const auto& reference : store.references(id))
      references.push_back(reference_json(reference));
    return {Json{{"run", metadata_json(*metadata)}, {"references", references}},
            0};
  }
  if (action == "transition" || action == "heartbeat") {
    if (action == "transition")
      arguments.allow({"root", "id", "status", "time-ms"});
    else
      arguments.allow({"root", "id", "time-ms"});
    SQLiteRunStore store(root);
    const auto timestamp = arguments.number(
        "time-ms", now_ms(), std::numeric_limits<std::uint64_t>::max());
    const auto metadata =
        action == "transition"
            ? store.transition(arguments.require("id"),
                               status_value(arguments.require("status")),
                               timestamp)
            : store.heartbeat(arguments.require("id"), timestamp);
    return {metadata_json(metadata), 0};
  }
  if (action == "recover") {
    arguments.allow({"root", "stale-ms", "time-ms"});
    static_cast<void>(arguments.require("stale-ms"));
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    SQLiteRunStore store(root);
    return {Json{{"interrupted_runs",
                  store.recover_interrupted(
                      arguments.number("stale-ms", 0U, maximum),
                      arguments.number("time-ms", now_ms(), maximum))}},
            0};
  }
  if (action == "verify") {
    arguments.allow({"root"});
    SQLiteRunStore store(root);
    const auto report = store.integrity_check();
    return {Json{{"valid", report.ok()},
                 {"checked_artifacts", report.checked_artifacts},
                 {"issues", report.issues}},
            report.ok() ? 0 : 1};
  }
  if (action == "backup") {
    arguments.allow({"root", "destination"});
    SQLiteRunStore store(root);
    return {backup_json(store.backup(
                std::filesystem::u8path(arguments.require("destination")))),
            0};
  }
  usage("run",
        "Choose "
        "start/list/show/events/metrics/export/transition/heartbeat/recover/"
        "verify/backup/restore.");
}
}  // namespace jyppx::liuding::cli
