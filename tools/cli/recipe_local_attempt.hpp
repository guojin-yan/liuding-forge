// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_RECIPE_LOCAL_ATTEMPT_HPP
#define JYPPX_LIUDING_TOOLS_RECIPE_LOCAL_ATTEMPT_HPP

#include "recipe_support.hpp"

#include <jyppx/liuding/experiment/artifact_store.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>

namespace jyppx::liuding::cli {

inline Result prepare_local_recipe(
    const Arguments& arguments, const Json& configuration,
    const Json& data_manifest, std::string_view model,
    std::uint64_t total_steps,
    std::string_view experiment_id = "image-classification") {
  using namespace recipe_support;
  namespace ex = experiment;
  const auto absolute = [&](std::string_view name) {
    const auto path = std::filesystem::u8path(arguments.require(name));
    require_recipe(path.is_absolute(), "Local recipe paths must be absolute");
    return path.lexically_normal();
  };
  const auto executable = absolute("executable");
  const auto root = absolute("root");
  const auto output = absolute("spec");
  const auto store_root = absolute("store");
  require_recipe(!std::filesystem::exists(root) &&
                     std::filesystem::is_directory(root.parent_path()) &&
                     !std::filesystem::exists(output) &&
                     std::filesystem::is_directory(output.parent_path()),
                 "Use unused recipe/spec paths with existing parents");
  const auto executable_sha = file_digest(executable, kExecutableLimit).hex;
  const auto pause = arguments.number("pause-after", 0, total_steps);
  require_recipe(pause == 0 || pause < total_steps,
                 "Local recipe pause must precede the final update");
  const auto timeout = arguments.number("timeout-ms", 3600000, 86400000);
  require_recipe(timeout != 0, "Local recipe timeout must be positive");
  const auto stamp = now_ms();
  const auto source =
      "recipe-inputs." +
      digest(configuration.dump() + "\n" + data_manifest.dump()).hex;
  ex::SQLiteRunStore store(store_root);
  const auto previous = store.find(source);
  const auto created = previous ? previous->created_at_unix_ms : stamp;
  store.start({.run_id = source,
               .project_id = "liuding-forge",
               .experiment_id = std::string(experiment_id),
               .display_name = "Frozen native recipe inputs",
               .status = ex::RunStatus::running,
               .created_at_unix_ms = created,
               .updated_at_unix_ms = created,
               .lineage = {},
               .tags = {}});
  const auto pin = [&](std::string name, const Json& value) {
    const auto content = value.dump();
    require_recipe(content.size() <= (1U << 20U),
                   "Local recipe input exceeds 1 MiB");
    const auto ref = store.put_artifact(
        source, name, ex::RunArtifactKind::artifact, name, bytes(content),
        {.logical_name = name + ".json",
         .media_type = "application/json",
         .producer = "liuding-forge.recipe.local-spec",
         .expected_size_bytes = content.size()},
        created, digest(content));
    return Json{{"run_id", source},
                {"reference_id", name},
                {"sha256", ref.artifact.digest.hex},
                {"size_bytes", ref.artifact.size_bytes}};
  };
  const auto config = pin("configuration", configuration);
  const auto data = pin("data_manifest", data_manifest);
  store.transition(source, ex::RunStatus::completed, now_ms());
  Json argv{"recipe",
            "local-run",
            "--file",
            path_text(std::filesystem::absolute(
                std::filesystem::u8path(arguments.require("file")))),
            "--root",
            path_text(root),
            "--format",
            "json"};
  if (pause != 0) {
    argv.push_back("--pause-after");
    argv.push_back(std::to_string(pause));
  }
  Json training{{"model_id", model},
                {"configuration", config},
                {"data_manifest", data},
                {"resume", nullptr},
                {"outputs",
                 {{{"reference_id", "checkpoint.last"},
                   {"kind", "checkpoint"},
                   {"role", "last"},
                   {"max_size_bytes", kArtifactLimit}},
                  {{"reference_id", "model.weights"},
                   {"kind", "artifact"},
                   {"role", "model.weights"},
                   {"max_size_bytes", kArtifactLimit}},
                  {{"reference_id", "training.result"},
                   {"kind", "artifact"},
                   {"role", "training.result"},
                   {"max_size_bytes", 1U << 20U}}}}};
  if (pause != 0)
    training["retry_resume"] = {{"checkpoint_reference_id", "checkpoint.last"},
                                {"restore_policy", "exact_only"}};
  Json spec{
      {"format", "jyppx.liuding.local_run_spec"},
      {"schema_version", 1},
      {"project_id", "liuding-forge"},
      {"experiment_id", experiment_id},
      {"executable", path_text(executable)},
      {"executable_sha256", executable_sha},
      {"arguments", std::move(argv)},
      {"working_directory", path_text(root.parent_path())},
      {"environment", {{"OMP_NUM_THREADS", "1"}, {"MKL_NUM_THREADS", "1"}}},
      {"inherit_environment", true},
      {"timeout_ms", timeout},
      {"max_attempts", pause == 0 ? 1 : 2},
      {"retry_exit_codes", pause == 0 ? Json::array() : Json{75}},
      {"max_stdout_bytes", 1U << 20U},
      {"max_stderr_bytes", 1U << 20U},
      {"training", std::move(training)}};
  require_recipe(spec.dump().size() <= 65536,
                 "Local recipe spec exceeds 64 KiB");
  write_json(output, spec);
  return {Json{{"spec", path_text(output)}, {"specification", spec}}, 0};
}

// The local worker owns lifecycle, retry admission and cleanup. This adapter
// only verifies its effective inputs and publishes native recipe files.
class RecipeLocalAttempt final {
 public:
  RecipeLocalAttempt(const Json& configuration, const Json& data_manifest,
                     std::string_view model)
      : store_(std::filesystem::u8path(environment("STORE_ROOT"))),
        job_(environment("JOB_ID")),
        attempt_(environment("ATTEMPT_ID")) {
    using namespace recipe_support;
    const auto spec_sha = environment("SPEC_SHA256");
    const auto owner = store_.find(attempt_);
    require_recipe(owner && owner->status == experiment::RunStatus::running &&
                       owner->lineage.parent_run_id == job_ &&
                       owner->lineage.config_digest == spec_sha,
                   "Recipe local-run requires its owned running attempt");
    stamp_ = owner->created_at_unix_ms;
    const auto spec = read_json(job_, "local.spec", spec_sha, 65536);
    require_recipe(spec.at("format") == "jyppx.liuding.local_run_spec" &&
                       spec.at("schema_version") == 1 &&
                       spec.at("project_id") == owner->project_id,
                   "Recipe local attempt specification differs");
    training_ = spec.at("training");
    std::string suffix;
    if (training_.contains("retry_resume")) {
      require_recipe(training_.at("retry_resume") ==
                         Json{{"checkpoint_reference_id", "checkpoint.last"},
                              {"restore_policy", "exact_only"}},
                     "Recipe retry requires checkpoint.last and exact_only");
      const auto index = environment("ATTEMPT_INDEX");
      const auto frozen = read_json(attempt_, "local.training.inputs",
                                    environment("TRAINING_SHA256"), 65536);
      require_recipe(
          frozen.at("format") == "jyppx.liuding.local_training_inputs" &&
              frozen.at("schema_version") == 1 && frozen.at("job_id") == job_ &&
              frozen.at("attempt_id") == attempt_ &&
              frozen.at("spec_sha256") == spec_sha &&
              std::to_string(frozen.at("index").get<std::uint64_t>()) == index,
          "Recipe effective attempt inputs differ");
      for (const auto* name :
           {"model_id", "configuration", "data_manifest", "outputs"})
        require_recipe(frozen.at("training").at(name) == training_.at(name),
                       "Recipe retry changed an immutable training input");
      training_ = frozen.at("training");
      suffix = "." + index;
    }
    require_recipe(training_.at("model_id").get<std::string>() == model,
                   "Recipe local model identity differs");
    check_binding("configuration", suffix, configuration);
    check_binding("data_manifest", suffix, data_manifest);
    const auto& outputs = training_.at("outputs");
    require_recipe(outputs.is_array() && outputs.size() == 3,
                   "Recipe local-run requires its three typed outputs");
    std::set<std::string> seen;
    for (const auto& output : outputs) {
      const auto id = output.at("reference_id").get<std::string>();
      const bool checkpoint = id == "checkpoint.last";
      require_recipe(
          (checkpoint || id == "model.weights" || id == "training.result") &&
              seen.insert(id).second &&
              output.at("kind") == (checkpoint ? "checkpoint" : "artifact") &&
              output.at("role") == (checkpoint ? "last" : id) &&
              !store_.find_reference(attempt_, id),
          "Recipe local output declaration or empty baseline differs");
    }
    if (!training_.at("resume").is_null()) {
      require_recipe(
          training_.at("resume").at("restore_policy") == "exact_only",
          "Recipe local resume requires exact_only");
      const auto& binding = training_.at("resume").at("checkpoint");
      const auto reference =
          store_.find_reference(job_, "local.training.resume" + suffix);
      require_recipe(
          reference &&
              reference->kind == experiment::RunArtifactKind::checkpoint &&
              reference->artifact.digest.hex ==
                  binding.at("sha256").get<std::string>() &&
              reference->artifact.size_bytes ==
                  binding.at("size_bytes").get<std::uint64_t>(),
          "Recipe local checkpoint binding differs");
      resume_ = reference->artifact;
    }
  }

  bool resumes() const noexcept { return resume_.has_value(); }

  // Rebuild only a missing directory from the source attempt's complete
  // snapshot. Retain the new root's lock through the caller's whole run.
  std::unique_ptr<recipe_support::RunLock> restore_workspace(
      const std::filesystem::path& root) {
    using namespace recipe_support;
    namespace fs = std::filesystem;
    if (!resume_ || fs::exists(root)) return {};
    const auto source =
        training_.at("resume").at("checkpoint").at("run_id").get<std::string>();
    const auto reference = store_.find_reference(source, kWorkspace);
    require_recipe(
        reference && reference->role == kWorkspace,
        "Missing recipe workspace snapshot; retain the original directory");
    const auto snapshot =
        read_json(source, kWorkspace, reference->artifact.digest.hex, 65536);
    require_recipe(
        snapshot.at("format") == "jyppx.liuding.recipe_workspace_snapshot" &&
            snapshot.at("schema_version") == 1 &&
            snapshot.at("source_attempt_id") == source &&
            snapshot.at("checkpoint_sha256") == resume_->digest.hex &&
            snapshot.at("checkpoint_size_bytes") == resume_->size_bytes &&
            snapshot.at("build") == parse_json(build_fingerprint_json()),
        "Recipe workspace snapshot differs from checkpoint, source or build");
    const auto& files = snapshot.at("files");
    require_recipe(files.is_array() && !files.empty() && files.size() <= 128,
                   "Recipe workspace snapshot file count exceeds its bound");
    experiment::FileArtifactStore cas(store_.artifact_root(),
                                      {.max_artifact_bytes = kWorkspaceBytes});
    struct Input {
      fs::path path;
      std::unique_ptr<experiment::ArtifactReader> reader;
    };
    std::vector<Input> inputs;
    std::set<std::string> seen;
    std::uint64_t total = 0;
    for (const auto& file : files) {
      const auto relative = file.at("path").get<std::string>();
      const auto path = fs::u8path(relative);
      require_recipe(!path.empty() && !path.is_absolute() &&
                         !path.has_root_name() &&
                         relative.find('\\') == std::string::npos &&
                         relative.find(':') == std::string::npos &&
                         relative.find('\0') == std::string::npos &&
                         path_text(path.lexically_normal()) == relative &&
                         seen.insert(relative).second,
                     "Invalid recipe workspace snapshot path");
      for (const auto& component : path)
        require_recipe(component != "." && component != "..",
                       "Recipe workspace snapshot path escapes its root");
      require_recipe(relative.starts_with("backup/") ||
                         workspace_files().contains(relative),
                     "Unexpected recipe workspace snapshot file");
      const auto id = file.at("reference_id").get<std::string>();
      require_recipe(id == workspace_reference(relative),
                     "Recipe workspace file reference differs");
      const auto bound = store_.find_reference(source, id);
      const auto size = file.at("size_bytes").get<std::uint64_t>();
      require_recipe(bound &&
                         bound->kind == experiment::RunArtifactKind::artifact &&
                         bound->role == kWorkspace &&
                         bound->artifact.digest.hex ==
                             file.at("sha256").get<std::string>() &&
                         bound->artifact.size_bytes == size &&
                         size <= kWorkspaceBytes - total,
                     "Recipe workspace file binding or total size differs");
      auto reader = cas.open(bound->artifact.digest);
      require_recipe(reader->size_bytes() == size,
                     "Recipe workspace file content size differs");
      total += size;
      inputs.push_back({path, std::move(reader)});
    }
    for (const auto* mandatory :
         {"workspace/lineage.json", "workspace/config.json",
          "workspace/dataset.json", "backup/runs.sqlite3",
          "backup/manifest.jyppx-backup"})
      require_recipe(seen.contains(mandatory),
                     "Recipe workspace snapshot lacks a required file");
    experiment::FileArtifactStore checkpoint_cas(
        store_.artifact_root(), {.max_artifact_bytes = kArtifactLimit});
    auto checkpoint = checkpoint_cas.open(resume_->digest);
    require_recipe(checkpoint->size_bytes() == resume_->size_bytes,
                   "Recipe workspace checkpoint size differs");
    // Check existing ancestors before claiming a new directory. Files below
    // the claim are created by this attempt under its retained RunLock.
    auto parent = root.parent_path();
    require_recipe(fs::is_directory(parent),
                   "Recipe workspace parent directory must exist");
    for (auto cursor = parent; !cursor.empty();) {
      require_recipe(!fs::is_symlink(fs::symlink_status(cursor)),
                     "Recipe workspace parent must not traverse links");
#if defined(_WIN32)
      const auto attributes = GetFileAttributesW(cursor.c_str());
      require_recipe(
          attributes != INVALID_FILE_ATTRIBUTES &&
              (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0,
          "Recipe workspace parent must not traverse reparse points");
#endif
      const auto next = cursor.parent_path();
      if (cursor == next) break;
      cursor = next;
    }
    require_recipe(fs::create_directory(root),
                   "Recipe workspace directory was created by another process");
    auto lock = std::make_unique<RunLock>(root);
    const auto backup = root / ".local-resume-backup";
    for (auto& input : inputs) {
      const auto relative = input.path.lexically_relative(
          input.path.begin()->string() == "backup" ? fs::path("backup")
                                                   : fs::path("workspace"));
      const auto target = input.path.begin()->string() == "backup"
                              ? backup / relative
                              : root / relative;
      fs::create_directories(target.parent_path());
      copy_workspace_file(target, *input.reader);
    }
    // Empty CAS directories have no file reference in the snapshot, but are
    // part of the native backup layout even when no object is staged there.
    fs::create_directories(backup / "artifacts" / "objects");
    fs::create_directories(backup / "artifacts" / "tmp");
    experiment::SQLiteRunStore::restore_backup(
        backup, root / "run-store", workspace_options(), {},
        std::chrono::steady_clock::now() + std::chrono::seconds(30));
    copy_workspace_file(root / "latest.jyck", *checkpoint);
    fs::remove_all(backup);
    return lock;
  }

  void verify_resume(const std::filesystem::path& root) const {
    if (!resume_) return;
    const auto path = root / "latest.jyck";
    recipe_support::require_recipe(
        recipe_support::file_digest(path, resume_->size_bytes) ==
                resume_->digest &&
            std::filesystem::file_size(path) == resume_->size_bytes,
        "Recipe workspace checkpoint differs from the frozen exact input");
  }

  void publish_checkpoint(const std::filesystem::path& root, bool completed) {
    publish(root / (completed ? "final.jyck" : "latest.jyck"),
            "checkpoint.last", "application/octet-stream");
    if (!completed) publish_workspace(root);
  }

  void publish_completed(const std::filesystem::path& root) {
    publish_checkpoint(root, true);
    publish(root / "model.safetensors", "model.weights",
            "application/octet-stream");
    publish(root / "result.json", "training.result", "application/json");
  }

 private:
  static constexpr std::string_view kWorkspace = "recipe.workspace.snapshot";
  static constexpr std::uint64_t kWorkspaceBytes = 32ULL << 20U;

  static const std::set<std::string>& workspace_files() {
    static const std::set<std::string> files{
        "workspace/lineage.json",      "workspace/config.json",
        "workspace/dataset.json",      "workspace/build.json",
        "workspace/weight-input.json", "workspace/initialization.json"};
    return files;
  }

  static std::string workspace_reference(std::string_view path) {
    return "recipe.workspace." + recipe_support::digest(path).hex;
  }

  static experiment::SQLiteRunStoreOptions workspace_options() {
    experiment::SQLiteRunStoreOptions options;
    options.max_backup_database_bytes = kWorkspaceBytes;
    options.max_backup_manifest_bytes = 65536;
    options.artifact_limits.max_artifact_bytes = kWorkspaceBytes;
    return options;
  }

  static void copy_workspace_file(const std::filesystem::path& path,
                                  experiment::ArtifactReader& reader) {
    using namespace recipe_support;
    require_recipe(!std::filesystem::exists(path),
                   "Recipe workspace restore never overwrites a file");
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw IoError("Cannot create recipe workspace file");
    std::array<std::byte, 65536> buffer{};
    Sha256 hash;
    std::uint64_t count = 0;
    for (;;) {
      const auto got = reader.read(buffer);
      if (got == 0) break;
      require_recipe(got <= reader.size_bytes() - count,
                     "Recipe workspace content grew during restore");
      stream.write(reinterpret_cast<const char*>(buffer.data()),
                   static_cast<std::streamsize>(got));
      if (!stream) throw IoError("Cannot write recipe workspace file");
      hash.update(std::span(buffer.data(), got));
      count += got;
    }
    stream.close();
    require_recipe(stream && count == reader.size_bytes() &&
                       sha256_digest_hex(hash.finish()) == reader.digest().hex,
                   "Recipe workspace restore content differs");
  }

  void publish_workspace(const std::filesystem::path& root) {
    using namespace recipe_support;
    namespace fs = std::filesystem;
    const auto backup = root / ".local-resume-backup";
    // Bound the source before the native consistent backup creates copies.
    std::uint64_t source_bytes = 0;
    std::size_t source_files = 0;
    for (const auto& entry :
         fs::recursive_directory_iterator(root / "run-store")) {
      require_recipe(!entry.is_symlink(),
                     "Recipe run store must not contain links");
      if (!entry.is_regular_file()) continue;
      const auto size = entry.file_size();
      require_recipe(
          ++source_files <= 128 && size <= kWorkspaceBytes - source_bytes,
          "Recipe run-store snapshot exceeds 128 files or 32 MiB");
      source_bytes += size;
    }
    experiment::SQLiteRunStore inner(root / "run-store", workspace_options());
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    inner.backup(backup, {}, deadline);
    Json files = Json::array();
    std::uint64_t total = 0;
    const auto add = [&](const fs::path& path, const std::string& relative) {
      const auto size = fs::file_size(path);
      require_recipe(files.size() < 128 && size <= kWorkspaceBytes - total,
                     "Recipe workspace snapshot exceeds 128 files or 32 MiB");
      const auto sha = file_digest(path, kWorkspaceBytes);
      const auto id = workspace_reference(relative);
      const auto reference = store_.put_artifact_file(
          attempt_, id, experiment::RunArtifactKind::artifact,
          std::string(kWorkspace), path,
          {.logical_name = path_text(path.filename()),
           .media_type = "application/octet-stream",
           .producer = "liuding-forge.recipe.local-run",
           .expected_size_bytes = size},
          stamp_, sha, {}, deadline);
      files.push_back({{"path", relative},
                       {"reference_id", id},
                       {"sha256", reference.artifact.digest.hex},
                       {"size_bytes", size}});
      total += size;
    };
    for (const auto& name : workspace_files()) {
      const auto path = root / fs::u8path(name).filename();
      if (fs::exists(path)) add(path, name);
    }
    for (const auto& entry : fs::recursive_directory_iterator(backup))
      if (entry.is_regular_file())
        add(entry.path(),
            "backup/" + path_text(entry.path().lexically_relative(backup)));
    const auto checkpoint = store_.find_reference(attempt_, "checkpoint.last");
    require_recipe(checkpoint.has_value(),
                   "Recipe snapshot requires its checkpoint");
    const Json snapshot{
        {"format", "jyppx.liuding.recipe_workspace_snapshot"},
        {"schema_version", 1},
        {"source_attempt_id", attempt_},
        {"checkpoint_sha256", checkpoint->artifact.digest.hex},
        {"checkpoint_size_bytes", checkpoint->artifact.size_bytes},
        {"build", parse_json(read_text(root / "lineage.json")).at("build")},
        {"files", std::move(files)}};
    const auto content = snapshot.dump();
    require_recipe(content.size() <= 65536,
                   "Recipe workspace manifest exceeds 64 KiB");
    store_.put_artifact(attempt_, std::string(kWorkspace),
                        experiment::RunArtifactKind::artifact,
                        std::string(kWorkspace), bytes(content),
                        {.logical_name = "recipe-workspace.json",
                         .media_type = "application/json",
                         .producer = "liuding-forge.recipe.local-run",
                         .expected_size_bytes = content.size()},
                        stamp_, digest(content));
    fs::remove_all(backup);
  }

  static std::string environment(std::string_view name) {
    const auto key = "JYPPX_LIUDING_LOCAL_" + std::string(name);
#if defined(_WIN32)
    const std::wstring wide_key(key.begin(), key.end());
    const auto* value = _wgetenv(wide_key.c_str());
    recipe_support::require_recipe(value && *value,
                                   "Recipe local-run requires local execute");
    return recipe_support::path_text(std::filesystem::path(value));
#else
    const auto* value = std::getenv(key.c_str());
    recipe_support::require_recipe(value && *value,
                                   "Recipe local-run requires local execute");
    return value;
#endif
  }

  Json read_json(std::string_view run, std::string_view id,
                 std::string_view sha, std::uint64_t maximum) {
    using namespace recipe_support;
    const auto reference = store_.find_reference(run, id);
    require_recipe(
        reference && reference->kind == experiment::RunArtifactKind::artifact &&
            reference->artifact.digest.hex == sha &&
            reference->artifact.size_bytes <= maximum,
        "Recipe frozen local reference differs");
    experiment::FileArtifactStore cas(store_.artifact_root(),
                                      {.max_artifact_bytes = maximum});
    auto reader = cas.open(reference->artifact.digest);
    require_recipe(reader->size_bytes() == reference->artifact.size_bytes,
                   "Recipe frozen local input size differs");
    std::string content(static_cast<std::size_t>(reader->size_bytes()), '\0');
    std::size_t offset = 0;
    while (offset < content.size()) {
      const auto count = reader->read(std::as_writable_bytes(
          std::span(content.data() + offset, content.size() - offset)));
      require_recipe(count != 0, "Truncated recipe local input");
      offset += count;
    }
    require_recipe(digest(content).hex == sha,
                   "Recipe frozen local input digest differs");
    return parse_json(content);
  }

  void check_binding(std::string_view name, const std::string& suffix,
                     const Json& actual) {
    const auto& binding = training_.at(name);
    const auto id = "local.training." + std::string(name) + suffix;
    const auto reference = store_.find_reference(job_, id);
    recipe_support::require_recipe(
        reference && reference->artifact.size_bytes ==
                         binding.at("size_bytes").get<std::uint64_t>(),
        "Recipe frozen local input binding size differs");
    recipe_support::require_recipe(
        read_json(job_, id, binding.at("sha256").get<std::string>(),
                  1U << 20U) == actual,
        "Recipe configuration or dataset differs from frozen local input");
  }

  void publish(const std::filesystem::path& path, std::string_view id,
               std::string_view media_type) {
    using namespace recipe_support;
    const bool checkpoint = id == "checkpoint.last";
    const auto output = std::find_if(
        training_.at("outputs").begin(), training_.at("outputs").end(),
        [&](const auto& value) {
          return value.at("reference_id").template get<std::string>() == id;
        });
    const auto maximum =
        output->at("max_size_bytes").template get<std::uint64_t>();
    const auto sha = file_digest(path, maximum);
    store_.put_artifact_file(
        attempt_, std::string(id),
        checkpoint ? experiment::RunArtifactKind::checkpoint
                   : experiment::RunArtifactKind::artifact,
        checkpoint ? "last" : std::string(id), path,
        {.logical_name = path_text(path.filename()),
         .media_type = std::string(media_type),
         .producer = "liuding-forge.recipe.local-run",
         .expected_size_bytes = std::filesystem::file_size(path)},
        stamp_, sha);
  }

  experiment::SQLiteRunStore store_;
  std::string job_;
  std::string attempt_;
  std::uint64_t stamp_ = 0;
  Json training_;
  std::optional<ArtifactRef> resume_;
};
}  // namespace jyppx::liuding::cli
#endif  // JYPPX_LIUDING_TOOLS_RECIPE_LOCAL_ATTEMPT_HPP
