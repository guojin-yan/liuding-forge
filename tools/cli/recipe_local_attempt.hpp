// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_TOOLS_RECIPE_LOCAL_ATTEMPT_HPP
#define JYPPX_LIUDING_TOOLS_RECIPE_LOCAL_ATTEMPT_HPP

#include "recipe_support.hpp"

#include <jyppx/liuding/experiment/artifact_store.hpp>
#include <jyppx/liuding/experiment/run_store.hpp>

#include <algorithm>
#include <cstdlib>
#include <set>

namespace jyppx::liuding::cli {

inline Result prepare_local_recipe(const Arguments& arguments,
                                   const Json& configuration,
                                   const Json& data_manifest,
                                   std::string_view model,
                                   std::uint64_t total_steps) {
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
  const auto executable_sha = file_digest(executable).hex;
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
               .experiment_id = "image-classification",
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
      {"experiment_id", "image-classification"},
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

  // This first adapter resumes the retained recipe directory; it does not
  // reconstruct a lost workspace or infer ownership from an existing file.
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
  }

  void publish_completed(const std::filesystem::path& root) {
    publish_checkpoint(root, true);
    publish(root / "model.safetensors", "model.weights",
            "application/octet-stream");
    publish(root / "result.json", "training.result", "application/json");
  }

 private:
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
