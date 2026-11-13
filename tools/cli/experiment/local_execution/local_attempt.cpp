// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
#include "local_attempt.hpp"

#include "local_cuda_device.hpp"
#include "local_process.hpp"
#include "recipe_support.hpp"

#include <jyppx/liuding/experiment/local_execution/runner.hpp>
namespace jyppx::liuding::cli {
namespace fs = std::filesystem;
namespace {
Result convert(experiment::LocalJobResult value) {
  return {parse_json(value.json), value.status_code};
}
fs::path this_executable() {
#if defined(_WIN32)
  std::wstring buffer(32768, L'\0');
  const auto length = GetModuleFileNameW(nullptr, buffer.data(),
                                         static_cast<DWORD>(buffer.size()));
  if (!length || length >= buffer.size())
    throw IoError("Cannot identify the local execution worker binary");
  buffer.resize(length);
  return fs::path(buffer);
#elif defined(__linux__)
  return fs::read_symlink("/proc/self/exe");
#else
  throw CompatibilityError("Local execution workers require Windows or Linux");
#endif
}
experiment::LocalJobWorkerLaunch launcher() {
  const auto executable = this_executable();
  return {
      executable,
      recipe_support::file_digest(executable, recipe_support::kExecutableLimit)
          .hex,
      {"local", "worker", "--format", "json"}};
}
}  // namespace
void validate_local_execution_spec(const Json& spec, bool inspect_files) {
  static_cast<void>(
      experiment::local_execution_spec_sha256(spec.dump(), inspect_files));
}
void validate_local_training_source(experiment::SQLiteRunStore& store,
                                    std::string_view project,
                                    const Json& binding,
                                    experiment::RunArtifactKind kind,
                                    std::uint64_t maximum_bytes) {
  experiment::validate_local_training_source_json(
      store, project, binding.dump(), kind, maximum_bytes);
}
Result execute_local_job(const fs::path& root, std::string_view id,
                         const Json& spec, std::optional<std::string> parent) {
  return convert(experiment::execute_local_job(root, id, spec.dump(),
                                               launcher(), std::move(parent)));
}
Result control_local_job(std::string_view action, const fs::path& root,
                         std::string_view id) {
  using C = experiment::LocalJobControl;
  if (action != "show" && action != "recover" && action != "cancel")
    throw InvalidArgument("Local control requires show/recover/cancel");
  return convert(experiment::control_local_job(
      action == "show"      ? C::show
      : action == "recover" ? C::recover
                            : C::cancel,
      root, id,
      action == "recover" ? launcher() : experiment::LocalJobWorkerLaunch{}));
}

Result local_attempt_command(const Arguments& arguments) {
  const auto action = arguments.words()[1];
  if (action == "cuda-devices") {
    arguments.allow({});
    auto devices = Json::array();
    for (const auto& device : local_cuda_devices())
      devices.push_back({{"uuid", device.uuid},
                         {"total_memory_bytes", device.total_memory_bytes},
                         {"free_memory_bytes", device.free_memory_bytes},
                         {"compute_capability",
                          {device.compute_major, device.compute_minor}}});
    return {{{"format", "jyppx.liuding.local_cuda_devices"},
             {"schema_version", 1},
             {"source", "nvml"},
             {"devices", std::move(devices)}},
            0};
  }
  if (action == "cpu-affinity") {
    arguments.allow({});
    return {{{"format", "jyppx.liuding.local_cpu_affinity"},
             {"schema_version", 1},
             {"cpu_ids", local_process_available_cpu_ids()}},
            0};
  }
  if (action == "environment") {
    arguments.allow({"spec"});
    const auto input = read_text(fs::u8path(arguments.require("spec")), 65536);
    return {{{"format", "jyppx.liuding.local_environment"},
             {"schema_version", 1},
#if defined(_WIN32)
             {"platform", "windows-utf16le"},
#else
             {"platform", "posix-bytes"},
#endif
             {"sha256", experiment::local_execution_environment_sha256(input)},
             {"excluded_prefix", "JYPPX_LIUDING_LOCAL_"}},
            0};
  }
  if (action != "execute" && action != "show" && action != "recover" &&
      action != "cancel" && action != "abandon" && action != "worker")
    usage("local",
          "Use "
          "cuda-devices/cpu-affinity/environment/execute/show/recover/cancel/"
          "abandon.");
  if (action == "execute")
    arguments.allow({"root", "id", "spec"});
  else if (action == "abandon")
    arguments.allow({"root", "id", "confirmation"});
  else if (action != "worker")
    arguments.allow({"root", "id"});
  const auto root = fs::u8path(arguments.require("root"));
  const auto id = arguments.require("id");
  if (action == "worker") {
    arguments.allow({"root", "id", "attempt", "intent-sha256"});
    return convert(experiment::run_local_job_worker(
        root, id, arguments.number("attempt", 0, 16),
        arguments.require("intent-sha256")));
  }
  if (action == "abandon")
    return convert(experiment::abandon_local_job(
        root, id,
        read_text(fs::u8path(arguments.require("confirmation")), 65536)));
  if (action == "execute")
    return execute_local_job(
        root, id,
        parse_json(read_text(fs::u8path(arguments.require("spec")), 65536)));
  return control_local_job(action, root, id);
}
}  // namespace jyppx::liuding::cli
