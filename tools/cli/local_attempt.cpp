// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "local_attempt.hpp"

#include "local_process.hpp"
#include "recipe_support.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <limits>
#include <memory>
#include <set>
#include <thread>
#include <vector>

namespace jyppx::liuding::cli {
namespace {
using namespace experiment;
namespace fs = std::filesystem;
using recipe_support::bytes;
using recipe_support::digest;
using recipe_support::now_ms;
using recipe_support::path_text;

constexpr std::uint64_t kSpecBytes = 65536;
constexpr std::uint64_t kCaptureBytes = 8U << 20U;
constexpr std::uint64_t kMaximumAttempts = 16;
constexpr std::string_view kProducer = "jyppx.liuding.cli.local/v1";

void require(bool condition, std::string_view message) {
  if (!condition) throw InvalidArgument(std::string(message));
}

std::string text(const Json& value) {
  const auto result = value.get<std::string>();
  require(result.find('\0') == std::string::npos,
          "Local execution strings must not contain NUL");
  return result;
}

std::uint64_t positive(const Json& value, std::uint64_t maximum) {
  const auto result = unsigned_number(value, maximum);
  require(result != 0, "Local execution bounds must be positive");
  return result;
}

void schema(const Json& value, std::string_view format) {
  require(value.at("format").get<std::string>() == format,
          "Unknown local execution format");
  if (unsigned_number(value.at("schema_version"), 1) != 1)
    throw CompatibilityError("Unsupported local execution schema");
}

void identifier(std::string_view value) {
  require(!value.empty() && value.size() <= 128 &&
              std::all_of(value.begin(), value.end(),
                          [](unsigned char c) {
                            return (c >= 'a' && c <= 'z') ||
                                   (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9') || c == '.' ||
                                   c == '_' || c == '-';
                          }),
          "Local job IDs require 1-128 ASCII letters, digits, '.', '_' or '-'");
}

void ordinary_path(const fs::path& path) {
  require(path.is_absolute(), "Local execution paths must be absolute");
  auto part = path.lexically_normal();
  while (!part.empty()) {
    require(!fs::is_symlink(fs::symlink_status(part)),
            "Local execution paths must not traverse symbolic links");
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(part.c_str());
    require(attributes != INVALID_FILE_ATTRIBUTES &&
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0,
            "Local execution paths must not traverse reparse points");
#endif
    const auto parent = part.parent_path();
    if (parent == part) break;
    part = parent;
  }
}

// These files carry no state. Never unlink them: replacing the inode would
// defeat a still-running owner's OS lock. Both kinds are non-inherited.
class OwnerLock final {
 public:
  OwnerLock(const fs::path& root, std::string_view job, bool worker) {
    const auto directory = root / ".local-attempt-locks";
    fs::create_directories(directory);
    ordinary_path(directory);
    const auto path =
        directory / (digest(job).hex + (worker ? ".worker" : ".dispatch"));
    require(!fs::is_symlink(fs::symlink_status(path)),
            "Local execution lock must not be a symbolic link");
#if defined(_WIN32)
    handle_ = CreateFileW(
        path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      const auto error = GetLastError();
      if (error != ERROR_SHARING_VIOLATION && error != ERROR_LOCK_VIOLATION)
        throw IoError("Cannot open local execution ownership lock");
    } else {
      BY_HANDLE_FILE_INFORMATION info{};
      if (!GetFileInformationByHandle(handle_, &info) ||
          (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
        throw IoError("Invalid local execution ownership lock");
      }
    }
#else
    handle_ =
        ::open(path.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (handle_ < 0)
      throw IoError("Cannot open local execution ownership lock");
    if (::flock(handle_, LOCK_EX | LOCK_NB) != 0) {
      const auto error = errno;
      ::close(handle_);
      handle_ = -1;
      if (error != EWOULDBLOCK && error != EAGAIN)
        throw IoError("Cannot acquire local execution ownership lock");
    }
#endif
  }
  ~OwnerLock() {
#if defined(_WIN32)
    if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
    if (handle_ >= 0) ::close(handle_);
#endif
  }
  bool held() const noexcept {
#if defined(_WIN32)
    return handle_ != INVALID_HANDLE_VALUE;
#else
    return handle_ >= 0;
#endif
  }
  OwnerLock(const OwnerLock&) = delete;
  OwnerLock& operator=(const OwnerLock&) = delete;

 private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int handle_ = -1;
#endif
};

struct TrainingInput {
  std::string run;
  std::string reference;
  ArtifactDigest digest;
  std::uint64_t size = 0;
};

struct TrainingSpec {
  std::string model;
  TrainingInput configuration;
  TrainingInput data_manifest;
  std::optional<TrainingInput> resume;
};

TrainingInput training_input(const Json& value, std::uint64_t limit) {
  fields(value, {"run_id", "reference_id", "sha256", "size_bytes"});
  TrainingInput result;
  result.run = text(value.at("run_id"));
  result.reference = text(value.at("reference_id"));
  identifier(result.run);
  identifier(result.reference);
  result.digest.hex = text(value.at("sha256"));
  validate_artifact_digest(result.digest);
  result.size = positive(value.at("size_bytes"), limit);
  return result;
}

TrainingSpec training_spec(const Json& value) {
  fields(value, {"model_id", "configuration", "data_manifest", "resume"});
  TrainingSpec result;
  result.model = text(value.at("model_id"));
  identifier(result.model);
  result.configuration = training_input(value.at("configuration"), 1U << 20U);
  result.data_manifest = training_input(value.at("data_manifest"), 1U << 20U);
  if (!value.at("resume").is_null()) {
    const auto& resume = value.at("resume");
    fields(resume, {"checkpoint", "restore_policy"});
    require(text(resume.at("restore_policy")) == "exact_only",
            "Local training currently requires explicit exact_only restore");
    result.resume = training_input(resume.at("checkpoint"), 1ULL << 30U);
  }
  return result;
}

struct ResourceRequest {
  std::uint64_t cpu_slots = 0;
  std::uint64_t memory_bytes = 0;
  std::optional<std::string> gpu;
  std::uint64_t gpu_memory_bytes = 0;
};

struct Resources {
  Json capacity;
  ResourceRequest request;
};

Resources resource_spec(const Json& value) {
  fields(value, {"capacity", "request"});
  Resources result;
  result.capacity = value.at("capacity");
  fields(result.capacity, {"cpu_slots", "memory_bytes", "gpus"});
  const auto slots = positive(result.capacity.at("cpu_slots"), 65536);
  const auto memory = positive(result.capacity.at("memory_bytes"),
                               std::numeric_limits<std::int64_t>::max());
  const auto& gpus = result.capacity.at("gpus");
  require(gpus.is_object() && gpus.size() <= 16,
          "Resource capacity allows at most 16 explicit GPU identities");
  for (const auto& [id, bytes] : gpus.items()) {
    identifier(id);
    static_cast<void>(
        positive(bytes, std::numeric_limits<std::int64_t>::max()));
  }
  const auto& request = value.at("request");
  fields(request, {"cpu_slots", "memory_bytes", "gpu"});
  result.request.cpu_slots = positive(request.at("cpu_slots"), slots);
  result.request.memory_bytes = positive(request.at("memory_bytes"), memory);
  if (!request.at("gpu").is_null()) {
    const auto& gpu = request.at("gpu");
    fields(gpu, {"id", "memory_bytes"});
    result.request.gpu = text(gpu.at("id"));
    identifier(*result.request.gpu);
    require(gpus.contains(*result.request.gpu),
            "Requested GPU identity is absent from the fixed capacity");
    result.request.gpu_memory_bytes =
        positive(gpu.at("memory_bytes"),
                 positive(gpus.at(*result.request.gpu),
                          std::numeric_limits<std::int64_t>::max()));
  }
  return result;
}

struct Spec {
  Json json;
  std::string canonical;
  std::string sha256;
  std::string executable_sha256;
  std::string project;
  std::string experiment;
  LocalProcessOptions process;
  std::uint64_t maximum_attempts = 1;
  std::uint64_t retry_delay_ms = 0;
  std::optional<std::uint64_t> deadline_unix_ms;
  std::set<std::uint32_t> retry_exit_codes;
  std::optional<TrainingSpec> training;
  std::optional<Resources> resources;
};

Spec read_spec(const Json& value) {
  fields(
      value,
      {"format", "schema_version", "project_id", "experiment_id", "executable",
       "executable_sha256", "arguments", "working_directory", "environment",
       "inherit_environment", "timeout_ms", "max_attempts", "retry_exit_codes"},
      {"retry_delay_ms", "max_stdout_bytes", "max_stderr_bytes",
       "deadline_unix_ms", "training", "resources"});
  schema(value, "jyppx.liuding.local_run_spec");
  Spec result;
  result.json = value;
  result.canonical = value.dump();
  require(result.canonical.size() <= kSpecBytes,
          "Local execution spec exceeds 64 KiB");
  result.sha256 = digest(result.canonical).hex;
  result.project = text(value.at("project_id"));
  result.experiment = text(value.at("experiment_id"));
  identifier(result.project);
  identifier(result.experiment);
  result.executable_sha256 = text(value.at("executable_sha256"));
  validate_artifact_digest({.hex = result.executable_sha256});
  auto& process = result.process;
  process.executable = fs::u8path(text(value.at("executable")));
  process.working_directory = fs::u8path(text(value.at("working_directory")));
  require(process.executable.is_absolute() &&
              process.working_directory.is_absolute(),
          "Local executable and working directory must be absolute");
  require(
      value.at("arguments").is_array() && value.at("arguments").size() <= 256,
      "Local execution allows at most 256 arguments");
  std::size_t launch_bytes = path_text(process.executable).size() + 1;
  for (const auto& item : value.at("arguments")) {
    process.arguments.push_back(text(item));
    launch_bytes += process.arguments.back().size() + 1;
  }
  require(value.at("environment").is_object() &&
              value.at("environment").size() <= 128,
          "Local execution allows at most 128 environment overrides");
  for (const auto& [name, item] : value.at("environment").items()) {
    require(!name.empty() && name.find('=') == std::string::npos &&
                name.find('\0') == std::string::npos &&
                !name.starts_with("JYPPX_LIUDING_LOCAL_"),
            "Invalid or reserved local environment name");
    process.environment[name] = text(item);
    launch_bytes += name.size() + process.environment[name].size() + 2;
  }
  require(launch_bytes <= kSpecBytes,
          "Local explicit argv/environment exceeds 64 KiB");
  process.inherit_environment = value.at("inherit_environment").get<bool>();
  process.timeout = std::chrono::milliseconds(
      positive(value.at("timeout_ms"), 24ULL * 60 * 60 * 1000));
  process.max_stdout_bytes = static_cast<std::size_t>(positive(
      value.value("max_stdout_bytes", Json(1U << 20U)), kCaptureBytes));
  process.max_stderr_bytes = static_cast<std::size_t>(positive(
      value.value("max_stderr_bytes", Json(1U << 20U)), kCaptureBytes));
  result.maximum_attempts =
      positive(value.at("max_attempts"), kMaximumAttempts);
  result.retry_delay_ms =
      unsigned_number(value.value("retry_delay_ms", Json(0)), 60000);
  if (value.contains("deadline_unix_ms"))
    result.deadline_unix_ms = positive(
        value.at("deadline_unix_ms"), std::numeric_limits<std::int64_t>::max());
  require(value.at("retry_exit_codes").is_array() &&
              value.at("retry_exit_codes").size() <= 64,
          "Declare at most 64 retryable native exit codes");
  for (const auto& item : value.at("retry_exit_codes")) {
    const auto code = static_cast<std::uint32_t>(
        positive(item, std::numeric_limits<std::uint32_t>::max()));
    require(result.retry_exit_codes.insert(code).second,
            "Retry exit codes must be unique and nonzero");
  }
  if (value.contains("training"))
    result.training = training_spec(value.at("training"));
  if (value.contains("resources"))
    result.resources = resource_spec(value.at("resources"));
  return result;
}

// Bind to committed run/CAS identities, not mutable external paths. The
// consumer owns model/data interpretation and the actual native restore.
RunMetadata job_metadata(SQLiteRunStore& store, std::string_view job);
Spec frozen_spec(SQLiteRunStore& store, const RunMetadata& job);
Json outcome(SQLiteRunStore& store, std::string_view job, const Spec& spec,
             std::uint64_t index);
RunStatus terminal_status(const Json& result);
std::string attempt_id(std::string_view job, std::uint64_t index);

void training_source(SQLiteRunStore& store, std::string_view project,
                     const TrainingInput& input) {
  const auto source = store.find(input.run);
  require(source && source->project_id == project &&
              is_terminal_run_status(source->status) &&
              source->status != RunStatus::interrupted,
          "Training inputs require a terminal source run in this project");
  const auto kind_tag = source->tags.find("liuding.local.kind");
  if (kind_tag != source->tags.end() && kind_tag->second == "attempt") {
    require(source->lineage.parent_run_id &&
                source->tags.contains("liuding.local.index"),
            "Local training source has incomplete attempt ownership");
    const auto parent = job_metadata(store, *source->lineage.parent_run_id);
    const auto source_spec = frozen_spec(store, parent);
    const auto index = positive(
        parse_json(source->tags.at("liuding.local.index")), kMaximumAttempts);
    require(source->run_id == attempt_id(parent.run_id, index) &&
                source->lineage.config_digest == source_spec.sha256 &&
                source->experiment_id == source_spec.experiment,
            "Local training source differs from its frozen attempt identity");
    const auto confirmed = outcome(store, parent.run_id, source_spec, index);
    require(confirmed.value("process_tree_cleanup_confirmed",
                            !source_spec.resources),
            "Local training source has unconfirmed native process cleanup");
    require(terminal_status(confirmed) == source->status,
            "Local training source requires its confirmed attempt result");
  }
}

std::vector<RunArtifactReference> training_references(
    SQLiteRunStore& store, const Spec& spec,
    std::optional<std::string_view> pinned_job = std::nullopt,
    std::stop_token stop = {}, TelemetryDeadline deadline = std::nullopt) {
  std::vector<RunArtifactReference> result;
  if (!spec.training) return result;
  FileArtifactStore cas(store.artifact_root());
  const auto add = [&](const TrainingInput& input, std::string role,
                       RunArtifactKind kind) {
    if (!pinned_job) training_source(store, spec.project, input);
    const auto reference = store.find_reference(
        pinned_job ? *pinned_job : std::string_view(input.run),
        pinned_job ? std::string_view(role) : std::string_view(input.reference),
        stop, deadline);
    require(reference && reference->kind == kind &&
                (!pinned_job || reference->role == role) &&
                reference->artifact.digest == input.digest &&
                reference->artifact.size_bytes == input.size,
            "Training input differs from its committed run/CAS identity");
    if (pinned_job) {
      const auto reader = cas.open(input.digest, stop, deadline);
      require(reader->size_bytes() == input.size,
              "Training input content size differs from the frozen spec");
    }
    auto bound = *reference;
    bound.reference_id = role;
    bound.role = std::move(role);
    result.push_back(std::move(bound));
  };
  add(spec.training->configuration, "local.training.configuration",
      RunArtifactKind::artifact);
  add(spec.training->data_manifest, "local.training.data_manifest",
      RunArtifactKind::artifact);
  if (spec.training->resume)
    add(*spec.training->resume, "local.training.resume",
        RunArtifactKind::checkpoint);
  return result;
}

void preflight(const Spec& spec) {
  ordinary_path(spec.process.working_directory);
  require(fs::is_directory(spec.process.working_directory),
          "Local working directory must already exist");
  const auto actual = recipe_support::file_digest(spec.process.executable);
  require(actual.hex == spec.executable_sha256,
          "Local executable SHA-256 differs from the frozen spec");
}

RunArtifactReference put_json(SQLiteRunStore& store, std::string_view run,
                              std::string id, std::string role,
                              const Json& value, std::uint64_t timestamp) {
  const auto content = value.dump();
  return store.put_artifact(run, std::move(id), RunArtifactKind::artifact,
                            std::move(role), bytes(content),
                            {.logical_name = "local-execution.json",
                             .media_type = "application/json",
                             .producer = std::string(kProducer),
                             .expected_size_bytes = content.size()},
                            timestamp, digest(content));
}

Json reference_json(SQLiteRunStore& store, std::string_view run,
                    std::string_view id, std::string_view role,
                    std::uint64_t maximum = kSpecBytes) {
  const auto ref = store.find_reference(run, id);
  if (!ref || ref->kind != RunArtifactKind::artifact || ref->role != role ||
      ref->artifact.size_bytes > maximum)
    throw CompatibilityError(
        "Missing or incompatible local execution reference");
  FileArtifactStore cas(store.artifact_root());
  auto reader = cas.open(ref->artifact.digest);
  if (reader->size_bytes() != ref->artifact.size_bytes)
    throw SerializationError("Local execution reference size mismatch");
  std::string result(static_cast<std::size_t>(reader->size_bytes()), '\0');
  std::size_t offset = 0;
  while (offset < result.size()) {
    const auto got = reader->read(
        std::as_writable_bytes(std::span(result.data(), result.size()))
            .subspan(offset));
    if (!got) throw SerializationError("Truncated local execution artifact");
    offset += got;
  }
  if (digest(result) != ref->artifact.digest)
    throw SerializationError("Local execution artifact changed while reading");
  return parse_json(result);
}

RunMetadata job_metadata(SQLiteRunStore& store, std::string_view job) {
  const auto metadata = store.find(job);
  if (!metadata || !metadata->tags.contains("liuding.local.kind") ||
      metadata->tags.at("liuding.local.kind") != "job")
    throw InvalidArgument("Select an existing local execution job");
  return *metadata;
}

Spec frozen_spec(SQLiteRunStore& store, const RunMetadata& job) {
  auto spec =
      read_spec(reference_json(store, job.run_id, "local.spec", "local.spec"));
  require(job.lineage.config_digest == spec.sha256,
          "Local execution spec differs from job lineage");
  return spec;
}

std::string attempt_id(std::string_view job, std::uint64_t index) {
  return "local-" + digest(job).hex + ".attempt-" + std::to_string(index);
}

std::uint64_t process_id() {
#if defined(_WIN32)
  return GetCurrentProcessId();
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
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

std::string_view native_status(LocalProcessStatus status) {
  switch (status) {
    case LocalProcessStatus::exited:
      return "exited";
    case LocalProcessStatus::signalled:
      return "signalled";
    case LocalProcessStatus::cancelled:
      return "cancelled";
    case LocalProcessStatus::timed_out:
      return "timed_out";
    case LocalProcessStatus::output_limit:
      return "output_limit";
  }
  throw InvalidArgument("Unknown native local execution status");
}

bool succeeded(const Json& result) {
  return result.at("status").get<std::string>() == "exited" &&
         unsigned_number(result.at("exit_code"),
                         std::numeric_limits<std::uint32_t>::max()) == 0;
}

RunStatus terminal_status(const Json& result) {
  if (succeeded(result)) return RunStatus::completed;
  return result.at("status").get<std::string>() == "cancelled"
             ? RunStatus::cancelled
             : RunStatus::failed;
}

Json outcome(SQLiteRunStore& store, std::string_view job, const Spec& spec,
             std::uint64_t index) {
  const auto id = attempt_id(job, index);
  auto value = reference_json(store, id, "local.result", "local.result");
  fields(value,
         {"format", "schema_version", "job_id", "attempt_id", "index",
          "spec_sha256", "status", "exit_code", "signal", "process_id",
          "stdout_bytes", "stderr_bytes", "elapsed_ms", "finished_at_unix_ms",
          "stdout_sha256", "stderr_sha256", "failure_reason"},
         {"process_tree_cleanup_confirmed"});
  schema(value, "jyppx.liuding.local_attempt_result");
  require(value.at("job_id").get<std::string>() == job &&
              value.at("attempt_id").get<std::string>() == id &&
              unsigned_number(value.at("index"), kMaximumAttempts) == index &&
              value.at("spec_sha256").get<std::string>() == spec.sha256,
          "Local attempt result identity mismatch");
  const auto state = value.at("status").get<std::string>();
  require(state == "exited" || state == "signalled" || state == "cancelled" ||
              state == "timed_out" || state == "output_limit" ||
              state == "launch_error" || state == "monitor_error",
          "Unknown local attempt result state");
  if (state == "exited")
    static_cast<void>(unsigned_number(
        value.at("exit_code"), std::numeric_limits<std::uint32_t>::max()));
  else
    require(value.at("exit_code").is_null(),
            "Non-exit local result has exit code");
  for (const auto name : {"stdout", "stderr"}) {
    const auto ref = store.find_reference(id, std::string("local.") + name);
    require(
        ref && ref->role == std::string("local.") + name &&
            ref->kind == RunArtifactKind::artifact &&
            ref->artifact.digest.hex ==
                value.at(std::string(name) + "_sha256").get<std::string>() &&
            ref->artifact.size_bytes <= kCaptureBytes,
        "Local attempt output identity mismatch");
    FileArtifactStore cas(store.artifact_root());
    static_cast<void>(cas.open(ref->artifact.digest));
  }
  return value;
}

constexpr std::string_view kResourcePool = "local-resource-pool-v1";
// This name cannot collide with a valid public job ID (which excludes '/').
constexpr std::string_view kResourceLock = "jyppx.liuding/local-resources/v1";

std::string resource_reference(std::uint64_t index, bool released = false) {
  return std::string(released ? "local.resource.release."
                              : "local.resource.lease.") +
         std::to_string(index);
}

Json resource_lease(const RunMetadata& job, const Spec& spec,
                    std::uint64_t index) {
  require(spec.resources.has_value(),
          "Resource lease requires a frozen request");
  return {{"format", "jyppx.liuding.local_resource_lease"},
          {"schema_version", 1},
          {"job_id", job.run_id},
          {"attempt_id", attempt_id(job.run_id, index)},
          {"index", index},
          {"spec_sha256", spec.sha256},
          {"capacity_sha256", digest(spec.resources->capacity.dump()).hex},
          {"request", spec.json.at("resources").at("request")}};
}

// A release is a durable receipt for native cleanup AND the atomically
// committed result/output identities. Free locks, terminal metadata and PIDs
// alone never release a lease. Concurrent identical receipts are idempotent.
bool release_resources(SQLiteRunStore& store, const RunMetadata& job,
                       const Spec& spec, std::uint64_t index) {
  if (!spec.resources) return true;
  const auto lease_id = resource_reference(index);
  const auto lease = store.find_reference(job.run_id, lease_id);
  require(lease && reference_json(store, job.run_id, lease_id, lease_id) ==
                       resource_lease(job, spec, index),
          "Resource reservation differs from its frozen attempt request");
  const auto id = attempt_id(job.run_id, index);
  const auto attempt = store.find(id);
  if (!attempt) return false;
  const auto result_ref = store.find_reference(id, "local.result");
  if (!result_ref) return false;
  const auto result = outcome(store, job.run_id, spec, index);
  if (!result.value("process_tree_cleanup_confirmed", false)) return false;
  require(attempt && attempt->lineage.parent_run_id == job.run_id &&
              attempt->lineage.config_digest == spec.sha256,
          "Resource release attempt lineage mismatch");
  const auto intent = reference_json(store, id, "local.intent", "local.intent");
  require(intent == Json{{"format", "jyppx.liuding.local_attempt_intent"},
                         {"schema_version", 1},
                         {"job_id", job.run_id},
                         {"attempt_id", id},
                         {"index", index},
                         {"spec_sha256", spec.sha256}},
          "Resource release requires the original immutable intent");
  const auto owner = reference_json(store, id, "local.owner", "local.owner");
  fields(owner, {"format", "schema_version", "job_id", "attempt_id", "index",
                 "spec_sha256", "worker_process_id"});
  schema(owner, "jyppx.liuding.local_attempt_owner");
  require(text(owner.at("job_id")) == job.run_id &&
              text(owner.at("attempt_id")) == id &&
              text(owner.at("spec_sha256")) == spec.sha256 &&
              positive(owner.at("index"), kMaximumAttempts) == index &&
              positive(owner.at("worker_process_id"),
                       std::numeric_limits<std::uint64_t>::max()) != 0,
          "Resource release worker identity mismatch");
  const auto reference_digest = [&](std::string_view name) {
    const auto reference = store.find_reference(id, name);
    require(reference.has_value(), "Resource release evidence disappeared");
    return reference->artifact.digest.hex;
  };
  const auto release_id = resource_reference(index, true);
  const Json receipt{{"format", "jyppx.liuding.local_resource_release"},
                     {"schema_version", 1},
                     {"job_id", job.run_id},
                     {"attempt_id", id},
                     {"index", index},
                     {"spec_sha256", spec.sha256},
                     {"lease_sha256", lease->artifact.digest.hex},
                     {"intent_sha256", reference_digest("local.intent")},
                     {"owner_sha256", reference_digest("local.owner")},
                     {"result_sha256", result_ref->artifact.digest.hex},
                     {"stdout_sha256", reference_digest("local.stdout")},
                     {"stderr_sha256", reference_digest("local.stderr")},
                     {"process_tree_cleanup_confirmed", true}};
  put_json(store, job.run_id, release_id, release_id, receipt,
           job.created_at_unix_ms);
  return true;
}

// Caller holds the STORE-wide, non-inherited resource lock until the attempt
// and intent have also committed. The reservation is durable BEFORE either:
// a crash can retain capacity, but cannot start an unaccounted attempt.
std::string reserve_resources(SQLiteRunStore& store, const RunMetadata& job,
                              const Spec& spec, std::uint64_t index) {
  const auto pool = store.find(kResourcePool);
  if (!spec.resources) return pool ? "resource_request_required" : "";
  const Json pool_value{{"format", "jyppx.liuding.local_resource_capacity"},
                        {"schema_version", 1},
                        {"capacity", spec.resources->capacity}};
  const auto pool_digest = digest(pool_value.dump()).hex;
  if (pool) {
    require(pool->tags.contains("liuding.local.kind") &&
                pool->tags.at("liuding.local.kind") == "resource_pool" &&
                pool->lineage.config_digest == pool_digest,
            "STORE resource capacity is immutable; use its exact capacity");
    if (store.find_reference(kResourcePool, "local.resource.capacity"))
      require(reference_json(store, kResourcePool, "local.resource.capacity",
                             "local.resource.capacity") == pool_value,
              "STORE resource capacity identity mismatch");
  }
  auto remaining_slots =
      spec.resources->capacity.at("cpu_slots").get<std::uint64_t>();
  auto remaining_memory =
      spec.resources->capacity.at("memory_bytes").get<std::uint64_t>();
  std::set<std::string> occupied_gpus;
  bool own_reservation = false;
  RunListRequest page;
  page.limit = 256;
  while (true) {
    const auto rows = store.list(page);
    if (rows.empty()) break;
    page.after_run_id = rows.back().run_id;
    for (const auto& other : rows) {
      if (!other.tags.contains("liuding.local.kind") ||
          other.tags.at("liuding.local.kind") != "job" ||
          !store.find_reference(other.run_id, "local.spec"))
        continue;
      const auto frozen = frozen_spec(store, other);
      for (std::uint64_t number = 1; number <= frozen.maximum_attempts;
           ++number) {
        const auto id = attempt_id(other.run_id, number);
        const auto lease_id = resource_reference(number);
        const auto lease = store.find_reference(other.run_id, lease_id);
        const auto attempt = store.find(id);
        if (!lease) {
          if (!attempt) break;
          if (frozen.resources)
            throw CompatibilityError(
                "Managed attempt has no prior reservation");
          if (!store.find_reference(id, "local.result"))
            return "resource_unmanaged_attempt";
          // Older unmetered attempts cannot be ignored merely because their
          // metadata is terminal. Confirm their committed output identities.
          const auto result = outcome(store, other.run_id, frozen, number);
          if (!result.value("process_tree_cleanup_confirmed", false))
            return "resource_unmanaged_attempt";
          continue;
        }
        require(frozen.resources &&
                    frozen.resources->capacity == spec.resources->capacity,
                "Existing resource reservation has a different STORE capacity");
        if (release_resources(store, other, frozen, number)) continue;
        const auto& request = frozen.resources->request;
        require(request.cpu_slots <= remaining_slots &&
                    request.memory_bytes <= remaining_memory,
                "Committed resource reservations exceed their fixed capacity");
        remaining_slots -= request.cpu_slots;
        remaining_memory -= request.memory_bytes;
        if (request.gpu)
          require(occupied_gpus.insert(*request.gpu).second,
                  "Committed resource reservations overlap one exclusive GPU");
        if (other.run_id == job.run_id && number == index)
          own_reservation = true;
      }
    }
  }
  if (!own_reservation) {
    const auto& request = spec.resources->request;
    if (request.cpu_slots > remaining_slots ||
        request.memory_bytes > remaining_memory ||
        (request.gpu && occupied_gpus.contains(*request.gpu)))
      return "resource_unavailable";
  }
  const auto timestamp = pool ? pool->created_at_unix_ms : now_ms();
  if (!pool)
    store.start({.run_id = std::string(kResourcePool),
                 .project_id = "liuding.local.resources",
                 .experiment_id = "capacity",
                 .display_name = "Local resource capacity",
                 .status = RunStatus::running,
                 .created_at_unix_ms = timestamp,
                 .updated_at_unix_ms = timestamp,
                 .lineage = {.parent_run_id = std::nullopt,
                             .source_digest = std::nullopt,
                             .build_fingerprint = std::nullopt,
                             .config_digest = pool_digest,
                             .data_manifest_digest = std::nullopt},
                 .tags = {{"liuding.local.kind", "resource_pool"}}});
  // Repair a crash between pool metadata creation and its first CAS attach.
  put_json(store, kResourcePool, "local.resource.capacity",
           "local.resource.capacity", pool_value, timestamp);
  const auto lease_id = resource_reference(index);
  put_json(store, job.run_id, lease_id, lease_id,
           resource_lease(job, spec, index), job.created_at_unix_ms);
  return "";
}

void finish_run(SQLiteRunStore& store, std::string_view id, RunStatus state) {
  const auto current = store.find(id);
  require(current.has_value(), "Local execution run disappeared");
  store.transition(id, state, std::max(now_ms(), current->updated_at_unix_ms));
}

bool cancelled(SQLiteRunStore& store, std::string_view id) {
  return store.find_reference(id, "local.cancel").has_value();
}

Json describe(SQLiteRunStore& store, const RunMetadata& job, const Spec& spec,
              bool active = false, std::string_view state = {}) {
  Json attempts = Json::array();
  for (std::uint64_t index = 1; index <= spec.maximum_attempts; ++index) {
    const auto id = attempt_id(job.run_id, index);
    const auto metadata = store.find(id);
    if (!metadata) break;
    Json item{{"attempt_id", id},
              {"index", index},
              {"run_status", run_status_name(metadata->status)}};
    if (store.find_reference(id, "local.result"))
      item["result"] = outcome(store, job.run_id, spec, index);
    else
      item["result"] = nullptr;
    if (spec.resources) {
      item["resource_reserved"] =
          store.find_reference(job.run_id, resource_reference(index))
              .has_value();
      item["resource_released"] =
          store.find_reference(job.run_id, resource_reference(index, true))
              .has_value();
    }
    attempts.push_back(std::move(item));
  }
  const auto current = job_metadata(store, job.run_id);
  return {{"schema_version", 1},
          {"job_id", job.run_id},
          {"status", state.empty() ? run_status_name(current.status) : state},
          {"run_status", run_status_name(current.status)},
          {"owner_active", active},
          {"spec_sha256", spec.sha256},
          {"maximum_attempts", spec.maximum_attempts},
          {"cancel_requested", cancelled(store, job.run_id)},
          {"attempts", std::move(attempts)}};
}

// The worker owns the execution lock independently of its caller. A surviving
// worker can commit after the caller dies. If a worker is itself killed before
// committing a result, no later invocation assumes its descendants are gone.
Result worker(SQLiteRunStore& store, const fs::path& root,
              const RunMetadata& job, const Spec& spec, const Arguments& args) {
  args.allow({"root", "id", "attempt", "intent-sha256"});
  const auto index = args.number("attempt", 0, spec.maximum_attempts);
  require(index != 0, "Worker requires an existing positive attempt");
  std::unique_ptr<OwnerLock> owner;
  const auto ownership_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  do {
    owner = std::make_unique<OwnerLock>(root, job.run_id, true);
    if (owner->held()) break;
    owner.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  } while (std::chrono::steady_clock::now() < ownership_deadline);
  require(owner && owner->held(), "Another worker still owns this local job");
  const auto id = attempt_id(job.run_id, index);
  const auto metadata = store.find(id);
  require(job_metadata(store, job.run_id).status == RunStatus::running &&
              metadata && metadata->status == RunStatus::running &&
              metadata->lineage.parent_run_id == job.run_id &&
              metadata->lineage.config_digest == spec.sha256,
          "Worker requires its running frozen attempt");
  const auto intent = store.find_reference(id, "local.intent");
  require(intent && intent->role == "local.intent" &&
              intent->artifact.digest.hex == args.require("intent-sha256"),
          "Worker intent digest does not match the admitted attempt");
  const auto intent_value =
      reference_json(store, id, "local.intent", "local.intent");
  fields(intent_value, {"format", "schema_version", "job_id", "attempt_id",
                        "index", "spec_sha256"});
  schema(intent_value, "jyppx.liuding.local_attempt_intent");
  require(
      intent_value.at("job_id").get<std::string>() == job.run_id &&
          intent_value.at("attempt_id").get<std::string>() == id &&
          intent_value.at("spec_sha256").get<std::string>() == spec.sha256 &&
          unsigned_number(intent_value.at("index"), kMaximumAttempts) == index,
      "Worker intent identity mismatch");
  if (spec.resources) {
    const auto lease_id = resource_reference(index);
    require(
        reference_json(store, job.run_id, lease_id, lease_id) ==
                resource_lease(job, spec, index) &&
            !store.find_reference(job.run_id, resource_reference(index, true)),
        "Worker requires its unreleased, frozen resource reservation");
  }
  require(!store.find_reference(id, "local.owner") &&
              !store.find_reference(id, "local.result"),
          "This immutable local attempt has already been claimed");
  put_json(store, id, "local.owner", "local.owner",
           {{"format", "jyppx.liuding.local_attempt_owner"},
            {"schema_version", 1},
            {"job_id", job.run_id},
            {"attempt_id", id},
            {"index", index},
            {"spec_sha256", spec.sha256},
            {"worker_process_id", process_id()}},
           metadata->created_at_unix_ms);

  LocalProcessResult native;
  std::string status;
  std::string failure_reason;
  std::atomic<bool> monitor_error{false};
  bool runner_entered = false;
  std::stop_source cancellation;
  {
    // Database failure during cancellation monitoring is fail-closed. The
    // native runner still cleans its owned process tree before publication.
    std::jthread monitor([&](std::stop_token stop) {
      while (!stop.stop_requested()) {
        try {
          if (cancelled(store, job.run_id)) {
            cancellation.request_stop();
            return;
          }
        } catch (...) {
          monitor_error = true;
          cancellation.request_stop();
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
      }
    });
    try {
      if (cancelled(store, job.run_id)) {
        native.status = LocalProcessStatus::cancelled;
        native.process_tree_cleanup_confirmed = true;
      } else {
        preflight(spec);
        auto options = spec.process;
        const auto started = std::chrono::steady_clock::now();
        auto deadline = started + options.timeout;
        if (spec.deadline_unix_ms) {
          const auto wall = now_ms();
          const auto left =
              wall < *spec.deadline_unix_ms ? *spec.deadline_unix_ms - wall : 0;
          deadline =
              std::min(deadline, started + std::chrono::milliseconds(left));
        }
        if (spec.training) {
          static_cast<void>(training_references(
              store, spec, job.run_id, cancellation.get_token(), deadline));
          options.environment["JYPPX_LIUDING_LOCAL_STORE_ROOT"] =
              path_text(root);
          options.environment["JYPPX_LIUDING_LOCAL_SPEC_SHA256"] = spec.sha256;
          const auto remaining = deadline - std::chrono::steady_clock::now();
          if (remaining <= std::chrono::steady_clock::duration::zero())
            throw DeadlineExceeded(
                "Training input verification exhausted the attempt deadline");
          options.timeout =
              std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
          if (options.timeout.count() <= 0)
            throw DeadlineExceeded(
                "Training input verification exhausted the attempt deadline");
        }
        options.environment["JYPPX_LIUDING_LOCAL_JOB_ID"] = job.run_id;
        options.environment["JYPPX_LIUDING_LOCAL_ATTEMPT_ID"] = id;
        options.environment["JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX"] =
            std::to_string(index);
        if (cancelled(store, job.run_id)) cancellation.request_stop();
        const auto wall = now_ms();
        if (spec.deadline_unix_ms && wall >= *spec.deadline_unix_ms) {
          native.status = LocalProcessStatus::timed_out;
          native.process_tree_cleanup_confirmed = true;
        } else {
          if (spec.deadline_unix_ms)
            options.timeout = std::min(
                options.timeout,
                std::chrono::milliseconds(*spec.deadline_unix_ms - wall));
          if (spec.resources) {
            const auto& request = spec.resources->request;
            options.environment["JYPPX_LIUDING_LOCAL_CPU_SLOTS"] =
                std::to_string(request.cpu_slots);
            options.environment["JYPPX_LIUDING_LOCAL_MEMORY_BYTES"] =
                std::to_string(request.memory_bytes);
            options.environment["JYPPX_LIUDING_LOCAL_GPU_ID"] =
                request.gpu.value_or("");
            options.environment["JYPPX_LIUDING_LOCAL_GPU_MEMORY_BYTES"] =
                std::to_string(request.gpu_memory_bytes);
          }
          runner_entered = true;
          native = LocalProcessRunner{}.run(options, cancellation.get_token());
        }
      }
      status = std::string(native_status(native.status));
    } catch (const Error& error) {
      native.process_tree_cleanup_confirmed = !runner_entered;
      status = "launch_error";
      if (spec.training && error.code() == ErrorCode::cancelled)
        status = "cancelled";
      if (spec.training && error.code() == ErrorCode::deadline_exceeded)
        status = "timed_out";
      failure_reason = error.what();
      if (failure_reason.size() > 2048)
        failure_reason =
            "Local process launch failed; diagnostic exceeds 2048 bytes";
      native.exit_code.reset();
    }
    monitor.request_stop();
  }
  if (monitor_error.load()) {
    status = "monitor_error";
    failure_reason =
        "Cancellation monitoring failed; child execution was stopped";
    native.exit_code.reset();
    native.signal.reset();
  }
  const auto timestamp = std::max(now_ms(), metadata->created_at_unix_ms);
  Json result{
      {"format", "jyppx.liuding.local_attempt_result"},
      {"schema_version", 1},
      {"job_id", job.run_id},
      {"attempt_id", id},
      {"index", index},
      {"spec_sha256", spec.sha256},
      {"status", status},
      {"exit_code", native.exit_code ? Json(*native.exit_code) : Json(nullptr)},
      {"signal", native.signal ? Json(*native.signal) : Json(nullptr)},
      {"process_id", native.process_id},
      {"stdout_bytes", native.stdout_bytes},
      {"stderr_bytes", native.stderr_bytes},
      {"elapsed_ms", native.elapsed.count()},
      {"finished_at_unix_ms", timestamp},
      {"process_tree_cleanup_confirmed", native.process_tree_cleanup_confirmed},
      {"failure_reason",
       failure_reason.empty() ? Json(nullptr) : Json(failure_reason)},
      {"stdout_sha256", digest(native.stdout_capture).hex},
      {"stderr_sha256", digest(native.stderr_capture).hex}};
  const auto result_text = result.dump();
  const auto upload = [&](std::string name, std::string_view content,
                          std::string media) {
    return RunArtifactUpload{
        .reference_id = "local." + name,
        .kind = RunArtifactKind::artifact,
        .role = "local." + name,
        .bytes = bytes(content),
        .descriptor = {.logical_name = name,
                       .media_type = std::move(media),
                       .producer = std::string(kProducer),
                       .expected_size_bytes = content.size()},
        .expected_digest = digest(content),
        .attached_at_unix_ms = timestamp};
  };
  const std::array uploads{
      upload("stdout", native.stdout_capture, "application/octet-stream"),
      upload("stderr", native.stderr_capture, "application/octet-stream"),
      upload("result", result_text, "application/json")};
  // A complete result and both raw output references commit atomically. A
  // crash before/after the following lifecycle transition is replayable.
  store.put_artifacts(id, uploads);
  static_cast<void>(release_resources(store, job, spec, index));
  finish_run(store, id, terminal_status(result));
  return {{{"attempt_id", id}, {"result_committed", true}}, 0};
}

Result drive(SQLiteRunStore& store, const fs::path& root,
             const RunMetadata& job, const Spec& spec) {
  OwnerLock dispatch(root, job.run_id, false);
  if (!dispatch.held()) return {describe(store, job, spec, true, "active"), 7};
  {
    OwnerLock execution(root, job.run_id, true);
    if (!execution.held())
      return {describe(store, job, spec, true, "active"), 7};
  }
  const auto current_status = job_metadata(store, job.run_id).status;
  if (is_terminal_run_status(current_status))
    return {describe(store, job, spec),
            current_status == RunStatus::completed   ? 0
            : current_status == RunStatus::cancelled ? 130
                                                     : 1};

  for (std::uint64_t index = 1; index <= spec.maximum_attempts; ++index) {
    const auto id = attempt_id(job.run_id, index);
    auto attempt = store.find(id);
    if (attempt && store.find_reference(id, "local.result")) {
      const auto result = outcome(store, job.run_id, spec, index);
      if (!release_resources(store, job, spec, index))
        return {
            describe(store, job, spec, false, "resource_cleanup_unconfirmed"),
            7};
      finish_run(store, id, terminal_status(result));
      if (succeeded(result)) {
        finish_run(store, job.run_id, RunStatus::completed);
        return {describe(store, job, spec), 0};
      }
      const bool retryable =
          result.at("status").get<std::string>() == "exited" &&
          spec.retry_exit_codes.contains(
              result.at("exit_code").get<std::uint32_t>());
      if (cancelled(store, job.run_id) ||
          result.at("status").get<std::string>() == "cancelled") {
        finish_run(store, job.run_id, RunStatus::cancelled);
        return {describe(store, job, spec), 130};
      }
      if (!retryable || index == spec.maximum_attempts) {
        finish_run(store, job.run_id, RunStatus::failed);
        return {describe(store, job, spec), 1};
      }
      continue;
    }
    // Never infer a dead process tree from a free lock, PID or elapsed time.
    // A persisted launch intent without a complete result is ambiguous even
    // when no worker currently holds the lock. Keep it fenced indefinitely.
    if (attempt)
      return {describe(store, job, spec, false, "recovery_required"), 7};
    const bool reserved =
        spec.resources &&
        store.find_reference(job.run_id, resource_reference(index));
    if (cancelled(store, job.run_id) && !reserved) {
      finish_run(store, job.run_id, RunStatus::cancelled);
      return {describe(store, job, spec), 130};
    }
    if (!reserved && index > 1 && spec.retry_delay_ms != 0) {
      const auto until = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(spec.retry_delay_ms);
      while (std::chrono::steady_clock::now() < until) {
        if (cancelled(store, job.run_id)) {
          finish_run(store, job.run_id, RunStatus::cancelled);
          return {describe(store, job, spec), 130};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
      }
    }
    if (!reserved) preflight(spec);
    if (spec.training && !reserved) {
      // Repair a crash before attachment using the same frozen identities.
      // No attempt/intent exists yet; an ambiguous existing attempt is fenced
      // above before any repair or relaunch is considered.
      auto references = training_references(store, spec);
      for (auto& reference : references)
        reference.attached_at_unix_ms = job.created_at_unix_ms;
      // attach() verifies committed CAS content and records the complete
      // relationship batch atomically. Avoid a second pre-attachment hash.
      store.attach(job.run_id, references);
    }
    RunArtifactReference intent;
    {
      OwnerLock resources(root, kResourceLock, false);
      if (!resources.held())
        return {describe(store, job, spec, false, "resource_busy"), 7};
      const auto admission = reserve_resources(store, job, spec, index);
      if (!admission.empty())
        return {describe(store, job, spec, false, admission), 7};
      const auto timestamp = now_ms();
      RunMetadata metadata{
          .run_id = id,
          .project_id = spec.project,
          .experiment_id = spec.experiment,
          .display_name = "Local attempt " + std::to_string(index),
          .status = RunStatus::running,
          .created_at_unix_ms = timestamp,
          .updated_at_unix_ms = timestamp,
          .lineage = job.lineage,
          .tags = {{"liuding.local.kind", "attempt"},
                   {"liuding.local.index", std::to_string(index)}}};
      metadata.lineage.parent_run_id = job.run_id;
      store.start(metadata);
      intent = put_json(store, id, "local.intent", "local.intent",
                        {{"format", "jyppx.liuding.local_attempt_intent"},
                         {"schema_version", 1},
                         {"job_id", job.run_id},
                         {"attempt_id", id},
                         {"index", index},
                         {"spec_sha256", spec.sha256}},
                        timestamp);
    }
    LocalProcessOptions bootstrap;
    bootstrap.executable = this_executable();
    bootstrap.arguments = {"local",
                           "worker",
                           "--root",
                           path_text(root),
                           "--id",
                           job.run_id,
                           "--attempt",
                           std::to_string(index),
                           "--intent-sha256",
                           intent.artifact.digest.hex,
                           "--format",
                           "json"};
    bootstrap.working_directory = root;
    bootstrap.inherit_environment = true;
    bootstrap.max_stdout_bytes = 16384;
    bootstrap.max_stderr_bytes = 16384;
    bootstrap.timeout = spec.process.timeout + std::chrono::seconds(40);
    try {
      static_cast<void>(LocalProcessRunner{}.run(bootstrap));
    } catch (const Error&) {
      // Launch/transport failures cannot prove whether a worker crossed its
      // launch boundary. Only its committed CAS result permits progress.
    }
    {
      OwnerLock execution(root, job.run_id, true);
      if (!execution.held())
        return {describe(store, job, spec, true, "active"), 7};
    }
    if (!store.find_reference(id, "local.result"))
      return {describe(store, job, spec, false, "recovery_required"), 7};
    // Revisit this confirmed result through exactly the recovery path above.
    --index;
  }
  throw InvalidArgument("Local execution exhausted its bounded attempt state");
}
Result operate(std::string_view action, const fs::path& selected_root,
               std::string id, std::optional<Spec> requested,
               std::optional<std::string> parent_run_id,
               const Arguments* arguments = nullptr) {
  identifier(id);
  require(id != kResourcePool, "Local resource capacity ID is reserved");
  const auto root = fs::absolute(selected_root).lexically_normal();
  if (action != "execute")
    require(fs::is_regular_file(root / "runs.sqlite3"),
            "Local execution storage does not exist");
  if (requested) preflight(*requested);
  SQLiteRunStore store(root);
  ordinary_path(root);
  if (action == "execute") {
    require(requested.has_value(), "Local execution requires a frozen spec");
    if (parent_run_id) {
      identifier(*parent_run_id);
      require(store.find(*parent_run_id).has_value(),
              "Local execution parent run does not exist");
    }
    OwnerLock dispatch(root, id, false);
    if (!dispatch.held())
      throw IoError("Another caller owns this local job dispatch");
    const auto existing = store.find(id);
    if (!existing) {
      const auto timestamp = now_ms();
      store.start({.run_id = id,
                   .project_id = requested->project,
                   .experiment_id = requested->experiment,
                   .display_name = "Local execution " + id,
                   .status = RunStatus::running,
                   .created_at_unix_ms = timestamp,
                   .updated_at_unix_ms = timestamp,
                   .lineage = {.parent_run_id = parent_run_id,
                               .source_digest = requested->executable_sha256,
                               .build_fingerprint =
                                   digest(build_fingerprint_json()).hex,
                               .config_digest = requested->sha256,
                               .data_manifest_digest = std::nullopt},
                   .tags = {{"liuding.local.kind", "job"}}});
      put_json(store, id, "local.spec", "local.spec", requested->json,
               timestamp);
    } else {
      const auto job = job_metadata(store, id);
      require(job.lineage.config_digest == requested->sha256,
              "Existing local job has a different frozen spec");
      require(job.lineage.parent_run_id == parent_run_id,
              "Existing local job has a different parent run");
      // Repair the crash between identical job creation and first spec attach;
      // no attempt can be admitted before this immutable reference exists.
      if (!store.find_reference(id, "local.spec"))
        put_json(store, id, "local.spec", "local.spec", requested->json,
                 job.created_at_unix_ms);
    }
  }
  const auto job = job_metadata(store, id);
  const auto spec = frozen_spec(store, job);
  if (action == "worker") {
    require(arguments != nullptr, "Local worker requires CLI arguments");
    return worker(store, root, job, spec, *arguments);
  }
  if (action == "show") {
    OwnerLock owner(root, id, true);
    return {describe(store, job, spec, !owner.held()), 0};
  }
  if (action == "cancel") {
    if (!is_terminal_run_status(job.status) && !cancelled(store, id))
      put_json(store, id, "local.cancel", "local.cancel",
               {{"format", "jyppx.liuding.local_cancel"},
                {"schema_version", 1},
                {"job_id", id},
                {"spec_sha256", spec.sha256}},
               job.created_at_unix_ms);
    return {describe(store, job, spec), 0};
  }
  return drive(store, root, job, spec);
}
}  // namespace

void validate_local_execution_spec(const Json& spec, bool inspect_files) {
  const auto parsed = read_spec(spec);
  if (inspect_files) preflight(parsed);
}

void validate_local_training_source(SQLiteRunStore& store,
                                    std::string_view project_id,
                                    const Json& binding, RunArtifactKind kind) {
  require(
      kind == RunArtifactKind::artifact || kind == RunArtifactKind::checkpoint,
      "Typed training inputs require an artifact or checkpoint kind");
  const auto input = training_input(
      binding, kind == RunArtifactKind::checkpoint ? 1ULL << 30U : 1U << 20U);
  training_source(store, project_id, input);
  const auto reference = store.find_reference(input.run, input.reference);
  require(reference && reference->kind == kind &&
              reference->artifact.digest == input.digest &&
              reference->artifact.size_bytes == input.size,
          "Training input differs from its committed run/CAS identity");
  FileArtifactStore cas(store.artifact_root());
  const auto reader = cas.open(input.digest);
  require(reader->size_bytes() == input.size,
          "Training input content size differs from the frozen spec");
}

Result execute_local_job(const fs::path& root, std::string_view job_id,
                         const Json& spec,
                         std::optional<std::string> parent_run_id) {
  return operate("execute", root, std::string(job_id), read_spec(spec),
                 std::move(parent_run_id));
}

Result control_local_job(std::string_view action, const fs::path& root,
                         std::string_view job_id) {
  require(action == "show" || action == "recover" || action == "cancel",
          "Local control requires show/recover/cancel");
  return operate(action, root, std::string(job_id), std::nullopt, std::nullopt);
}

Result local_attempt_command(const Arguments& arguments) {
  const auto action = arguments.words()[1];
  if (action != "execute" && action != "show" && action != "recover" &&
      action != "cancel" && action != "worker")
    usage("local", "Use execute/show/recover/cancel.");
  if (action == "execute")
    arguments.allow({"root", "id", "spec"});
  else if (action != "worker")
    arguments.allow({"root", "id"});
  std::optional<Spec> requested;
  if (action == "execute")
    requested = read_spec(parse_json(
        read_text(fs::u8path(arguments.require("spec")), kSpecBytes)));
  return operate(action, fs::u8path(arguments.require("root")),
                 arguments.require("id"), std::move(requested), std::nullopt,
                 &arguments);
}

}  // namespace jyppx::liuding::cli
