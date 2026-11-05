// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "local_search.hpp"

#include "image_search_bridge.hpp"
#include "local_attempt.hpp"
#include "paired_search_bridge.hpp"
#include "recipe_support.hpp"
#include "transfer_search_bridge.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/version.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <span>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace jyppx::liuding::cli {
namespace {
using namespace experiment;
namespace fs = std::filesystem;
using recipe_support::bytes;
using recipe_support::digest;
using recipe_support::now_ms;
using recipe_support::path_text;
constexpr std::uint64_t kPlanBytes = 256U << 10U;
constexpr std::uint64_t kEvidenceBytes = 1U << 20U;
constexpr std::uint64_t kMaximumTrials = 256;
constexpr std::uint64_t kMaximumSpace = 1000000;
constexpr std::uint64_t kMaximumDuration = 24ULL * 60 * 60 * 1000;
constexpr std::string_view kProducer = "jyppx.liuding.cli.search/v1";
constexpr std::string_view kPairedRecipe = "jyppx.liuding.paired-affine/v1";
constexpr std::string_view kImageRecipe =
    "jyppx.liuding.imagefolder.resnet18/v1";
constexpr std::string_view kTransferSearchRecipe =
    "jyppx.liuding.imagefolder.transfer/v1";
constexpr std::string_view kRecipeBridge = "search.recipe.bridge";

void require(bool condition, std::string_view reason) {
  if (!condition) throw InvalidArgument(std::string(reason));
}

std::string text(const Json &value) {
  const auto result = value.get<std::string>();
  require(result.find('\0') == std::string::npos,
          "Search strings must not contain NUL");
  return result;
}

struct RecipeBridgeProfile {
  std::string_view recipe;
  std::string_view experiment;
  std::string_view native_experiment;
  std::string_view model;
  std::string_view native_run;
  std::string_view manifest_format;
  std::string_view native_stream;
  std::string_view outer_stream;
  std::string_view outer_batch;
  std::string_view display_name;
  bool image;
  bool transfer;
};

const RecipeBridgeProfile &recipe_bridge_profile(const Json &plan) {
  static constexpr std::array<RecipeBridgeProfile, 3> profiles{
      {{kPairedRecipe, "paired-affine", "paired-affine", "paired-affine",
        "run.paired-affine", "jyppx.liuding.local_search_paired_bridge",
        "recipe.paired-affine.evaluation", "search.recipe.paired-affine.",
        "search.recipe.paired-affine.final-evaluation",
        "Confirmed native paired search training", false, false},
       {kImageRecipe, "image-classification", "image-classification",
        "resnet18", "run.image-classification",
        "jyppx.liuding.local_search_image_bridge",
        "recipe.image-classification.evaluation",
        "search.recipe.image-classification.",
        "search.recipe.image-classification.final-evaluation",
        "Confirmed native image search training", true, false},
       {kTransferSearchRecipe, "transfer-learning", "image-classification",
        "resnet50", "run.image-classification",
        "jyppx.liuding.local_search_transfer_bridge",
        "recipe.transfer-learning.evaluation",
        "search.recipe.transfer-learning.",
        "search.recipe.transfer-learning.final-evaluation",
        "Confirmed native frozen ResNet50 transfer search training", true,
        true}}};
  const auto &bridge = plan.at("recipe_bridge");
  for (const auto &profile : profiles) {
    Json expected{{"format", "jyppx.liuding.local_search_recipe_bridge"},
                  {"schema_version", 1},
                  {"recipe", std::string(profile.recipe)},
                  {"seed_policy", "configuration"}};
    if (profile.transfer && bridge.contains("transfer")) {
      const auto &transfer = bridge.at("transfer");
      fields(transfer, {"descriptor", "weight_input_sha256"});
      const auto identity = text(transfer.at("weight_input_sha256"));
      require(transfer.at("descriptor").is_object() && identity.size() == 64 &&
                  std::all_of(identity.begin(), identity.end(),
                              [](char c) {
                                return (c >= '0' && c <= '9') ||
                                       (c >= 'a' && c <= 'f');
                              }),
              "Transfer search requires its complete frozen weight input");
      expected["transfer"] = transfer;
    }
    if (bridge == expected &&
        (!profile.transfer || bridge.contains("transfer")))
      return profile;
  }
  throw InvalidArgument(
      "Search supports only its closed versioned paired-affine, image "
      "ResNet18 or frozen transfer ResNet50 bridge");
}

std::size_t image_objective_classes(
    const TelemetryMetricDefinition &definition) {
  const auto found = definition.parameters.find("classes");
  require(found != definition.parameters.end(),
          "Image search objective requires its class count");
  const auto &encoded = found->second;
  require(!encoded.empty() && encoded.size() <= 4 && encoded.front() != '0' &&
              std::all_of(encoded.begin(), encoded.end(),
                          [](unsigned char c) { return c >= '0' && c <= '9'; }),
          "Image search class count must be a canonical bounded integer");
  const auto count = std::stoull(encoded);
  require(count >= 2 && count <= 1024, "Image search requires 2-1024 classes");
  return static_cast<std::size_t>(count);
}

void identifier(std::string_view value) {
  require(!value.empty() && value.size() <= 128 &&
              std::all_of(value.begin(), value.end(),
                          [](unsigned char c) {
                            return (c >= 'a' && c <= 'z') ||
                                   (c >= 'A' && c <= 'Z') ||
                                   (c >= '0' && c <= '9') || c == '_' ||
                                   c == '-' || c == '.';
                          }),
          "Search identifiers require 1-128 ASCII letters, digits, '.', '_' or "
          "'-'");
}

std::uint64_t positive(const Json &value, std::uint64_t limit) {
  const auto result = unsigned_number(value, limit);
  require(result != 0, "Search budgets must be positive");
  return result;
}

void ordinary(const fs::path &path) {
  require(path.is_absolute(), "Search paths must be absolute");
  for (auto part = path.lexically_normal(); !part.empty();) {
    require(!fs::is_symlink(fs::symlink_status(part)),
            "Search paths must not traverse symbolic links");
#if defined(_WIN32)
    const auto attributes = GetFileAttributesW(part.c_str());
    require(attributes != INVALID_FILE_ATTRIBUTES &&
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0,
            "Search paths must not traverse reparse points");
#endif
    const auto parent = part.parent_path();
    if (parent == part) break;
    part = parent;
  }
}

class SearchLock final {
 public:
  SearchLock(const fs::path &root, std::string_view id) {
    const auto directory = root / ".local-search-locks";
    fs::create_directories(directory);
    ordinary(directory);
    const auto path = directory / digest(id).hex;
    require(!fs::is_symlink(fs::symlink_status(path)), "Invalid search lock");
#if defined(_WIN32)
    handle_ = CreateFileW(
        path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      const auto error = GetLastError();
      if (error != ERROR_SHARING_VIOLATION && error != ERROR_LOCK_VIOLATION)
        throw IoError("Cannot open search ownership lock");
    } else {
      BY_HANDLE_FILE_INFORMATION information{};
      if (!GetFileInformationByHandle(handle_, &information) ||
          (information.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
          information.nNumberOfLinks != 1) {
        CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
        throw IoError("Invalid search ownership lock");
      }
    }
#else
    handle_ =
        ::open(path.c_str(),
               O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK, 0600);
    if (handle_ < 0) throw IoError("Cannot open search ownership lock");
    struct stat information{};
    if (::fstat(handle_, &information) != 0 || !S_ISREG(information.st_mode) ||
        information.st_nlink != 1) {
      ::close(handle_);
      handle_ = -1;
      throw IoError("Invalid search ownership lock");
    }
    if (::flock(handle_, LOCK_EX | LOCK_NB) != 0) {
      const auto error = errno;
      ::close(handle_);
      handle_ = -1;
      if (error != EWOULDBLOCK && error != EAGAIN)
        throw IoError("Cannot acquire search ownership lock");
    }
#endif
  }
  ~SearchLock() {
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
  SearchLock(const SearchLock &) = delete;
  SearchLock &operator=(const SearchLock &) = delete;

 private:
#if defined(_WIN32)
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int handle_ = -1;
#endif
};

void bounded_json(const Json &value, unsigned depth, std::uint64_t &nodes) {
  require(depth <= 16 && ++nodes <= 16384,
          "Search configuration exceeds nesting/node limits");
  if (value.is_string()) static_cast<void>(text(value));
  if (value.is_number_float())
    require(std::isfinite(value.get<double>()),
            "Search numbers must be finite");
  if (value.is_structured())
    for (const auto &child : value) bounded_json(child, depth + 1, nodes);
}

TelemetryMetricDefinition definition(const Json &value) {
  fields(value, {"schema_version", "metric_id", "version", "unit",
                 "aggregation", "direction", "parameters", "context_keys"});
  TelemetryMetricDefinition result;
  result.schema_version =
      static_cast<std::uint32_t>(positive(value.at("schema_version"), 1));
  result.metric_id = text(value.at("metric_id"));
  result.version = static_cast<std::uint32_t>(
      positive(value.at("version"), std::numeric_limits<std::uint32_t>::max()));
  result.unit = static_cast<TelemetryMetricUnit>(positive(value.at("unit"), 9));
  result.aggregation =
      static_cast<TelemetryAggregation>(positive(value.at("aggregation"), 6));
  result.direction =
      static_cast<TelemetryMetricDirection>(positive(value.at("direction"), 3));
  result.parameters = value.at("parameters").get<TelemetryAttributes>();
  const auto keys = value.at("context_keys").get<std::vector<std::string>>();
  result.context_keys = std::set<std::string>(keys.begin(), keys.end());
  require(keys.size() == result.context_keys.size(),
          "Duplicate metric context keys");
  validate_telemetry_metric_definition(result);
  require(result.direction != TelemetryMetricDirection::none &&
              result.unit != TelemetryMetricUnit::unspecified &&
              result.aggregation != TelemetryAggregation::unspecified,
          "Search objectives require explicit units, reduction and direction");
  return result;
}

TelemetryStepKind step_kind(std::string_view name) {
  if (name == "global_step") return TelemetryStepKind::global_step;
  if (name == "epoch") return TelemetryStepKind::epoch;
  if (name == "micro_step") return TelemetryStepKind::micro_step;
  if (name == "sample") return TelemetryStepKind::sample;
  if (name == "elapsed_nanoseconds")
    return TelemetryStepKind::elapsed_nanoseconds;
  throw InvalidArgument("Unknown search objective step kind");
}

// nlohmann/json's noexcept null constructor has an exception-analysis false
// positive; preserve the aggregate's inferred exception contract.
// NOLINTNEXTLINE(bugprone-exception-escape)
struct Plan {
  Json json;
  std::string sha256;
  std::string project;
  std::string experiment;
  TelemetryMetricDefinition metric;
  TelemetryAttributes context;
  TelemetryStepKind step = TelemetryStepKind::global_step;
  std::string metric_sha256;
  std::uint64_t seed = 0;
  std::uint64_t trials = 0;
  std::uint64_t duration = 0;
  std::uint64_t failures = 0;
  std::uint64_t cardinality = 1;
  std::vector<std::uint64_t> sequence;
};

// Standard engine + explicit rejection sampling gives the same sequence on
// MSVC/libstdc++; std::uniform_int_distribution has no cross-library contract.
std::uint64_t uniform(std::mt19937_64 &engine, std::uint64_t bound) {
  const auto threshold = (std::uint64_t{0} - bound) % bound;
  std::uint64_t draw = 0;
  do {
    draw = engine();
  } while (draw < threshold);
  return draw % bound;
}

Plan read_plan(const Json &value) {
  fields(value,
         {"format", "schema_version", "project_id", "experiment_id", "strategy",
          "seed", "max_trials", "max_duration_ms", "max_failures", "space",
          "configuration", "execution", "objective", "inputs", "outputs"},
         {"recipe_bridge"});
  require(text(value.at("format")) == "jyppx.liuding.local_search_spec" &&
              positive(value.at("schema_version"), 1) == 1,
          "Unknown search schema");
  require(value.dump().size() <= kPlanBytes, "Search plan exceeds 256 KiB");
  std::uint64_t nodes = 0;
  bounded_json(value, 0, nodes);
  Plan result;
  result.json = value;
  result.sha256 = digest(value.dump()).hex;
  result.project = text(value.at("project_id"));
  result.experiment = text(value.at("experiment_id"));
  identifier(result.project);
  identifier(result.experiment);
  result.seed = unsigned_number(value.at("seed"),
                                std::numeric_limits<std::uint64_t>::max());
  result.trials = positive(value.at("max_trials"), kMaximumTrials);
  result.duration = positive(value.at("max_duration_ms"), kMaximumDuration);
  result.failures = positive(value.at("max_failures"), kMaximumTrials);
  const auto strategy = text(value.at("strategy"));
  require(strategy == "grid" || strategy == "random",
          "Choose grid or random search");
  const auto &space = value.at("space");
  require(space.is_array() && !space.empty() && space.size() <= 16,
          "Search requires 1-16 finite choice axes");
  require(value.at("configuration").is_object(),
          "Search configuration must be an object");
  std::set<std::string> names, pointers;
  for (const auto &axis : space) {
    fields(axis, {"name", "config_pointer", "values"});
    const auto name = text(axis.at("name"));
    identifier(name);
    require(names.insert(name).second, "Search parameter names must be unique");
    const auto pointer = text(axis.at("config_pointer"));
    require(!pointer.empty() && pointer.front() == '/' &&
                pointers.insert(pointer).second,
            "Search config pointers must be unique, non-root JSON pointers");
    const Json::json_pointer location(pointer);
    const auto &original = value.at("configuration").at(location);
    const auto &choices = axis.at("values");
    require(choices.is_array() && !choices.empty() && choices.size() <= 256,
            "Each search axis requires 1-256 choices");
    std::set<std::string> distinct;
    for (const auto &choice : choices) {
      require(!choice.is_null() && !choice.is_structured() &&
                  ((choice.is_number() && original.is_number()) ||
                   choice.type() == original.type()),
              "Search choices must be scalars matching the configuration field "
              "type");
      require(distinct.insert(choice.dump()).second,
              "Duplicate choices in search axis");
    }
    require(choices.size() <= kMaximumSpace / result.cardinality,
            "Search Cartesian space exceeds one million combinations");
    result.cardinality *= choices.size();
  }
  const auto &objective = value.at("objective");
  fields(objective, {"definition", "context", "step_kind", "selection"});
  result.metric = definition(objective.at("definition"));
  result.metric_sha256 = telemetry_metric_definition_digest(result.metric).hex;
  result.context = objective.at("context").get<TelemetryAttributes>();
  result.step = step_kind(text(objective.at("step_kind")));
  require(text(objective.at("selection")) == "last",
          "Search selects the last committed matching observation");
  std::set<std::string> context_keys;
  for (const auto &[key, item] : result.context) {
    static_cast<void>(item);
    context_keys.insert(key);
  }
  require(context_keys == result.metric.context_keys,
          "Search objective context keys differ from its definition");
  // Reuse the telemetry validator for UTF-8, context value and envelope limits
  // before creating a search, rather than discovering an impossible objective
  // only after a training process has run.
  TelemetryMetricPoint objective_probe;
  objective_probe.definition = result.metric;
  objective_probe.context = result.context;
  objective_probe.step.kind = result.step;
  objective_probe.observation = make_telemetry_metric_value(0.0);
  TelemetryEnvelope probe;
  probe.identity = {.stream_id = "search.preflight",
                    .event_id = "objective",
                    .idempotency_key = "objective",
                    .sequence = 1};
  probe.payload = TelemetryMetricBatch{{std::move(objective_probe)}};
  validate_telemetry_envelope(probe);
  const auto &execution = value.at("execution");
  const bool bridged = value.contains("recipe_bridge");
  if (bridged) {
    const auto &profile = recipe_bridge_profile(value);
    const auto expected_objective =
        profile.image
            ? image_search_objective(image_objective_classes(result.metric))
            : paired_search_objective();
    require(
        result.project == "liuding-forge" &&
            result.experiment == profile.experiment &&
            execution.contains("training") &&
            result.metric_sha256 ==
                telemetry_metric_definition_digest(expected_objective).hex &&
            result.context ==
                TelemetryAttributes{{"slice", "overall"}, {"source", "all"}} &&
            result.step == TelemetryStepKind::sample,
        "Recipe search requires its native project, model and complete "
        "accuracy objective");
    const auto &arguments = execution.at("arguments");
    Json expected{"recipe", "local-run",           "--file",   "${config}",
                  "--root", "${recipe_directory}", "--format", "json"};
    const bool retry = arguments.is_array() && arguments.size() == 10;
    if (retry) {
      require(arguments.at(8) == "--pause-after",
              "Recipe search only accepts an optional --pause-after");
      const auto pause_text = text(arguments.at(9));
      require(!pause_text.empty() && pause_text.size() <= 7 &&
                  pause_text.front() != '0' &&
                  std::all_of(
                      pause_text.begin(), pause_text.end(),
                      [](unsigned char c) { return c >= '0' && c <= '9'; }) &&
                  std::stoull(pause_text) <= 1000000,
              "Recipe search pause must be a positive bounded integer");
      expected.push_back("--pause-after");
      expected.push_back(pause_text);
    }
    require(arguments == expected &&
                unsigned_number(execution.at("max_attempts"), 16) ==
                    (retry ? 2 : 1) &&
                execution.at("retry_exit_codes") ==
                    (retry ? Json{75} : Json::array()),
            "Recipe search uses its whole-value local-run template and at "
            "most one exact checkpoint continuation");
    const auto &training = execution.at("training");
    require(
        text(training.at("model_id")) == profile.model &&
            training.at("resume").is_null() && training.contains("outputs") &&
            training.contains("retry_resume") == retry &&
            (!retry || training.at("retry_resume") ==
                           Json{{"checkpoint_reference_id", "checkpoint.last"},
                                {"restore_policy", "exact_only"}}),
        "Recipe search starts fresh and only continues its owned exact "
        "checkpoint.last");
    Json outputs = Json::array();
    for (const auto &output : training.at("outputs")) {
      fields(output, {"reference_id", "kind", "role", "max_size_bytes"});
      static_cast<void>(positive(output.at("max_size_bytes"),
                                 recipe_support::kArtifactLimit));
      auto declaration = output;
      declaration.erase("max_size_bytes");
      outputs.push_back(std::move(declaration));
    }
    const Json required_outputs{{{"reference_id", "checkpoint.last"},
                                 {"kind", "checkpoint"},
                                 {"role", "last"}},
                                {{"reference_id", "model.weights"},
                                 {"kind", "artifact"},
                                 {"role", "model.weights"}},
                                {{"reference_id", "training.result"},
                                 {"kind", "artifact"},
                                 {"role", "training.result"}}};
    require(
        outputs == required_outputs && value.at("outputs") == required_outputs,
        "Recipe search requires its three typed native outputs in order");
  }
  if (execution.contains("training")) {
    const auto &training = execution.at("training");
    fields(training, {"model_id", "configuration", "data_manifest", "resume"},
           bridged ? std::initializer_list<std::string_view>{"outputs",
                                                             "retry_resume"}
                   : std::initializer_list<std::string_view>{});
    require(training.at("configuration").is_string() &&
                text(training.at("configuration")) == "${trial_configuration}",
            "Typed search owns configuration; use ${trial_configuration}");
  }
  require(
      execution.at("project_id").get<std::string>() == result.project &&
          execution.at("experiment_id").get<std::string>() == result.experiment,
      "Search and execution project/experiment must match");
  require(bridged || (unsigned_number(execution.at("max_attempts"), 16) == 1 &&
                      execution.at("retry_exit_codes").is_array() &&
                      execution.at("retry_exit_codes").empty()),
          "Search trials execute once; checkpoint continuation requires a "
          "new search with explicit frozen inputs");
  require(!execution.contains("deadline_unix_ms"),
          "Search owns its frozen absolute execution deadline");
  const auto validate_bindings = [&](const Json &bindings, bool output) {
    require(bindings.is_array() && bindings.size() <= 64 &&
                (!output || !bindings.empty()),
            "Search declares at most 64 inputs/outputs and requires an output");
    std::set<std::string> ids;
    for (const auto &binding : bindings) {
      if (output) {
        fields(binding, {"reference_id", "kind", "role"});
        const auto kind = text(binding.at("kind"));
        require(kind == "artifact" || kind == "checkpoint",
                "Unknown expected output kind");
        const auto role = text(binding.at("role"));
        require(!role.empty() && role.size() <= 128,
                "Expected outputs require a role of at most 128 bytes");
        const auto id = text(binding.at("reference_id"));
        identifier(id);
        require(ids.insert(id).second, "Duplicate expected output reference");
      } else {
        fields(binding, {"name", "run_id", "reference_id", "sha256"});
        const auto name = text(binding.at("name"));
        identifier(name);
        require(name.size() <= 115,
                "Search input names must leave room for the search.input. role "
                "prefix");
        require(ids.insert(name).second, "Duplicate declared input");
        identifier(text(binding.at("run_id")));
        identifier(text(binding.at("reference_id")));
        validate_artifact_digest({.hex = text(binding.at("sha256"))});
      }
    }
  };
  validate_bindings(value.at("inputs"), false);
  validate_bindings(value.at("outputs"), true);
  const auto count = std::min(result.trials, result.cardinality);
  std::mt19937_64 engine(result.seed);
  std::map<std::uint64_t, std::uint64_t> permutation;
  for (std::uint64_t index = 0; index < count; ++index) {
    if (strategy == "grid") {
      result.sequence.push_back(index);
    } else {
      const auto remaining = result.cardinality - index;
      const auto selected = uniform(engine, remaining);
      const auto lookup = [&](std::uint64_t key) {
        const auto found = permutation.find(key);
        return found == permutation.end() ? key : found->second;
      };
      result.sequence.push_back(lookup(selected));
      permutation[selected] = lookup(remaining - 1);
      permutation.erase(remaining - 1);
    }
  }
  return result;
}

RunArtifactReference put_json(SQLiteRunStore &store, std::string_view run,
                              std::string id, std::string role,
                              const Json &value, std::uint64_t stamp) {
  const auto content = value.dump();
  require(content.size() <= kEvidenceBytes, "Search artifact exceeds 1 MiB");
  return store.put_artifact(run, std::move(id), RunArtifactKind::artifact,
                            std::move(role), bytes(content),
                            {.logical_name = "search.json",
                             .media_type = "application/json",
                             .producer = std::string(kProducer),
                             .expected_size_bytes = content.size()},
                            stamp, digest(content));
}

Json artifact_json(SQLiteRunStore &store,
                   const RunArtifactReference &reference) {
  require(reference.kind == RunArtifactKind::artifact &&
              reference.artifact.size_bytes <= kEvidenceBytes,
          "Search JSON evidence exceeds its bounded artifact contract");
  FileArtifactStore cas(store.artifact_root());
  auto reader = cas.open(reference.artifact.digest);
  require(reader->size_bytes() == reference.artifact.size_bytes,
          "Search artifact size mismatch");
  std::string content(static_cast<std::size_t>(reader->size_bytes()), '\0');
  std::size_t offset = 0;
  while (offset != content.size()) {
    const auto count = reader->read(
        std::as_writable_bytes(std::span(content.data(), content.size()))
            .subspan(offset));
    if (!count) throw SerializationError("Truncated search evidence");
    offset += count;
  }
  require(digest(content) == reference.artifact.digest,
          "Search evidence digest mismatch");
  return parse_json(content);
}

Json load_json(SQLiteRunStore &store, std::string_view run,
               std::string_view id) {
  const auto reference = store.find_reference(run, id);
  if (!reference || reference->kind != RunArtifactKind::artifact ||
      reference->role != id || reference->artifact.size_bytes > kEvidenceBytes)
    throw CompatibilityError("Missing/incompatible search evidence reference");
  return artifact_json(store, *reference);
}

Json reference_json(const RunArtifactReference &reference) {
  return {{"reference_id", reference.reference_id},
          {"kind", reference.kind == RunArtifactKind::checkpoint ? "checkpoint"
                                                                 : "artifact"},
          {"role", reference.role},
          {"sha256", reference.artifact.digest.hex},
          {"size_bytes", reference.artifact.size_bytes}};
}

void verify_content(SQLiteRunStore &store,
                    const RunArtifactReference &reference) {
  FileArtifactStore cas(store.artifact_root());
  auto reader = cas.open(reference.artifact.digest);
  require(reader->size_bytes() == reference.artifact.size_bytes,
          "Search bound artifact size mismatch");
}

void validate_inputs(SQLiteRunStore &store, const Plan &plan) {
  for (const auto &input : plan.json.at("inputs")) {
    const auto reference = store.find_reference(text(input.at("run_id")),
                                                text(input.at("reference_id")));
    require(
        reference && reference->artifact.digest.hex == text(input.at("sha256")),
        "Declared search input differs from the committed run/CAS artifact");
    verify_content(store, *reference);
  }
  const auto &execution = plan.json.at("execution");
  if (execution.contains("training")) {
    const auto &training = execution.at("training");
    validate_local_training_source(store, plan.project,
                                   training.at("data_manifest"),
                                   RunArtifactKind::artifact);
    if (!training.at("resume").is_null())
      validate_local_training_source(store, plan.project,
                                     training.at("resume").at("checkpoint"),
                                     RunArtifactKind::checkpoint);
  }
}

RunMetadata search_metadata(SQLiteRunStore &store, std::string_view id) {
  const auto run = store.find(id);
  require(run && run->tags.contains("liuding.search.kind") &&
              run->tags.at("liuding.search.kind") == "search",
          "Select an existing search run");
  return *run;
}

void validate_identity(const RunMetadata &search, const Plan &plan) {
  require(search.project_id == plan.project &&
              search.experiment_id == plan.experiment &&
              search.lineage.config_digest == plan.sha256 &&
              search.lineage.source_digest ==
                  text(plan.json.at("execution").at("executable_sha256")) &&
              search.lineage.build_fingerprint &&
              !search.lineage.build_fingerprint->empty(),
          "Search metadata differs from its frozen plan/provenance");
}

std::string trial_id(std::string_view search, std::uint64_t index) {
  return "search-" + digest(search).hex + ".trial-" + std::to_string(index);
}

// nlohmann/json's noexcept null constructor has an exception-analysis false
// positive; preserve the aggregate's inferred exception contract.
// NOLINTNEXTLINE(bugprone-exception-escape)
struct Trial {
  std::string id;
  std::string training;
  Json parameters;
  Json configuration;
  Json training_configuration;
  std::string config_sha256;
  std::uint64_t seed = 0;
  fs::path root;
  fs::path directory;
  Json execution;
};

Trial trial(const Plan &plan, std::string_view search, const fs::path &root,
            std::uint64_t index, std::uint64_t deadline) {
  require(index >= 1 && index <= plan.sequence.size(),
          "Trial index outside frozen sequence");
  Trial result;
  result.root = root;
  result.id = trial_id(search, index);
  result.training = result.id + ".training";
  result.parameters = Json::object();
  result.configuration = plan.json.at("configuration");
  auto coordinate = plan.sequence[static_cast<std::size_t>(index - 1)];
  const auto &space = plan.json.at("space");
  for (std::size_t offset = space.size(); offset != 0; --offset) {
    const auto &axis = space.at(offset - 1);
    const auto &choices = axis.at("values");
    const auto choice = coordinate % choices.size();
    coordinate /= choices.size();
    result.parameters[text(axis.at("name"))] =
        choices.at(static_cast<std::size_t>(choice));
    result.configuration[Json::json_pointer(text(axis.at("config_pointer")))] =
        choices.at(static_cast<std::size_t>(choice));
  }
  result.config_sha256 = digest(result.configuration.dump()).hex;
  result.training_configuration = result.configuration;
  if (plan.json.contains("recipe_bridge") &&
      recipe_bridge_profile(plan.json).transfer)
    result.training_configuration = transfer_search_configuration(
        result.configuration, plan.json.at("recipe_bridge").at("transfer"));
  // SplitMix64, fully specified unsigned arithmetic; no mutable RNG state.
  result.seed = plan.seed + index * 0x9e3779b97f4a7c15ULL;
  result.seed = (result.seed ^ (result.seed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  result.seed = (result.seed ^ (result.seed >> 27U)) * 0x94d049bb133111ebULL;
  result.seed ^= result.seed >> 31U;
  result.directory = root / "search-work" / digest(search).hex /
                     ("trial-" + std::to_string(index));
  std::map<std::string, std::string> substitutions{
      {"${store}", path_text(root)},
      {"${trial_id}", result.id},
      {"${training_run_id}", result.training},
      {"${config}", path_text(result.directory / "config.json")},
      {"${config_sha256}", result.config_sha256},
      {"${trial_directory}", path_text(result.directory)},
      {"${trial_seed}", std::to_string(result.seed)}};
  if (plan.json.contains("recipe_bridge"))
    substitutions["${recipe_directory}"] =
        path_text(result.directory / "recipe");
  for (const auto &[name, value] : result.parameters.items())
    substitutions["${param." + name + "}"] =
        value.is_string() ? text(value) : value.dump();
  for (const auto &input : plan.json.at("inputs"))
    substitutions["${input." + text(input.at("name")) + "}"] =
        text(input.at("sha256"));
  const auto expand = [&](const Json &value) {
    auto original = text(value);
    const auto found = substitutions.find(original);
    if (found != substitutions.end()) return found->second;
    require(
        original.find("${") == std::string::npos,
        "Search placeholders must be known whole argument/environment values");
    return original;
  };
  result.execution = plan.json.at("execution");
  for (auto &value : result.execution.at("arguments")) value = expand(value);
  for (auto &value : result.execution.at("environment")) value = expand(value);
  result.execution["working_directory"] =
      expand(result.execution.at("working_directory"));
  result.execution["deadline_unix_ms"] = deadline;
  if (result.execution.contains("training")) {
    // The input run is sealed before native dispatch. A running search is not
    // itself a terminal typed-input source, and completion here means input
    // publication only, never completion of a training trial.
    result.execution["training"]["configuration"] = {
        {"run_id", result.id + ".inputs"},
        {"reference_id", "configuration"},
        {"sha256", digest(result.training_configuration.dump()).hex},
        {"size_bytes", result.training_configuration.dump().size()}};
  }
  return result;
}

void materialize(const Trial &value) {
  fs::create_directories(value.directory);
  ordinary(value.directory);
  const auto path = value.directory / "config.json";
  if (fs::exists(path)) {
    ordinary(path);
    require(fs::hard_link_count(path) == 1 &&
                recipe_support::file_digest(path, kPlanBytes).hex ==
                    value.config_sha256,
            "Trial configuration view differs from frozen CAS content");
    return;
  }
  const auto temporary = value.directory / "config.json.pending";
  require(!fs::is_symlink(fs::symlink_status(temporary)),
          "Invalid pending trial config view");
  if (fs::exists(temporary)) {
    ordinary(temporary);
    require(fs::hard_link_count(temporary) == 1,
            "Pending trial config view must not be hard-linked");
  }
  recipe_support::write_file(temporary, bytes(value.configuration.dump()));
  fs::rename(temporary, path);
}

Json admission(const Trial &value, std::uint64_t index, const Plan &plan) {
  return {
      {"format", "jyppx.liuding.local_search_admission"},
      {"schema_version", 1},
      {"index", index},
      {"trial_id", value.id},
      {"training_run_id", value.training},
      {"search_spec_sha256", plan.sha256},
      {"space_index", plan.sequence.at(static_cast<std::size_t>(index - 1))},
      {"parameters", value.parameters},
      {"configuration", value.configuration},
      {"config_sha256", value.config_sha256},
      {"trial_seed", value.seed},
      {"store_root", path_text(value.root)},
      {"execution", value.execution}};
}

void seal_trial_inputs(SQLiteRunStore &store, const RunMetadata &search,
                       const Plan &plan, const Trial &value,
                       std::uint64_t index, bool child_exists) {
  if (!value.execution.contains("training")) return;
  const auto id = value.id + ".inputs";
  const auto existing = store.find(id);
  require(
      !child_exists || (existing && existing->status == RunStatus::completed),
      "Admitted native execution requires its already sealed input run");
  require(!existing || existing->status == RunStatus::running ||
              existing->status == RunStatus::completed,
          "Trial input run is not in its publication lifecycle");
  RunMetadata metadata;
  metadata.run_id = id;
  metadata.project_id = plan.project;
  metadata.experiment_id = plan.experiment;
  metadata.display_name = "Committed inputs for trial " + std::to_string(index);
  metadata.status = RunStatus::running;
  metadata.created_at_unix_ms = search.created_at_unix_ms;
  metadata.updated_at_unix_ms = search.created_at_unix_ms;
  metadata.lineage.parent_run_id = search.run_id;
  metadata.lineage.source_digest = plan.sha256;
  metadata.lineage.build_fingerprint = search.lineage.build_fingerprint;
  metadata.lineage.config_digest = value.config_sha256;
  metadata.lineage.data_manifest_digest =
      text(value.execution.at("training").at("data_manifest").at("sha256"));
  metadata.tags = {{"liuding.search.kind", "inputs"},
                   {"liuding.search.role", "trial.configuration"},
                   {"liuding.search.index", std::to_string(index)},
                   {"liuding.search.spec_sha256", plan.sha256}};
  // start()/put_artifact()/transition() are independently idempotent. A crash
  // between them is repaired from the exact admission; conflicting identities
  // are rejected by the existing store rather than silently adopted.
  store.start(metadata);
  const auto reference =
      put_json(store, id, "configuration", "configuration",
               value.training_configuration, search.created_at_unix_ms);
  require(reference.artifact.digest.hex ==
              digest(value.training_configuration.dump()).hex,
          "Trial input configuration differs from its admission");
  verify_content(store, reference);
  store.transition(id, RunStatus::completed, search.created_at_unix_ms);
}

Json bound_json(SQLiteRunStore &store, const Json &binding) {
  const auto reference = store.find_reference(text(binding.at("run_id")),
                                              text(binding.at("reference_id")));
  require(reference && reference->kind == RunArtifactKind::artifact &&
              reference->artifact.digest.hex == text(binding.at("sha256")) &&
              reference->artifact.size_bytes ==
                  unsigned_number(binding.at("size_bytes"), kEvidenceBytes),
          "Paired bridge JSON differs from its frozen binding");
  return artifact_json(store, *reference);
}

struct RecipeBridgeReceipt {
  std::string run;
  std::string id;
};

RecipeBridgeReceipt recipe_bridge_receipt(SQLiteRunStore &store,
                                          const Trial &value) {
  const auto job = store.find(value.id);
  require(job && job->lineage.parent_run_id,
          "Paired bridge trial has no owning search");
  // A completed local job is immutable. Its parent search is still running
  // while each trial's confirmed receipt is first committed. The full trial
  // digest gives every trial a distinct reference within that owning search.
  return {*job->lineage.parent_run_id,
          std::string(kRecipeBridge) + "." + digest(value.id).hex};
}

void validate_recipe_trial_inputs(SQLiteRunStore &store, const Plan &plan,
                                  const Trial &value) {
  const auto &profile = recipe_bridge_profile(plan.json);
  const auto inspected =
      profile.transfer
          ? inspect_transfer_search_inputs(value.directory / "config.json")
      : profile.image
          ? inspect_image_search_inputs(value.directory / "config.json")
          : inspect_paired_search_inputs(value.directory / "config.json");
  require(text(value.configuration.at("recipe")) == profile.recipe,
          "Recipe trial configuration differs from its closed bridge identity");
  if (profile.transfer)
    require(
        inspected.at("training_configuration") == value.training_configuration,
        "Transfer descriptor or weight identity differs from frozen search "
        "input");
  if (profile.image)
    require(plan.metric_sha256 ==
                telemetry_metric_definition_digest(
                    image_search_objective(
                        inspected.at("data_manifest").at("classes").size()))
                    .hex,
            "Image objective class count differs from its complete dataset");
  require(
      inspected.at("configuration") == value.configuration &&
          digest(inspected.at("configuration").dump()).hex ==
              value.config_sha256 &&
          inspected.at("data_manifest") ==
              bound_json(store,
                         value.execution.at("training").at("data_manifest")),
      "Paired trial configuration must equal the resolved recipe and "
      "its frozen dataset before dispatch");
  const auto &arguments = value.execution.at("arguments");
  if (arguments.size() == 10)
    require(std::stoull(text(arguments.at(9))) <
                positive(inspected.at("total_steps"), 1000000),
            "Paired search pause must precede the final update");
  require(!fs::exists(value.directory / "recipe"),
          "Paired trial requires its unused recipe directory");
}

std::string canonical_hex(std::span<const std::byte> content) {
  constexpr std::string_view digits = "0123456789abcdef";
  require(content.size() <= 65536,
          "Paired bridge metric exceeds its evidence limit");
  std::string encoded;
  encoded.reserve(content.size() * 2);
  for (const auto item : content) {
    const auto value = std::to_integer<unsigned>(item);
    encoded.push_back(digits[value >> 4U]);
    encoded.push_back(digits[value & 15U]);
  }
  return encoded;
}

TelemetryEnvelope bridge_metric(const Json &manifest) {
  const auto encoded = text(manifest.at("native_metric_canonical_hex"));
  require(
      !encoded.empty() && encoded.size() <= 131072 && encoded.size() % 2 == 0,
      "Invalid paired native metric evidence size");
  const auto nibble = [](char value) -> unsigned {
    if (value >= '0' && value <= '9') return unsigned(value - '0');
    if (value >= 'a' && value <= 'f') return unsigned(value - 'a' + 10);
    throw CompatibilityError("Invalid paired native metric evidence hex");
  };
  std::vector<std::byte> content;
  content.reserve(encoded.size() / 2);
  for (std::size_t i = 0; i < encoded.size(); i += 2)
    content.push_back(
        std::byte((nibble(encoded[i]) << 4U) | nibble(encoded[i + 1])));
  require(sha256_hex(content) == text(manifest.at("native_metric_sha256")),
          "Paired native metric evidence digest differs");
  return decode_telemetry_envelope(content);
}

// Local control validates the complete owner/intent/output receipt chain.
// Retain that exact final-attempt evidence, including its effective resume;
// the search's original null resume must never stand in for a continuation.
Json recipe_attempt_evidence(SQLiteRunStore &store, const Trial &value,
                             const Json &local) {
  require(local.at("status") == "completed" &&
              !local.at("owner_active").get<bool>() &&
              local.at("job_id") == value.id &&
              local.at("spec_sha256") == digest(value.execution.dump()).hex &&
              local.at("attempts").is_array() && !local.at("attempts").empty(),
          "Paired bridge requires its completed, unowned local job");
  const auto &final = local.at("attempts").back();
  const auto index = positive(final.at("index"), 2);
  require(index == local.at("attempts").size(),
          "Paired bridge final attempt index differs");
  const auto attempt = text(final.at("attempt_id"));
  const auto outcome = load_json(store, attempt, "local.result");
  require(
      outcome == final.at("result") && outcome.at("job_id") == value.id &&
          outcome.at("attempt_id") == attempt && outcome.at("index") == index &&
          outcome.at("spec_sha256") == local.at("spec_sha256") &&
          outcome.at("status") == "exited" && outcome.at("exit_code") == 0 &&
          outcome.at("process_tree_cleanup_confirmed").get<bool>() &&
          outcome.at("training_outputs_confirmed").get<bool>(),
      "Paired bridge requires confirmed final native cleanup and outputs");
  auto training = value.execution.at("training");
  training.erase("retry_resume");
  Json inputs_sha = nullptr;
  if (value.execution.at("training").contains("retry_resume")) {
    const auto frozen = load_json(store, attempt, "local.training.inputs");
    const auto job_frozen = load_json(
        store, value.id, "local.training.inputs." + std::to_string(index));
    require(frozen == job_frozen && frozen.at("job_id") == value.id &&
                frozen.at("attempt_id") == attempt &&
                frozen.at("index") == index &&
                frozen.at("spec_sha256") == local.at("spec_sha256") &&
                outcome.at("training_sha256") == digest(frozen.dump()).hex,
            "Paired bridge effective training manifest differs");
    for (const auto *name :
         {"model_id", "configuration", "data_manifest", "outputs"})
      require(frozen.at("training").at(name) == training.at(name),
              "Paired bridge changed an immutable typed input");
    if (index == 1) {
      require(frozen.at("training").at("resume").is_null() &&
                  frozen.at("previous_result_sha256").is_null(),
              "Paired first attempt must start from its frozen null resume");
    } else {
      const auto &previous = local.at("attempts").at(index - 2);
      const auto previous_id = text(previous.at("attempt_id"));
      const auto checkpoint =
          store.find_reference(previous_id, "checkpoint.last");
      const auto previous_result =
          store.find_reference(previous_id, "local.result");
      require(checkpoint && previous_result &&
                  checkpoint->kind == RunArtifactKind::checkpoint &&
                  previous.at("result").at("status") == "exited" &&
                  previous.at("result").at("exit_code") == 75 &&
                  previous.at("result")
                      .at("process_tree_cleanup_confirmed")
                      .get<bool>() &&
                  frozen.at("previous_result_sha256") ==
                      previous_result->artifact.digest.hex &&
                  frozen.at("training").at("resume") ==
                      Json{{"restore_policy", "exact_only"},
                           {"checkpoint",
                            {{"run_id", previous_id},
                             {"reference_id", "checkpoint.last"},
                             {"sha256", checkpoint->artifact.digest.hex},
                             {"size_bytes", checkpoint->artifact.size_bytes}}}},
              "Paired continuation must bind its exact prior checkpoint and "
              "confirmed retryable result");
    }
    training = frozen.at("training");
    inputs_sha = digest(frozen.dump()).hex;
  }
  const auto receipt = load_json(store, attempt, "local.training.outputs");
  auto receipt_training = training;
  receipt_training.erase("outputs");
  require(
      receipt.at("confirmed").get<bool>() &&
          receipt.at("failure_reason").is_null() &&
          receipt.at("job_id") == value.id &&
          receipt.at("attempt_id") == attempt && receipt.at("index") == index &&
          receipt.at("spec_sha256") == local.at("spec_sha256") &&
          receipt.at("training") == receipt_training &&
          outcome.at("training_outputs_sha256") == digest(receipt.dump()).hex &&
          receipt.at("outputs").size() == 3,
      "Paired bridge output receipt differs from its effective inputs");
  Json outputs = Json::array();
  for (const auto &output : receipt.at("outputs")) {
    const auto reference =
        store.find_reference(attempt, text(output.at("reference_id")));
    require(reference && output.at("confirmed").get<bool>() &&
                output.at("failure_reason").is_null() &&
                output.at("sha256") == reference->artifact.digest.hex &&
                output.at("size_bytes") == reference->artifact.size_bytes,
            "Paired bridge output differs from its confirmed receipt");
    verify_content(store, *reference);
    outputs.push_back(reference_json(*reference));
  }
  return {{"trial_id", value.id},
          {"training_run_id", value.training},
          {"spec_sha256", local.at("spec_sha256")},
          {"attempt_id", attempt},
          {"attempt_index", index},
          {"local_result_sha256", digest(outcome.dump()).hex},
          {"training_outputs_sha256", digest(receipt.dump()).hex},
          {"effective_inputs_sha256", inputs_sha},
          {"training", training},
          {"outputs", outputs}};
}

const TelemetryMetricPoint &validate_native_metric(
    const TelemetryEnvelope &envelope, const Plan &plan, const Trial &value,
    const Json &result) {
  const auto &profile = recipe_bridge_profile(plan.json);
  const auto *batch = std::get_if<TelemetryMetricBatch>(&envelope.payload);
  require(batch && batch->points.size() == (profile.image ? 1U : 2U) &&
              envelope.identity.stream_id == profile.native_stream &&
              envelope.identity.event_id == "evaluation.final" &&
              envelope.identity.idempotency_key == "evaluation.final" &&
              envelope.identity.sequence == 1 &&
              envelope.correlation.project_id == plan.project &&
              envelope.correlation.run_id == profile.native_run &&
              envelope.attributes ==
                  TelemetryAttributes{
                      {"recipe", std::string(profile.recipe)},
                      {"config_sha256", value.config_sha256},
                      {"dataset_sha256", text(value.execution.at("training")
                                                  .at("data_manifest")
                                                  .at("sha256"))},
                      {"model_sha256", text(result.at("model_sha256"))}},
          "Paired native committed evaluation identity differs");
  const auto &point = batch->points.front();
  require(telemetry_metric_definition_digest(point.definition).hex ==
                  plan.metric_sha256 &&
              point.context == plan.context && point.step.kind == plan.step &&
              point.step.value == result.at("validation_samples") &&
              point.sample_count == point.step.value &&
              point.observation.status == TelemetryValueStatus::finite &&
              point.observation.value &&
              std::isfinite(*point.observation.value) &&
              *point.observation.value == result.at("accuracy"),
          "Recipe native objective is missing, nonfinite or incompatible");
  if (!profile.image)
    require(batch->points.back().definition.metric_id == "regression.mse" &&
                batch->points.back().observation.status ==
                    TelemetryValueStatus::finite &&
                batch->points.back().observation.value &&
                *batch->points.back().observation.value == result.at("mse"),
            "Paired native objective is missing, nonfinite or incompatible");
  return point;
}

Json validate_recipe_bridge(SQLiteRunStore &store, const Plan &plan,
                            const Trial &value, const Json &local) {
  const auto &profile = recipe_bridge_profile(plan.json);
  const auto receipt = recipe_bridge_receipt(store, value);
  const auto manifest = load_json(store, receipt.run, receipt.id);
  fields(manifest, {"format",
                    "schema_version",
                    "search_spec_sha256",
                    "trial_id",
                    "training_run_id",
                    "spec_sha256",
                    "attempt_id",
                    "attempt_index",
                    "local_result_sha256",
                    "training_outputs_sha256",
                    "effective_inputs_sha256",
                    "training",
                    "outputs",
                    "seed_policy",
                    "training_seed",
                    "trial_seed",
                    "native_source_digest",
                    "native_build_fingerprint",
                    "native_metric_canonical_hex",
                    "native_metric_sha256",
                    "native_metric_run_sequence"});
  require(text(manifest.at("format")) == profile.manifest_format &&
              manifest.at("schema_version") == 1 &&
              manifest.at("search_spec_sha256") == plan.sha256 &&
              manifest.at("seed_policy") == "configuration" &&
              manifest.at("training_seed") ==
                  unsigned_number(value.configuration.at("seed"), 2147483647) &&
              manifest.at("trial_seed") == value.seed,
          "Paired bridge schema or frozen search identity differs");
  const auto evidence = recipe_attempt_evidence(store, value, local);
  for (const auto &[key, item] : evidence.items())
    require(manifest.at(key) == item,
            "Paired bridge differs from confirmed final-attempt evidence");
  const auto result =
      load_json(store, text(evidence.at("attempt_id")), "training.result");
  const auto &lineage = result.at("lineage");
  const auto data =
      bound_json(store, evidence.at("training").at("data_manifest"));
  const auto steps =
      ((positive(data.at("train").at("samples"),
                 profile.image ? 1000000 : 4096) +
        positive(value.configuration.at("batch_size"), 1024) - 1) /
       positive(value.configuration.at("batch_size"), 1024)) *
      positive(value.configuration.at("epochs"), 1000);
  const bool recipe_lineage =
      profile.image
          ? result.contains("transfer") == profile.transfer &&
                data.at("classes").is_array() &&
                data.at("classes").size() >= 2 &&
                data.at("classes").size() <= 1024 &&
                plan.metric_sha256 ==
                    telemetry_metric_definition_digest(
                        image_search_objective(data.at("classes").size()))
                        .hex
          : text(lineage.at("recipe")) == kPairedRecipe &&
                lineage.at("numerics") == result.at("numerics");
  if (profile.transfer) {
    const auto &frozen = plan.json.at("recipe_bridge").at("transfer");
    const auto configuration =
        bound_json(store, evidence.at("training").at("configuration"));
    const auto &transfer = result.at("transfer");
    require(configuration == value.training_configuration &&
                value.configuration.at("device") == "cpu" &&
                value.configuration.at("model") == "resnet50" &&
                value.configuration.at("backbone_mode") == "frozen" &&
                value.configuration.at("batch_norm") == "eval" &&
                transfer.at("backbone_mode") == "frozen" &&
                transfer.at("batch_norm") == "eval" &&
                transfer.at("weight_input_sha256") ==
                    frozen.at("weight_input_sha256") &&
                transfer.at("initial").at("backbone_sha256") ==
                    transfer.at("final").at("backbone_sha256") &&
                !result.contains("calibration"),
            "Transfer result must preserve its frozen descriptor, backbone and "
            "BN policy");
  } else if (profile.image) {
    const bool calibrated =
        value.configuration.value("batch_norm_calibration", "off") ==
        "train_cumulative_once";
    require(
        result.contains("calibration") == calibrated,
        "Image result must preserve its declared BatchNorm calibration policy");
    if (calibrated) {
      const auto &calibration = result.at("calibration");
      const auto samples = positive(data.at("train").at("samples"), 1000000);
      const auto batch = positive(value.configuration.at("batch_size"), 1024);
      require(value.configuration.at("device") == "cpu" &&
                  calibration.at("phase") == "completed" &&
                  calibration.at("optimizer_updates") == steps &&
                  calibration.at("samples") == samples &&
                  calibration.at("batches") == (samples + batch - 1) / batch &&
                  calibration.at("calibrated_model_sha256") ==
                      result.at("model_sha256"),
              "Image calibration must complete one training-data pass without "
              "extra optimizer updates");
    }
  }
  const auto schedule = value.configuration.value("lr_schedule", "epoch_step");
  const Json scheduler_steps = profile.image && schedule != "epoch_step"
                                   ? Json(steps)
                                   : value.configuration.at("epochs");
  const auto job = store.find(value.id);
  require(
      job && result.at("schema_version") == 1 && recipe_lineage &&
          text(result.at("recipe")) == profile.recipe &&
          text(result.at("run_id")) == profile.native_run &&
          result.at("status") == "completed" &&
          result.at("quality_passed").get<bool>() &&
          result.at("global_step") == steps &&
          result.at("optimizer_steps") == steps &&
          result.at("completed_epochs") == value.configuration.at("epochs") &&
          result.at("scheduler_steps") == scheduler_steps &&
          result.at("validation_samples") ==
              data.at("validation").at("samples") &&
          result.at("model_sha256") ==
              evidence.at("outputs").at(1).at("sha256") &&
          lineage.at("config_sha256") == value.config_sha256 &&
          lineage.at("dataset_sha256") ==
              evidence.at("training").at("data_manifest").at("sha256") &&
          parse_json(text(manifest.at("native_build_fingerprint"))) ==
              lineage.at("build") &&
          digest(text(manifest.at("native_build_fingerprint"))).hex ==
              job->lineage.build_fingerprint &&
          !text(manifest.at("native_source_digest")).empty() &&
          manifest.at("native_source_digest") ==
              lineage.at("build").at("git_commit") &&
          positive(manifest.at("native_metric_run_sequence"), 1000000) != 0,
      "Paired bridge result/configuration/data/model/build lineage differs");
  const auto envelope = bridge_metric(manifest);
  static_cast<void>(validate_native_metric(envelope, plan, value, result));
  return manifest;
}

// A partial publication can contain no references or the complete atomic
// attachment, and no event or exactly the one native-derived event. Other
// writers cannot turn a running partial bridge into a different objective.
struct RecipePublication {
  RunMetadata metadata;
  std::vector<RunArtifactReference> references;
  TelemetryEnvelope event;
};

RecipePublication recipe_publication(SQLiteRunStore &store, const Plan &plan,
                                     const Trial &value, const Json &manifest) {
  const auto &profile = recipe_bridge_profile(plan.json);
  const auto job = store.find(value.id);
  require(job && job->lineage.parent_run_id,
          "Paired trial has no owning search");
  const auto search = search_metadata(store, *job->lineage.parent_run_id);
  validate_identity(search, plan);
  const auto manifest_sha = digest(manifest.dump()).hex;
  RunMetadata metadata;
  metadata.run_id = value.training;
  metadata.project_id = plan.project;
  metadata.experiment_id = plan.experiment;
  metadata.display_name = profile.display_name;
  metadata.status = RunStatus::running;
  metadata.created_at_unix_ms = search.created_at_unix_ms;
  metadata.updated_at_unix_ms = search.created_at_unix_ms;
  metadata.lineage.parent_run_id = value.id;
  metadata.lineage.source_digest = text(manifest.at("native_source_digest"));
  metadata.lineage.build_fingerprint =
      text(manifest.at("native_build_fingerprint"));
  metadata.lineage.config_digest = value.config_sha256;
  metadata.lineage.data_manifest_digest =
      text(manifest.at("training").at("data_manifest").at("sha256"));
  metadata.tags = {
      {"liuding.training.model_id", std::string(profile.model)},
      {"liuding.search.recipe", std::string(profile.recipe)},
      {"liuding.search.bridge_sha256", manifest_sha},
      {"liuding.search.attempt_id", text(manifest.at("attempt_id"))}};
  std::vector<RunArtifactReference> references;
  const auto attach = [&](std::string_view run, std::string_view id,
                          std::string_view renamed) {
    auto reference = store.find_reference(run, id);
    require(reference.has_value(),
            "Paired bridge publication source is missing");
    verify_content(store, *reference);
    reference->reference_id = renamed;
    reference->role = renamed;
    reference->attached_at_unix_ms = search.created_at_unix_ms;
    references.push_back(std::move(*reference));
  };
  const auto &training = manifest.at("training");
  for (const auto *name : {"configuration", "data_manifest"}) {
    const auto &binding = training.at(name);
    attach(text(binding.at("run_id")), text(binding.at("reference_id")),
           "search.training." + std::string(name));
  }
  if (!training.at("resume").is_null()) {
    const auto &binding = training.at("resume").at("checkpoint");
    attach(text(binding.at("run_id")), text(binding.at("reference_id")),
           "search.training.resume");
  }
  for (const auto &input : plan.json.at("inputs"))
    attach(text(input.at("run_id")), text(input.at("reference_id")),
           "search.input." + text(input.at("name")));
  for (const auto &output : manifest.at("outputs")) {
    auto reference = store.find_reference(text(manifest.at("attempt_id")),
                                          text(output.at("reference_id")));
    require(reference.has_value(), "Paired native output disappeared");
    reference->attached_at_unix_ms = search.created_at_unix_ms;
    references.push_back(std::move(*reference));
  }
  const auto receipt = recipe_bridge_receipt(store, value);
  attach(receipt.run, receipt.id, kRecipeBridge);
  const auto native = bridge_metric(manifest);
  const auto result =
      load_json(store, text(manifest.at("attempt_id")), "training.result");
  const auto &point = validate_native_metric(native, plan, value, result);
  const TelemetryEnvelope event{
      .identity = {.stream_id = std::string(profile.outer_stream) +
                                digest(value.training).hex,
                   .event_id = "evaluation.final",
                   .idempotency_key = "evaluation.final",
                   .sequence = 1},
      .time = native.time,
      .correlation = {.project_id = plan.project,
                      .run_id = value.training,
                      .attempt_id = text(manifest.at("attempt_id"))},
      .attributes = {{"recipe", std::string(profile.recipe)},
                     {"bridge_sha256", manifest_sha},
                     {"native_metric_sha256",
                      text(manifest.at("native_metric_sha256"))}},
      .payload = TelemetryMetricBatch{{point}}};
  return {std::move(metadata), std::move(references), event};
}

void validate_recipe_publication(SQLiteRunStore &store, const Trial &value,
                                 const RecipePublication &expected,
                                 bool complete) {
  const auto metadata = store.find(value.training);
  const auto &wanted = expected.metadata;
  require(metadata &&
              (complete ? metadata->status == RunStatus::completed
                        : metadata->status == RunStatus::running ||
                              metadata->status == RunStatus::completed) &&
              metadata->project_id == wanted.project_id &&
              metadata->experiment_id == wanted.experiment_id &&
              metadata->display_name == wanted.display_name &&
              metadata->created_at_unix_ms == wanted.created_at_unix_ms &&
              metadata->updated_at_unix_ms == wanted.updated_at_unix_ms &&
              metadata->lineage.parent_run_id == wanted.lineage.parent_run_id &&
              metadata->lineage.source_digest == wanted.lineage.source_digest &&
              metadata->lineage.build_fingerprint ==
                  wanted.lineage.build_fingerprint &&
              metadata->lineage.config_digest == wanted.lineage.config_digest &&
              metadata->lineage.data_manifest_digest ==
                  wanted.lineage.data_manifest_digest &&
              metadata->tags == wanted.tags,
          "Paired outer training metadata differs from its frozen bridge");
  const auto actual = store.references(value.training);
  const bool attached = !actual.empty();
  require(
      (!complete && !attached) || actual.size() == expected.references.size(),
      "Paired outer references must be empty or the complete atomic batch");
  if (attached) {
    for (const auto &reference : expected.references) {
      const auto found =
          std::find_if(actual.begin(), actual.end(), [&](const auto &item) {
            return item.reference_id == reference.reference_id;
          });
      require(
          found != actual.end() &&
              reference_json(*found) == reference_json(reference) &&
              found->artifact.logical_name == reference.artifact.logical_name &&
              found->artifact.media_type == reference.artifact.media_type &&
              found->artifact.producer == reference.artifact.producer &&
              found->attached_at_unix_ms == reference.attached_at_unix_ms,
          "Paired outer reference differs from its frozen bridge");
      verify_content(store, *found);
    }
  }
  const auto events = store.events(value.training, {.limit = 2});
  require(
      !events.next_after && events.events.size() <= 1 &&
          (!complete || events.events.size() == 1) &&
          (events.events.empty() ||
           (attached && events.events.front().run_sequence == 1 &&
            telemetry_content_digest(events.events.front().envelope) ==
                telemetry_content_digest(expected.event))),
      "Paired outer evaluation must be its unique exact native-derived event");
}

void publish_recipe_training(SQLiteRunStore &store, const RunMetadata &search,
                             const Plan &plan, const Trial &value,
                             const Json &local) {
  if (local.at("status") != "completed") return;
  const auto &profile = recipe_bridge_profile(plan.json);
  const auto bridge_receipt = recipe_bridge_receipt(store, value);
  require(bridge_receipt.run == search.run_id,
          "Paired bridge receipt must belong to the owning search");
  if (!store.find_reference(bridge_receipt.run, bridge_receipt.id)) {
    require(
        !store.find(value.training),
        "Paired bridge cannot adopt a training run without its prior receipt");
    auto manifest = recipe_attempt_evidence(store, value, local);
    const auto attempt = text(manifest.at("attempt_id"));
    const auto root = value.directory / "recipe" / "run-store";
    require(fs::is_regular_file(root / "runs.sqlite3"),
            "Completed paired recipe requires its native evidence store");
    ordinary(root / "runs.sqlite3");
    SQLiteRunStore native(root);
    const auto metadata = native.find(profile.native_run);
    const auto result = load_json(store, attempt, "training.result");
    require(metadata && metadata->status == RunStatus::completed &&
                metadata->project_id == plan.project &&
                metadata->experiment_id == profile.native_experiment &&
                metadata->lineage.config_digest == value.config_sha256 &&
                metadata->lineage.data_manifest_digest ==
                    text(value.execution.at("training")
                             .at("data_manifest")
                             .at("sha256")) &&
                metadata->lineage.source_digest &&
                metadata->lineage.build_fingerprint &&
                load_json(native, profile.native_run, "config") ==
                    value.configuration &&
                load_json(native, profile.native_run, "dataset") ==
                    bound_json(
                        store,
                        value.execution.at("training").at("data_manifest")) &&
                load_json(native, profile.native_run, "lineage") ==
                    result.at("lineage"),
            "Paired bridge requires the actual completed native run lineage");
    if (profile.transfer) {
      const auto reference =
          native.find_reference(profile.native_run, "weight-input");
      require(reference && reference->kind == RunArtifactKind::artifact &&
                  reference->role == "source_weight_contract",
              "Transfer native weight descriptor role differs");
      require(
          artifact_json(native, *reference) ==
              plan.json.at("recipe_bridge").at("transfer").at("descriptor"),
          "Transfer native weight descriptor differs from frozen search input");
    }
    const std::array<std::string_view, 3> source_ids{"final-checkpoint",
                                                     "model", "result"};
    for (std::size_t i = 0; i < source_ids.size(); ++i) {
      const auto reference =
          native.find_reference(profile.native_run, source_ids[i]);
      require(
          reference &&
              reference->artifact.digest.hex ==
                  text(manifest.at("outputs").at(i).at("sha256")) &&
              reference->artifact.size_bytes ==
                  manifest.at("outputs").at(i).at("size_bytes") &&
              reference->kind == (i == 0 ? RunArtifactKind::checkpoint
                                         : RunArtifactKind::artifact),
          "Paired native run differs from its three confirmed typed outputs");
      verify_content(native, *reference);
    }
    const auto page = native.events(profile.native_run, {.limit = 4});
    require(page.events.size() == 1 && !page.next_after,
            "Paired bridge requires its unique committed native evaluation");
    const auto &event = page.events.front();
    static_cast<void>(
        validate_native_metric(event.envelope, plan, value, result));
    const auto encoded = encode_telemetry_envelope(event.envelope);
    manifest["format"] = std::string(profile.manifest_format);
    manifest["schema_version"] = 1;
    manifest["search_spec_sha256"] = plan.sha256;
    manifest["seed_policy"] = "configuration";
    manifest["training_seed"] = value.configuration.at("seed");
    manifest["trial_seed"] = value.seed;
    manifest["native_source_digest"] = *metadata->lineage.source_digest;
    manifest["native_build_fingerprint"] = *metadata->lineage.build_fingerprint;
    manifest["native_metric_canonical_hex"] = canonical_hex(encoded);
    manifest["native_metric_sha256"] = sha256_hex(encoded);
    manifest["native_metric_run_sequence"] = event.run_sequence;
    put_json(store, bridge_receipt.run, bridge_receipt.id, bridge_receipt.id,
             manifest, search.created_at_unix_ms);
  }
  const auto manifest = validate_recipe_bridge(store, plan, value, local);
  const auto publication = recipe_publication(store, plan, value, manifest);
  store.start(publication.metadata);
  validate_recipe_publication(store, value, publication, false);
  store.attach(value.training, publication.references);
  store.append(value.training, profile.outer_batch,
               std::span(&publication.event, 1));
  validate_recipe_publication(store, value, publication, false);
  store.transition(value.training, RunStatus::completed,
                   search.created_at_unix_ms);
  // Validate again after closing the run: another public writer could have
  // raced the final pre-completion check. Such evidence is preserved, never
  // admitted to selection. New appends are rejected by the completed run.
  validate_recipe_publication(store, value, publication, true);
}

Json evaluate(SQLiteRunStore &store, const Plan &plan, const Trial &value,
              std::uint64_t index, const Json &local) {
  Json result{{"format", "jyppx.liuding.local_search_trial_result"},
              {"schema_version", 1},
              {"index", index},
              {"trial_id", value.id},
              {"training_run_id", value.training},
              {"search_spec_sha256", plan.sha256},
              {"config_sha256", value.config_sha256},
              {"parameters", value.parameters},
              {"trial_seed", value.seed},
              {"eligible", false},
              {"reason", "execution_failed"},
              {"training_lineage", nullptr},
              {"metric", nullptr},
              {"outputs", Json::array()}};
  if (value.execution.contains("training"))
    result["training_inputs"] = value.execution.at("training");
  Json effective = nullptr;
  if (plan.json.contains("recipe_bridge")) {
    result["recipe_bridge_sha256"] = nullptr;
    result["effective_training_inputs"] = nullptr;
  }
  if (local.at("status").get<std::string>() != "completed") return result;
  if (plan.json.contains("recipe_bridge")) {
    const auto manifest = validate_recipe_bridge(store, plan, value, local);
    const auto publication = recipe_publication(store, plan, value, manifest);
    validate_recipe_publication(store, value, publication, true);
    effective = manifest.at("training");
    result["recipe_bridge_sha256"] = digest(manifest.dump()).hex;
    result["effective_training_inputs"] = effective;
  }
  const auto training = store.find(value.training);
  if (!training || training->status != RunStatus::completed) {
    result["reason"] = "training_not_completed";
    return result;
  }
  if (training->project_id != plan.project ||
      training->experiment_id != plan.experiment ||
      training->lineage.parent_run_id != value.id ||
      training->lineage.config_digest != value.config_sha256 ||
      !training->lineage.source_digest ||
      training->lineage.source_digest->empty() ||
      !training->lineage.build_fingerprint ||
      training->lineage.build_fingerprint->empty()) {
    result["reason"] = "training_lineage_mismatch";
    return result;
  }
  result["training_lineage"] = {
      {"parent_run_id", *training->lineage.parent_run_id},
      {"config_digest", *training->lineage.config_digest},
      {"source_digest", *training->lineage.source_digest},
      {"build_fingerprint", *training->lineage.build_fingerprint},
      {"data_manifest_digest",
       training->lineage.data_manifest_digest
           ? Json(*training->lineage.data_manifest_digest)
           : Json(nullptr)}};
  if (value.execution.contains("training")) {
    const auto &inputs =
        effective.is_null() ? value.execution.at("training") : effective;
    if (training->lineage.data_manifest_digest !=
            text(inputs.at("data_manifest").at("sha256")) ||
        !training->tags.contains("liuding.training.model_id") ||
        training->tags.at("liuding.training.model_id") !=
            text(inputs.at("model_id"))) {
      result["reason"] = "training_input_mismatch";
      return result;
    }
    const auto matches = [&](const Json &binding, std::string_view role,
                             RunArtifactKind kind) {
      const auto reference = store.find_reference(value.training, role);
      if (!reference || reference->kind != kind || reference->role != role ||
          reference->artifact.digest.hex != text(binding.at("sha256")) ||
          reference->artifact.size_bytes !=
              unsigned_number(binding.at("size_bytes"), 1ULL << 30U))
        return false;
      verify_content(store, *reference);
      return true;
    };
    if (!matches(inputs.at("configuration"), "search.training.configuration",
                 RunArtifactKind::artifact) ||
        !matches(inputs.at("data_manifest"), "search.training.data_manifest",
                 RunArtifactKind::artifact) ||
        (!inputs.at("resume").is_null() &&
         !matches(inputs.at("resume").at("checkpoint"),
                  "search.training.resume", RunArtifactKind::checkpoint))) {
      result["reason"] = "training_input_mismatch";
      return result;
    }
  }
  for (const auto &input : plan.json.at("inputs")) {
    const auto name = "search.input." + text(input.at("name"));
    const auto reference = store.find_reference(value.training, name);
    if (!reference || reference->role != name ||
        reference->artifact.digest.hex != text(input.at("sha256"))) {
      result["reason"] = "training_input_mismatch";
      return result;
    }
    verify_content(store, *reference);
  }
  for (const auto &output : plan.json.at("outputs")) {
    const auto reference =
        store.find_reference(value.training, text(output.at("reference_id")));
    const auto kind = text(output.at("kind")) == "checkpoint"
                          ? RunArtifactKind::checkpoint
                          : RunArtifactKind::artifact;
    if (!reference || reference->kind != kind ||
        reference->role != text(output.at("role"))) {
      result["reason"] = "training_output_mismatch";
      return result;
    }
    verify_content(store, *reference);
    result["outputs"].push_back(reference_json(*reference));
  }
  RunMetricPageRequest request;
  request.metric_id = plan.metric.metric_id;
  request.definition_version = plan.metric.version;
  request.step_kind = plan.step;
  request.limit = 256;
  std::optional<StoredRunMetricPoint> selected;
  std::uint64_t inspected = 0;
  while (true) {
    const auto page = store.metrics(value.training, request);
    request.snapshot_through = page.snapshot_through;
    for (const auto &point : page.points) {
      if (++inspected > 65536)
        throw ResourceExhausted(
            "Search metric scan exceeds 65536 points per trial");
      if (point.point.context == plan.context &&
          telemetry_metric_definition_digest(point.point.definition).hex ==
              plan.metric_sha256)
        selected = point;
    }
    if (!page.next_after) break;
    require(page.next_after->run_sequence > request.after.run_sequence ||
                (page.next_after->run_sequence == request.after.run_sequence &&
                 page.next_after->point_index > request.after.point_index),
            "Search metric cursor did not advance");
    request.after = *page.next_after;
  }
  if (!selected) {
    result["reason"] = "objective_missing_or_incompatible";
    return result;
  }
  const auto &point = selected->point;
  result["metric"] = {
      {"definition_sha256", plan.metric_sha256},
      {"definition", plan.json.at("objective").at("definition")},
      {"context", point.context},
      {"step_kind", plan.json.at("objective").at("step_kind")},
      {"step", point.step.value},
      {"status", static_cast<unsigned>(point.observation.status)},
      {"value", point.observation.value ? Json(*point.observation.value)
                                        : Json(nullptr)},
      {"reason", point.observation.reason},
      {"sample_count",
       point.sample_count ? Json(*point.sample_count) : Json(nullptr)},
      {"snapshot_through", *request.snapshot_through},
      {"run_sequence", selected->cursor.run_sequence},
      {"point_index", selected->cursor.point_index},
      {"event_id", selected->event_identity.event_id}};
  if (point.observation.status != TelemetryValueStatus::finite ||
      !point.observation.value || !std::isfinite(*point.observation.value)) {
    result["reason"] = "objective_nonfinite_or_missing";
    return result;
  }
  result["eligible"] = true;
  result["reason"] = "";
  return result;
}

std::vector<Json> results(SQLiteRunStore &store, const RunMetadata &search,
                          const Plan &plan) {
  std::vector<Json> result;
  std::set<std::string> references;
  for (const auto &reference : store.references(search.run_id))
    references.insert(reference.reference_id);
  bool gap = false;
  for (std::uint64_t index = 1; index <= plan.sequence.size(); ++index) {
    const auto reference = "search.result." + std::to_string(index);
    if (!references.contains(reference)) {
      gap = true;
      continue;
    }
    require(!gap,
            "Confirmed search results must form one contiguous trial prefix");
    const auto frozen = load_json(store, search.run_id,
                                  "search.admission." + std::to_string(index));
    const auto original_root = fs::u8path(text(frozen.at("store_root")));
    require(original_root.is_absolute(),
            "Frozen trial storage root must be absolute");
    const auto expected = trial(plan, search.run_id, original_root, index,
                                search.created_at_unix_ms + plan.duration);
    require(frozen == admission(expected, index, plan),
            "Confirmed trial admission differs from the frozen plan");
    require(load_json(store, search.run_id,
                      "search.config." + std::to_string(index)) ==
                expected.configuration,
            "Confirmed trial config differs from its admission");
    auto row = load_json(store, search.run_id, reference);
    fields(row,
           {"format", "schema_version", "index", "trial_id", "training_run_id",
            "search_spec_sha256", "config_sha256", "parameters", "trial_seed",
            "eligible", "reason", "training_lineage", "metric", "outputs"},
           {"training_inputs", "recipe_bridge_sha256",
            "effective_training_inputs"});
    require(row.contains("training_inputs") ==
                expected.execution.contains("training"),
            "Confirmed typed input presence differs from its admission");
    if (row.contains("training_inputs"))
      require(row.at("training_inputs") == expected.execution.at("training"),
              "Confirmed typed training inputs differ from their admission");
    const bool paired = plan.json.contains("recipe_bridge");
    require(row.contains("recipe_bridge_sha256") == paired &&
                row.contains("effective_training_inputs") == paired,
            "Confirmed recipe bridge presence differs from the search plan");
    if (paired) {
      require(row.at("recipe_bridge_sha256").is_null() ==
                  row.at("effective_training_inputs").is_null(),
              "Confirmed paired trial requires complete bridge evidence");
      if (!row.at("recipe_bridge_sha256").is_null()) {
        const auto local = control_local_job("show", store.root(), expected.id);
        require(local.data.at("status") == "completed" &&
                    !local.data.at("owner_active").get<bool>(),
                "Confirmed paired bridge requires resolved local ownership");
        const auto manifest =
            validate_recipe_bridge(store, plan, expected, local.data);
        require(
            row.at("recipe_bridge_sha256") == digest(manifest.dump()).hex &&
                row.at("effective_training_inputs") == manifest.at("training"),
            "Confirmed paired result differs from its frozen bridge");
      }
      require(!row.at("eligible").get<bool>() ||
                  !row.at("recipe_bridge_sha256").is_null(),
              "Eligible paired trial requires its confirmed native bridge");
      const auto local = control_local_job("show", store.root(), expected.id);
      require(evaluate(store, plan, expected, index, local.data) == row,
              "Confirmed paired result differs from its actual outer training "
              "evidence");
    }
    require(
        text(row.at("format")) == "jyppx.liuding.local_search_trial_result" &&
            positive(row.at("schema_version"), 1) == 1 &&
            row.at("trial_id").get<std::string>() == expected.id &&
            row.at("training_run_id").get<std::string>() == expected.training &&
            row.at("search_spec_sha256").get<std::string>() == plan.sha256 &&
            row.at("config_sha256").get<std::string>() ==
                expected.config_sha256 &&
            row.at("parameters") == expected.parameters &&
            unsigned_number(row.at("trial_seed"),
                            std::numeric_limits<std::uint64_t>::max()) ==
                expected.seed &&
            positive(row.at("index"), kMaximumTrials) == index,
        "Search trial result identity mismatch");
    const auto eligible = row.at("eligible").get<bool>();
    require(eligible == text(row.at("reason")).empty(),
            "Ineligible trial requires an explicit reason");
    require(row.at("outputs").is_array() &&
                row.at("outputs").size() <= plan.json.at("outputs").size(),
            "Invalid confirmed trial outputs");
    if (!row.at("metric").is_null()) {
      const auto &metric = row.at("metric");
      require(metric.at("definition") ==
                      plan.json.at("objective").at("definition") &&
                  metric.at("definition_sha256").get<std::string>() ==
                      plan.metric_sha256 &&
                  metric.at("context") == Json(plan.context) &&
                  metric.at("step_kind") ==
                      plan.json.at("objective").at("step_kind"),
              "Confirmed objective semantics differ from the frozen search");
    }
    if (eligible)
      require(!row.at("training_lineage").is_null() &&
                  !row.at("metric").is_null() &&
                  unsigned_number(row.at("metric").at("status"), 5) == 1 &&
                  row.at("outputs").size() == plan.json.at("outputs").size(),
              "Eligible trial requires complete training, finite metric and "
              "output evidence");
    if (eligible && !paired) {
      // A confirmed row is a cache of committed evidence. Recheck every
      // selectable ordinary trial just as the recipe bridge does, without
      // requiring the old executable/workspace or starting another attempt.
      const auto local = control_local_job("show", store.root(), expected.id);
      require(local.data.at("status") == "completed" &&
                  !local.data.at("owner_active").get<bool>(),
              "Eligible search result requires resolved local ownership");
      require(evaluate(store, plan, expected, index, local.data) == row,
              "Confirmed search result differs from its actual committed "
              "training, metric or output evidence");
    }
    result.push_back(std::move(row));
  }
  return result;
}

Json best(const std::vector<Json> &rows, const Plan &plan) {
  Json selected = nullptr;
  for (const auto &row : rows) {
    if (!row.at("eligible").get<bool>()) continue;
    const auto number = row.at("metric").at("value").get<double>();
    require(std::isfinite(number) &&
                row.at("metric").at("definition_sha256").get<std::string>() ==
                    plan.metric_sha256,
            "Invalid persisted objective semantics/value");
    if (selected.is_null() ||
        (plan.metric.direction == TelemetryMetricDirection::minimize
             ? number < selected.at("metric").at("value").get<double>()
             : number > selected.at("metric").at("value").get<double>()))
      selected = row;
  }
  return selected;  // Equal finite observations retain the earliest frozen
                    // trial index.
}

Json describe(SQLiteRunStore &store, const RunMetadata &search,
              const Plan &plan, bool active = false,
              std::string_view override_status = {}) {
  const auto rows = results(store, search, plan);
  const auto current = search_metadata(store, search.run_id);
  Json value{
      {"schema_version", 1},
      {"search_id", search.run_id},
      {"search_spec_sha256", plan.sha256},
      {"status", override_status.empty() ? run_status_name(current.status)
                                         : override_status},
      {"owner_active", active},
      {"cancel_requested",
       store.find_reference(search.run_id, "search.cancel").has_value()},
      {"strategy", plan.json.at("strategy")},
      {"seed", plan.seed},
      {"space_cardinality", plan.cardinality},
      {"planned_trials", plan.sequence.size()},
      {"confirmed_trials", rows.size()},
      {"failure_count", std::count_if(rows.begin(), rows.end(),
                                      [](const Json &row) {
                                        return !row.at("eligible").get<bool>();
                                      })},
      {"trials", rows},
      {"best", best(rows, plan)},
      {"tie_policy", "earliest_trial_index"},
      {"parallel_trials", 1}};
  if (store.find_reference(search.run_id, "search.result"))
    value["stop_reason"] =
        load_json(store, search.run_id, "search.result").at("stop_reason");
  return value;
}

Json confirmed_summary(SQLiteRunStore &store, const RunMetadata &search,
                       const Plan &plan, const std::vector<Json> &rows) {
  auto summary = load_json(store, search.run_id, "search.result");
  fields(summary,
         {"format", "schema_version", "search_id", "search_spec_sha256",
          "status", "stop_reason", "confirmed_trials", "best"});
  const auto reason = text(summary.at("stop_reason"));
  require(reason == "cancelled" || reason == "wall_time_budget" ||
              reason == "failure_budget" || reason == "max_trials" ||
              reason == "space_exhausted",
          "Unknown confirmed search stop reason");
  const auto selected = best(rows, plan);
  const auto status = reason == "cancelled" ? RunStatus::cancelled
                      : selected.is_null()  ? RunStatus::failed
                                            : RunStatus::completed;
  require(text(summary.at("format")) == "jyppx.liuding.local_search_result" &&
              positive(summary.at("schema_version"), 1) == 1 &&
              text(summary.at("search_id")) == search.run_id &&
              text(summary.at("search_spec_sha256")) == plan.sha256 &&
              unsigned_number(summary.at("confirmed_trials"), kMaximumTrials) ==
                  rows.size() &&
              text(summary.at("status")) == run_status_name(status) &&
              summary.at("best") == selected,
          "Search final result identity/count/selection mismatch");
  if (is_terminal_run_status(search.status))
    require(search.status == status,
            "Search lifecycle differs from its confirmed summary");
  if (reason == "max_trials" || reason == "space_exhausted")
    require(rows.size() == plan.sequence.size(),
            "Exhausted search has unconfirmed trials");
  return summary;
}

Result finalize(SQLiteRunStore &store, const RunMetadata &search,
                const Plan &plan, std::string_view reason, bool cancelled) {
  const auto rows = results(store, search, plan);
  const auto selected = best(rows, plan);
  const auto status = cancelled            ? RunStatus::cancelled
                      : selected.is_null() ? RunStatus::failed
                                           : RunStatus::completed;
  const auto summary = Json{{"format", "jyppx.liuding.local_search_result"},
                            {"schema_version", 1},
                            {"search_id", search.run_id},
                            {"search_spec_sha256", plan.sha256},
                            {"status", run_status_name(status)},
                            {"stop_reason", reason},
                            {"confirmed_trials", rows.size()},
                            {"best", selected}};
  put_json(store, search.run_id, "search.result", "search.result", summary,
           search.created_at_unix_ms);
  if (!selected.is_null())
    put_json(store, search.run_id, "search.best", "search.best", selected,
             search.created_at_unix_ms);
  const auto current = search_metadata(store, search.run_id);
  store.transition(search.run_id, status,
                   std::max(now_ms(), current.updated_at_unix_ms));
  return {describe(store, search, plan), cancelled            ? 130
                                         : selected.is_null() ? 1
                                                              : 0};
}

Result drive(SQLiteRunStore &store, const fs::path &root,
             const RunMetadata &search, const Plan &plan) {
  auto completed = results(store, search, plan);
  if (store.find_reference(search.run_id, "search.result")) {
    const auto summary = confirmed_summary(store, search, plan, completed);
    return finalize(store, search, plan,
                    summary.at("stop_reason").get<std::string>(),
                    summary.at("status").get<std::string>() == "cancelled");
  }
  require(!is_terminal_run_status(search_metadata(store, search.run_id).status),
          "Search terminated without confirmed result evidence");
  require(search.created_at_unix_ms <=
              std::numeric_limits<std::uint64_t>::max() - plan.duration,
          "Search deadline overflow");
  const auto deadline = search.created_at_unix_ms + plan.duration;
  auto failures = static_cast<std::uint64_t>(std::count_if(
      completed.begin(), completed.end(),
      [](const Json &row) { return !row.at("eligible").get<bool>(); }));
  for (std::uint64_t index = 1; index <= plan.sequence.size(); ++index) {
    if (completed.size() >= index) continue;
    const auto cancel =
        store.find_reference(search.run_id, "search.cancel").has_value();
    const auto wall = now_ms();
    require(
        wall >= search.created_at_unix_ms,
        "Wall clock moved before the frozen search start; preserve evidence");
    auto value = trial(plan, search.run_id, root, index, deadline);
    const auto admission_id = "search.admission." + std::to_string(index);
    const auto child = store.find(value.id);
    const auto admitted =
        store.find_reference(search.run_id, admission_id).has_value();
    require(!child || admitted,
            "Existing trial job has no prior frozen search admission; preserve "
            "evidence");
    require(child || !store.find(value.training),
            "Training run exists before its trial was admitted");
    // A child job may still own a process tree after the search caller died.
    // Even expired/cancelled searches must resolve that child before stopping.
    if (!child && (cancel || wall >= deadline || failures >= plan.failures))
      return finalize(store, search, plan,
                      cancel             ? "cancelled"
                      : wall >= deadline ? "wall_time_budget"
                                         : "failure_budget",
                      cancel);
    materialize(value);
    if (plan.json.contains("recipe_bridge") && !child)
      validate_recipe_trial_inputs(store, plan, value);
    validate_local_execution_spec(value.execution, !child);
    const auto record = admission(value, index, plan);
    if (admitted)
      require(load_json(store, search.run_id, admission_id) == record,
              "Frozen trial admission differs from the search plan");
    else
      put_json(store, search.run_id, admission_id, admission_id, record,
               search.created_at_unix_ms);
    const auto config_id = "search.config." + std::to_string(index);
    put_json(store, search.run_id, config_id, config_id, value.configuration,
             search.created_at_unix_ms);
    seal_trial_inputs(store, search, plan, value, index, child.has_value());
    if (child) {
      require(child->project_id == plan.project &&
                  child->experiment_id == plan.experiment &&
                  child->lineage.parent_run_id == search.run_id &&
                  child->lineage.config_digest ==
                      digest(value.execution.dump()).hex &&
                  child->lineage.source_digest ==
                      text(value.execution.at("executable_sha256")),
              "Existing trial job differs from its frozen admission");
      if ((cancel || wall >= deadline || failures >= plan.failures) &&
          store.find_reference(value.id, "local.spec"))
        static_cast<void>(control_local_job("cancel", root, value.id));
    }
    std::atomic<bool> monitor_failed{false};
    Result execution;
    {
      std::jthread monitor([&](std::stop_token stop) {
        try {
          SQLiteRunStore observer(root);
          while (!stop.stop_requested()) {
            if (observer.find_reference(search.run_id, "search.cancel") ||
                now_ms() >= deadline) {
              if (observer.find(value.id) &&
                  observer.find_reference(value.id, "local.spec")) {
                static_cast<void>(control_local_job("cancel", root, value.id));
                return;
              }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
          }
        } catch (...) {
          monitor_failed = true;
          try {
            static_cast<void>(control_local_job("cancel", root, value.id));
          } catch (...) {
          }
        }
      });
      // A completed or unconfirmed native attempt is recovered from run/CAS.
      // Do not require its old executable to still exist, or relaunch a worker
      // merely because the search caller died before recording its own row.
      execution = child && store.find_reference(value.id, "local.spec")
                      ? control_local_job("recover", root, value.id)
                      : execute_local_job(root, value.id, value.execution,
                                          search.run_id);
      monitor.request_stop();
    }
    if (monitor_failed.load())
      throw IoError(
          "Search cancellation monitoring failed; execution was conservatively "
          "stopped");
    if (execution.exit_code == 7)
      return {describe(store, search, plan,
                       execution.data.value("owner_active", false),
                       execution.data.at("status").get<std::string>()),
              7};
    if (plan.json.contains("recipe_bridge"))
      publish_recipe_training(store, search, plan, value, execution.data);
    const auto row = evaluate(store, plan, value, index, execution.data);
    const auto reference = "search.result." + std::to_string(index);
    put_json(store, search.run_id, reference, reference, row,
             search.created_at_unix_ms);
    if (!row.at("eligible").get<bool>()) ++failures;
    completed.push_back(row);
    if (store.find_reference(search.run_id, "search.cancel"))
      return finalize(store, search, plan, "cancelled", true);
    if (now_ms() >= deadline)
      return finalize(store, search, plan, "wall_time_budget", false);
    if (failures >= plan.failures)
      return finalize(store, search, plan, "failure_budget", false);
  }
  // A caller can die after the final per-trial result commit and before the
  // terminal summary. Replaying that prefix must still honor a durable cancel
  // and the frozen budgets, rather than treating the loop skip as completion.
  if (store.find_reference(search.run_id, "search.cancel"))
    return finalize(store, search, plan, "cancelled", true);
  if (now_ms() >= deadline)
    return finalize(store, search, plan, "wall_time_budget", false);
  if (failures >= plan.failures)
    return finalize(store, search, plan, "failure_budget", false);
  return finalize(
      store, search, plan,
      plan.sequence.size() < plan.trials ? "space_exhausted" : "max_trials",
      false);
}

Result export_search(SQLiteRunStore &store, const fs::path &root,
                     const RunMetadata &search, const Plan &plan,
                     const Arguments &arguments) {
  require(
      is_terminal_run_status(search_metadata(store, search.run_id).status) &&
          store.find_reference(search.run_id, "search.best").has_value(),
      "Finish the search and confirm a best trial before exporting");
  const auto selected = load_json(store, search.run_id, "search.best");
  const auto rows = results(store, search, plan);
  static_cast<void>(confirmed_summary(store, search, plan, rows));
  require(selected == best(rows, plan),
          "Best selection differs from confirmed trial evidence");
  const auto index = positive(selected.at("index"), kMaximumTrials);
  const auto frozen_root =
      plan.json.contains("recipe_bridge")
          ? fs::u8path(
                text(load_json(store, search.run_id,
                               "search.admission." + std::to_string(index))
                         .at("store_root")))
          : root;
  const auto value = trial(plan, search.run_id, frozen_root, index,
                           search.created_at_unix_ms + plan.duration);
  const auto local = control_local_job("show", root, value.id);
  require(evaluate(store, plan, value, index, local.data) == selected,
          "Best training/metric/output lineage no longer matches confirmed "
          "evidence");
  const auto output = fs::u8path(arguments.require("output"));
  require(output.is_absolute(),
          "Search export requires an absolute new destination");
  const auto backup = store.backup(output);
  return {{{"search_id", search.run_id},
           {"best", selected},
           {"output", path_text(output)},
           {"format", "jyppx.run-store-backup/v1"},
           {"contains_all_trial_history", true},
           {"deployable_model_bundle", false},
           {"database_bytes", backup.database_bytes},
           {"artifact_bytes", backup.artifact_bytes},
           {"artifact_count", backup.artifact_count}},
          0};
}
}  // namespace

Result local_search_command(const Arguments &arguments) {
  const auto action = arguments.words().at(1);
  require(action == "execute" || action == "show" || action == "recover" ||
              action == "cancel" || action == "export",
          "Use search execute/show/recover/cancel/export");
  if (action == "execute")
    arguments.allow({"root", "id", "spec"});
  else if (action == "export")
    arguments.allow({"root", "id", "output"});
  else
    arguments.allow({"root", "id"});
  const auto id = arguments.require("id");
  identifier(id);
  const auto root =
      fs::absolute(fs::u8path(arguments.require("root"))).lexically_normal();
  std::optional<Plan> requested;
  bool files_preflighted = false;
  const auto preflight_files = [&](const Plan &plan) {
    auto probe =
        trial(plan, id, root, 1, std::numeric_limits<std::int64_t>::max());
    if (text(plan.json.at("execution").at("working_directory")) ==
        "${trial_directory}")
      probe.execution["working_directory"] = path_text(
          fs::u8path(text(probe.execution.at("executable"))).parent_path());
    validate_local_execution_spec(probe.execution);
  };
  if (action == "execute") {
    requested = read_plan(parse_json(
        read_text(fs::u8path(arguments.require("spec")), kPlanBytes)));
    // Every chosen value must fit native argv/environment limits before any
    // admission. Inspect executable bytes only once, not once per candidate.
    for (std::uint64_t index = 1; index <= requested->sequence.size();
         ++index) {
      const auto probe = trial(*requested, id, root, index,
                               std::numeric_limits<std::int64_t>::max());
      validate_local_execution_spec(probe.execution, false);
      require(
          admission(probe, index, *requested).dump().size() <= kEvidenceBytes,
          "Search admission exceeds its evidence byte bound");
    }
    if (!fs::is_regular_file(root / "runs.sqlite3")) {
      preflight_files(*requested);
      files_preflighted = true;
    }
  } else {
    require(fs::is_regular_file(root / "runs.sqlite3"),
            "Search storage does not exist");
  }
  SQLiteRunStore store(root);
  ordinary(root);
  if (action == "cancel") {
    const auto search = search_metadata(store, id);
    const auto plan = read_plan(load_json(store, id, "search.spec"));
    validate_identity(search, plan);
    if (!is_terminal_run_status(search.status) &&
        !store.find_reference(id, "search.result"))
      put_json(store, id, "search.cancel", "search.cancel",
               {{"search_id", id}, {"search_spec_sha256", plan.sha256}},
               search.created_at_unix_ms);
    for (std::uint64_t index = 1; index <= plan.sequence.size(); ++index) {
      const auto child = store.find(trial_id(id, index));
      if (child && !is_terminal_run_status(child->status) &&
          store.find_reference(child->run_id, "local.spec")) {
        require(child->lineage.parent_run_id == id &&
                    store
                        .find_reference(
                            id, "search.admission." + std::to_string(index))
                        .has_value(),
                "Cancel requires a child owned by a prior search admission");
        static_cast<void>(control_local_job("cancel", root, child->run_id));
      }
    }
    return {describe(store, search, plan), 0};
  }
  SearchLock ownership(root, id);
  if (requested && !store.find(id)) {
    require(ownership.held(), "Another caller owns search creation");
    if (!files_preflighted) preflight_files(*requested);
    validate_inputs(store, *requested);
    const auto stamp = now_ms();
    RunMetadata metadata;
    metadata.run_id = id;
    metadata.project_id = requested->project;
    metadata.experiment_id = requested->experiment;
    metadata.display_name = "Local search " + id;
    metadata.status = RunStatus::running;
    metadata.created_at_unix_ms = stamp;
    metadata.updated_at_unix_ms = stamp;
    metadata.lineage.source_digest = requested->json.at("execution")
                                         .at("executable_sha256")
                                         .get<std::string>();
    metadata.lineage.build_fingerprint = digest(build_fingerprint_json()).hex;
    metadata.lineage.config_digest = requested->sha256;
    metadata.tags["liuding.search.kind"] = "search";
    store.start(metadata);
  }
  const auto search = search_metadata(store, id);
  if (requested) {
    require(search.lineage.config_digest == requested->sha256,
            "Existing search has a different frozen plan");
    if (!store.find_reference(id, "search.spec")) {
      require(ownership.held(), "Search plan attach requires ownership");
      put_json(store, id, "search.spec", "search.spec", requested->json,
               search.created_at_unix_ms);
    }
  }
  const auto plan = read_plan(load_json(store, id, "search.spec"));
  validate_identity(search, plan);
  if (action == "show")
    return {describe(store, search, plan, !ownership.held()), 0};
  if (!ownership.held())
    return {describe(store, search, plan, true, "active"), 7};
  validate_inputs(store, plan);
  if (action == "export")
    return export_search(store, root, search, plan, arguments);
  return drive(store, root, search, plan);
}
}  // namespace jyppx::liuding::cli
