// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "cli.hpp"
#include "migration_files.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <utility>

namespace jyppx::liuding::cli {
namespace {
struct Rule {
  std::string_view needle;
  std::string_view feature;
  std::string_view replacement;
};

// This table records retired public surfaces only. A proposed new include does
// not imply that constructors, state/optimizer semantics, or ABI are identical.
constexpr std::array rules{
    Rule{"jyppx_training_framework", "SDK-PACKAGE",
         "Use find_package(jyppx CONFIG) and explicit jyppx::liuding:: components."},
    Rule{"jyppx/version.hpp", "FOUNDATION-VERSION",
         "Review jyppx/liuding/foundation/version.hpp."},
    Rule{"jyppx/utils/logger.hpp", "TELEMETRY-LOGGING",
         "Review foundation/telemetry.hpp and training/event_sink.hpp "
         "contracts."},
    Rule{"jyppx/utils/tensorboard.hpp", "TELEMETRY-SIDECAR",
         "Use the explicit telemetry sidecar export contract."},
    Rule{"jyppx::utils", "FOUNDATION-OWNERSHIP",
         "Select the owning foundation/data/training API; no general namespace "
         "replacement."},
    Rule{"jyppx/data/dataloader.hpp", "DATA-PIPELINE",
         "Review jyppx/liuding/data/data_pipeline.hpp and explicit checkpoint/worker "
         "semantics."},
    Rule{"jyppx/data/thread_safe_queue.hpp", "DATA-WORKER",
         "Review the owned data pipeline worker contract; retired utility is "
         "not public."},
    Rule{"jyppx/nn/losses.hpp", "NN-LOSS-SPLIT",
         "Select the individual jyppx/liuding/nn/loss/ functional/module header."},
    Rule{"jyppx/nn/init.hpp", "NN-INIT-RETIREMENT",
         "Use explicit LibTorch initialization with caller-owned RNG policy."},
    Rule{"jyppx/nn/quantization.hpp", "QUANTIZATION-RESERVED",
         "Removed placeholder; no v1.0 replacement capability is claimed."},
    Rule{"jyppx/distributed/", "DISTRIBUTED-RESERVED",
         "Removed placeholder; v1.0 has no distributed strategy support."},
    Rule{"jyppx::distributed", "DISTRIBUTED-RESERVED",
         "Removed placeholder; use an explicitly supported single-device "
         "strategy."},
    Rule{"jyppx/optim/sgd.hpp", "OPTIM-SGD",
         "Review jyppx/liuding/optim/sgd_adapter.hpp; recreate optimizer state "
         "explicitly."},
    Rule{"jyppx/optim/adam.hpp", "OPTIM-ADAM",
         "Review jyppx/liuding/optim/adam_adapter.hpp and participant topology."},
    Rule{"jyppx/optim/adamw.hpp", "OPTIM-ADAMW",
         "Review jyppx/liuding/optim/adamw_adapter.hpp and participant topology."},
    Rule{"jyppx/optim/adagrad.hpp", "OPTIM-ADAGRAD",
         "Review jyppx/liuding/optim/adagrad_adapter.hpp and explicit N-1 state "
         "reader."},
    Rule{"jyppx/optim/rmsprop.hpp", "OPTIM-RMSPROP",
         "Review jyppx/liuding/optim/rmsprop_adapter.hpp and explicit N-1 state "
         "reader."},
    Rule{"jyppx/optim/lamb.hpp", "OPTIM-LAMB",
         "Review jyppx/liuding/optim/lamb_adapter.hpp and explicit native algorithm "
         "contract."},
    Rule{"jyppx/optim/nadam.hpp", "OPTIM-NADAM",
         "Review jyppx/liuding/optim/nadam_adapter.hpp and explicit native algorithm "
         "contract."},
    Rule{"jyppx/optim/optimizer.hpp", "OPTIM-ADAPTER",
         "Review jyppx/liuding/optim/optimizer_adapter.hpp and "
         "optimizer_rate_control.hpp."},
    Rule{"jyppx/optim/schedulers/", "OPTIM-SCHEDULER",
         "Select epoch/event/step/composite/linear scheduler contracts; "
         "signatures differ."},
    Rule{"jyppx/training/callback.hpp", "TRAINING-CALLBACK",
         "Review jyppx/liuding/training/callback_dispatcher.hpp; transactional "
         "lifecycle differs."},
    Rule{"jyppx/training/callbacks/early_stopping.hpp",
         "TRAINING-EARLY-STOPPING",
         "Review jyppx/liuding/training/callbacks/early_stopping_callback.hpp."},
    Rule{"jyppx/training/callbacks/logging_callback.hpp", "TRAINING-TELEMETRY",
         "Review durable telemetry/event sink integration."},
    Rule{"jyppx/training/callbacks/lr_scheduler_callback.hpp",
         "TRAINING-SCHEDULER", "Review jyppx/liuding/training/scheduler_adapter.hpp."},
    Rule{"jyppx/training/callbacks/model_checkpoint.hpp", "TRAINING-CHECKPOINT",
         "Review checkpoint_coordinator.hpp and file_checkpoint_store.hpp."},
    Rule{"jyppx/training/callbacks/progress_callback.hpp", "TRAINING-TELEMETRY",
         "Review telemetry observers; no implicit UI/progress callback "
         "replacement."},
    Rule{
        "jyppx/training/callbacks/tensorboard_callback.hpp",
        "TELEMETRY-SIDECAR",
        "Use explicit run export --backend tensorboard with a locked sidecar."},
};

bool source_file(const std::filesystem::path& path) {
  const auto extension = path.extension().string();
  return path.filename() == "CMakeLists.txt" || extension == ".cmake" ||
         extension == ".cpp" || extension == ".cc" || extension == ".cxx" ||
         extension == ".c" || extension == ".h" || extension == ".hpp" ||
         extension == ".hxx" || extension == ".ipp" || extension == ".inl";
}

bool skipped_directory(const std::filesystem::path& path) {
  const auto name = path.filename().string();
  return name == ".git" || name == ".hg" || name == ".svn" || name == "out" ||
         name == "build" || name == "third_party" || name == "node_modules" ||
         name == ".venv";
}

Result scan(const Arguments& arguments) {
  arguments.allow({"root", "max-files", "max-bytes", "max-findings"});
  const auto root = migration_files::checked_path(
      migration_files::path(arguments.require("root")));
  if (!std::filesystem::is_directory(root))
    throw InvalidArgument("API scan root must be a directory");
  const auto max_files = arguments.number("max-files", 10000U, 10000U);
  const auto max_bytes =
      arguments.number("max-bytes", 64U * 1024U * 1024U, 64U * 1024U * 1024U);
  const auto max_findings = arguments.number("max-findings", 4096U, 4096U);
  if (max_files == 0U || max_bytes == 0U || max_findings == 0U)
    throw InvalidArgument("API scan limits must be positive");
  std::vector<std::filesystem::path> files;
  std::size_t visited = 0U;
  for (auto iterator = std::filesystem::recursive_directory_iterator(root);
       iterator != std::filesystem::recursive_directory_iterator();
       ++iterator) {
    if (++visited > 50000U)
      throw ResourceExhausted("API scan directory entry limit exceeded");
    migration_files::checked_path(iterator->path());
    if (iterator.depth() > 32)
      throw ResourceExhausted("API scan directory depth exceeds 32");
    if (iterator->is_directory()) {
      if (skipped_directory(iterator->path()))
        iterator.disable_recursion_pending();
      continue;
    }
    if (!iterator->is_regular_file())
      throw InvalidArgument("API scan requires regular source files");
    if (!source_file(iterator->path())) continue;
    files.push_back(iterator->path());
    if (files.size() > max_files)
      throw ResourceExhausted("API scan file count exceeds limit");
  }
  std::sort(files.begin(), files.end());
  Json findings = Json::array();
  std::size_t total_bytes = 0U;
  for (const auto& file : files) {
    const auto content = migration_files::read(file, 2U * 1024U * 1024U);
    if (content.size() > max_bytes - total_bytes)
      throw ResourceExhausted("API scan aggregate byte limit exceeded");
    total_bytes += content.size();
    const auto relative = file.lexically_relative(root).generic_u8string();
    const auto name = std::string(
        reinterpret_cast<const char*>(relative.data()), relative.size());
    if (name.size() > 1024U)
      throw ResourceExhausted("API scan relative path exceeds limit");
    std::vector<std::size_t> newlines;
    for (std::size_t offset = 0U; offset < content.size(); ++offset)
      if (content[offset] == '\n') newlines.push_back(offset);
    std::set<std::pair<std::size_t, std::string_view>> emitted;
    for (const auto& rule : rules) {
      for (std::size_t position = 0U;
           (position = content.find(rule.needle, position)) !=
           std::string::npos;
           position += rule.needle.size()) {
        const auto line = 1U + static_cast<std::size_t>(
                                   std::lower_bound(newlines.begin(),
                                                    newlines.end(), position) -
                                   newlines.begin());
        if (!emitted.emplace(line, rule.needle).second) continue;
        if (findings.size() == max_findings)
          throw ResourceExhausted("API scan findings exceed limit");
        findings.push_back({{"path", name},
                            {"line", line},
                            {"feature_id", rule.feature},
                            {"retired_surface", rule.needle},
                            {"suggestion", rule.replacement},
                            {"automatic_change", false}});
      }
    }
  }
  return {
      Json{{"scanner", "bounded-lexical-v1"},
           {"source_untouched", true},
           {"files_scanned", files.size()},
           {"bytes_scanned", total_bytes},
           {"findings", findings},
           {"limitation",
            "Lexical diagnostics include comments/strings and inactive code; "
            "they do not prove AST completeness, API equivalence or ABI "
            "compatibility."}},
      findings.empty() ? 0 : 1};
}
}  // namespace

Result api_command(const Arguments& arguments) {
  if (arguments.words()[1] != "scan")
    usage("api", "Use api scan --root SOURCE_DIRECTORY.");
  try {
    return scan(arguments);
  } catch (const std::filesystem::filesystem_error&) {
    throw IoError("Could not inspect API migration source tree");
  }
}
}  // namespace jyppx::liuding::cli
