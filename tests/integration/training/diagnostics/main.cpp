// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "fixture.hpp"
#include "publication.hpp"

#include <jyppx/liuding/experiment/run_store.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>

namespace {
namespace ex = jyppx::liuding::experiment;
using namespace diagnostic_recovery;

class CleanupMarker final {
 public:
  explicit CleanupMarker(std::filesystem::path root) : root_(std::move(root)) {}
  ~CleanupMarker() { write_file(root_ / "cleanup.marker", "stack unwound\n"); }

 private:
  std::filesystem::path root_;
};

void execute(std::string_view mode, const std::filesystem::path& root,
             std::string_view boundary) {
  install_commit_hook();
  require(std::filesystem::is_directory(root), "fixture root must exist");
  ex::SQLiteRunStore store(root);
  if (mode == "seed") {
    seed_checkpoint(root);
    write_file(root / "expected.json", checkpoint_report(root));
    store.start({.run_id = "run-1",
                 .project_id = "project-1",
                 .experiment_id = "experiment-1",
                 .display_name = "diagnostic recovery fixture",
                 .created_at_unix_ms = 10,
                 .updated_at_unix_ms = 10});
  } else {
    require(mode == "crash" || mode == "replay" || mode == "conflict" ||
                mode == "gc",
            "unknown mode");
    if (mode == "gc") {
      ex::ArtifactStoreLimits limits;
      limits.stale_temp_age = std::chrono::seconds::zero();
      ex::FileArtifactStore cas(root / "artifacts", limits);
      (void)cas.recover();
      (void)store.collect_garbage(std::chrono::seconds::zero());
    } else {
      const auto bytes = checkpoint_report(root, mode == "conflict");
      if (mode != "conflict")
        require(bytes == read_file(root / "expected.json"),
                "restored report differs from frozen bytes");
      const auto publish = [&] {
        return store.put_artifact(
            "run-1", "diagnostics-1", ex::RunArtifactKind::artifact,
            "training.diagnostics", std::as_bytes(std::span(bytes)),
            {.logical_name = "training-diagnostics.json",
             .media_type =
                 "application/vnd.jyppx.training-diagnostics+json;version=1",
             .producer = "training.diagnostics/v1",
             .expected_size_bytes = bytes.size()},
            20);
      };
      if (mode == "crash") {
        require(boundary == "before-cas" || boundary == "after-cas" ||
                    boundary == "before-reference-commit" ||
                    boundary == "after-reference-before-ack",
                "unknown boundary");
        CleanupMarker marker(root);
        arm_publication(
            root, std::string(boundary),
            jyppx::liuding::sha256_hex(std::as_bytes(std::span(bytes))));
        (void)publish();
        require(boundary == "after-reference-before-ack",
                "publication escaped its fence");
        fence(boundary);
      } else if (mode == "conflict") {
        bool rejected = false;
        try {
          (void)publish();
        } catch (const jyppx::liuding::InvalidArgument&) {
          rejected = true;
        }
        require(rejected, "conflicting report silently replaced reference");
      } else {
        (void)publish();
      }
    }
  }
  require(store.integrity_check().ok(), "native store integrity failed");
  const nlohmann::json result{{"phase", "completed"},
                              {"mode", mode},
                              {"references", store.references("run-1").size()},
                              {"integrity_ok", true}};
  std::cout << result.dump() << '\n';
}

int guarded(std::string_view mode, const std::filesystem::path& root,
            std::string_view boundary) {
  try {
    execute(mode, std::filesystem::absolute(root), boundary);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "diagnostic-recovery: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  if (argc != 3 && argc != 4) return 2;
  const auto ascii = [](const wchar_t* value) {
    return std::filesystem::path(value).string();
  };
  return guarded(ascii(argv[1]), std::filesystem::path(argv[2]),
                 argc == 4 ? ascii(argv[3]) : "");
}
#else
int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  return guarded(argv[1], std::filesystem::path(argv[2]),
                 argc == 4 ? argv[3] : "");
}
#endif
