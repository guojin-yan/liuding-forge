// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "publication.hpp"

#include "fixture.hpp"

#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sqlite3.h>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace diagnostic_recovery {
namespace {
std::filesystem::path active_root;
std::string active_boundary;
std::string active_digest;
unsigned native_calls = 0;

int before_commit(void*) noexcept {
  if (active_boundary == "before-reference-commit") {
    try {
      fence(active_boundary);
    } catch (...) {
      std::_Exit(92);
    }
  }
  return 0;
}
int extension(sqlite3* connection, char**, const sqlite3_api_routines*) {
  sqlite3_commit_hook(connection, &before_commit, nullptr);
  return SQLITE_OK;
}
}  // namespace

void install_commit_hook() {
  require(sqlite3_auto_extension(reinterpret_cast<void (*)()>(&extension)) ==
              SQLITE_OK,
          "official SQLite hook registration failed");
}

void arm_publication(std::filesystem::path root, std::string boundary,
                     std::string digest) {
  active_root = std::move(root);
  active_boundary = std::move(boundary);
  active_digest = std::move(digest);
}

[[noreturn]] void fence(std::string_view boundary) {
#ifdef _WIN32
  const auto pid = ::_getpid();
#else
  const auto pid = ::getpid();
#endif
  const nlohmann::json ready{{"phase", "ready-for-forced-kill"},
                             {"process_id", pid},
                             {"boundary", boundary},
                             {"native_calls", native_calls}};
  std::cout << ready.dump() << '\n' << std::flush;
  require(static_cast<bool>(std::cout), "readiness could not be flushed");
  char value = 0;
  std::cin.get(value);
  std::_Exit(91);
}
}  // namespace diagnostic_recovery

// The test executable supplies only the private forwarding archive member.
// Actual file publication and SQLite commits use the production SDK routines.
namespace jyppx::liuding::experiment::detail {
std::error_code native_publish_artifact_file(
    const std::filesystem::path&, const std::filesystem::path&) noexcept;

std::error_code publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  using namespace diagnostic_recovery;
  if (!active_boundary.empty()) {
    const auto expected = active_root / "artifacts" / "objects" /
                          active_digest.substr(0, 2) / active_digest.substr(2);
    require(destination == expected, "unexpected report CAS destination");
    if (active_boundary == "before-cas") fence(active_boundary);
  }
  const auto error = native_publish_artifact_file(source, destination);
  if (!error && !active_boundary.empty()) {
    ++native_calls;
    if (active_boundary == "after-cas") fence(active_boundary);
  }
  return error;
}
}  // namespace jyppx::liuding::experiment::detail
