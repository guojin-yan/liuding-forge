// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/local_execution/process.hpp>
#include <jyppx/liuding/experiment/local_execution/runner.hpp>
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <span>
#include <thread>

namespace {
namespace ex = jyppx::liuding::experiment;
namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace std::chrono_literals;
std::string text(const fs::path& path) {
  const auto v = path.generic_u8string();
  return {reinterpret_cast<const char*>(v.data()), v.size()};
}
class LocalJobSdkTest : public testing::Test {
 protected:
  fs::path temporary;
  fs::path executable = fs::u8path(JYPPX_LOCAL_JOB_SDK_FIXTURE);
  std::string sha;
  void SetUp() override {
    static std::atomic<unsigned> sequence{0};
    temporary =
        fs::temp_directory_path() /
        ("liuding-sdk-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()) +
         "-" + std::to_string(sequence++));
    ASSERT_TRUE(fs::create_directory(temporary));
    std::ifstream file(executable, std::ios::binary);
    const std::string content{std::istreambuf_iterator<char>(file), {}};
    ASSERT_TRUE(file.eof() || static_cast<bool>(file));
    sha = jyppx::liuding::sha256_hex(
        std::as_bytes(std::span(content.data(), content.size())));
  }
  void TearDown() override { fs::remove_all(temporary); }
  ex::LocalJobWorkerLaunch worker() { return {executable, sha, {"--worker"}}; }
  Json spec() {
    return {{"format", "jyppx.liuding.local_run_spec"},
            {"schema_version", 1},
            {"project_id", "sdk"},
            {"experiment_id", "contracts"},
            {"executable", text(executable)},
            {"executable_sha256", sha},
            {"arguments", {"--wait"}},
            {"working_directory", text(temporary)},
            {"environment", Json::object()},
            {"inherit_environment", true},
            {"timeout_ms", 5000},
            {"max_attempts", 1},
            {"retry_exit_codes", Json::array()}};
  }
};
TEST_F(LocalJobSdkTest, NativeExampleConfirmsTypedRetryAndFreshProcessReplay) {
  ex::LocalProcessOptions command;
  command.executable = executable;
  command.arguments = {text(temporary / "example")};
  command.working_directory = temporary;
  command.inherit_environment = true;
  command.timeout = 15s;
  const auto result = ex::LocalProcessRunner{}.run(command);
  ASSERT_EQ(result.exit_code, 0) << result.stderr_capture;
  ASSERT_TRUE(result.process_tree_cleanup_confirmed);
  const auto output = Json::parse(result.stdout_capture);
  EXPECT_TRUE(output.at("passed").get<bool>());
  EXPECT_TRUE(output.at("fresh_process_replay").get<bool>());
  EXPECT_TRUE(output.at("typed_output_confirmed").get<bool>());
  EXPECT_EQ(output.at("attempts"), 2);
}
TEST_F(LocalJobSdkTest, InvalidWorkerIdentityAndJsonRejectBeforeStoreCreation) {
  const auto root = temporary / "invalid";
  auto launch = worker();
  launch.executable_sha256 = std::string(64, '0');
  EXPECT_THROW(ex::execute_local_job(root, "job", spec().dump(), launch),
               jyppx::liuding::Error);
  EXPECT_FALSE(fs::exists(root));
  launch = worker();
  launch.arguments = {std::string("bad\0argument", 12)};
  EXPECT_THROW(ex::execute_local_job(root, "job", spec().dump(), launch),
               jyppx::liuding::Error);
  EXPECT_FALSE(fs::exists(root));
  EXPECT_THROW(
      ex::execute_local_job(root, "job", "{\"x\":1,\"x\":2}", worker()),
      jyppx::liuding::Error);
  EXPECT_FALSE(fs::exists(root));
  EXPECT_THROW(ex::run_local_job_worker(root, "job", 0, std::string(64, '0')),
               jyppx::liuding::Error);
  EXPECT_FALSE(fs::exists(root));
}
TEST_F(LocalJobSdkTest,
       CancellationCommitsCleanupAndTerminalRecoveryNeverRelaunches) {
  const auto root = temporary / "cancel";
  auto running = std::async(std::launch::async, [&] {
    return ex::execute_local_job(root, "job", spec().dump(), worker());
  });
  bool launched = false;
  const auto deadline = std::chrono::steady_clock::now() + 8s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (fs::exists(root / "runs.sqlite3")) {
      try {
        const auto value = Json::parse(
            ex::control_local_job(ex::LocalJobControl::show, root, "job").json);
        if (!value.at("attempts").empty()) {
          ex::SQLiteRunStore store(root);
          launched = store
                         .find_reference(value.at("attempts")[0]
                                             .at("attempt_id")
                                             .get<std::string>(),
                                         "local.owner")
                         .has_value();
          if (launched) break;
        }
      } catch (const jyppx::liuding::Error&) {
      }
    }
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_TRUE(launched);
  EXPECT_EQ(ex::control_local_job(ex::LocalJobControl::cancel, root, "job")
                .status_code,
            0);
  const auto final = running.get();
  ASSERT_EQ(final.status_code, 130) << final.json;
  const auto value = Json::parse(final.json);
  EXPECT_EQ(value.at("status"), "cancelled");
  EXPECT_TRUE(value.at("attempts")[0]
                  .at("result")
                  .at("process_tree_cleanup_confirmed")
                  .get<bool>());
  EXPECT_EQ(
      ex::control_local_job(ex::LocalJobControl::recover, root, "job").json,
      final.json);
}
TEST_F(LocalJobSdkTest,
       AmbiguousIntentStaysFencedUntilExplicitExternalConfirmation) {
  const auto root = temporary / "unknown";
  auto launch = worker();
  launch.arguments = {"--reject-worker"};
  const auto first = ex::execute_local_job(root, "job", spec().dump(), launch);
  ASSERT_EQ(first.status_code, 7) << first.json;
  const auto recovered = ex::control_local_job(ex::LocalJobControl::recover,
                                               root, "job", worker());
  ASSERT_EQ(recovered.status_code, 7) << recovered.json;
  const auto value = Json::parse(recovered.json);
  ASSERT_EQ(value.at("attempts").size(), 1);
  auto confirmation = value.at("attempts")[0].at("abandonment_request");
  EXPECT_THROW(ex::abandon_local_job(root, "job", confirmation.dump()),
               jyppx::liuding::Error);
  // This test's bootstrap returned before invoking worker or application code.
  // Its synchronous launcher confirmed the entire native tree was reaped.
  confirmation["external_process_tree_cleanup_confirmed"] = true;
  confirmation["confirmed_by"] = "native-sdk-test";
  confirmation["confirmed_at_unix_ms"] = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  confirmation["reason"] = "Designated bootstrap returned before worker entry";
  confirmation["evidence"] =
      "Native fixture --reject-worker exits9 without launching descendants";
  const auto abandoned =
      ex::abandon_local_job(root, "job", confirmation.dump());
  EXPECT_EQ(abandoned.status_code, 1) << abandoned.json;
  EXPECT_EQ(Json::parse(abandoned.json).at("status"), "failed");
  EXPECT_EQ(ex::control_local_job(ex::LocalJobControl::recover, root, "job")
                .status_code,
            1);
}
}  // namespace
