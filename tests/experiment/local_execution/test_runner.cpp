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
#include <stdexcept>
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
  Json log_spec(std::string mode = "success") {
    auto value = spec();
    const auto fixture = fs::u8path(JYPPX_LOCAL_LOG_FIXTURE);
    std::ifstream file(fixture, std::ios::binary);
    const std::string content{std::istreambuf_iterator<char>(file), {}};
    value["executable"] = text(fixture);
    value["executable_sha256"] = jyppx::liuding::sha256_hex(
        std::as_bytes(std::span(content.data(), content.size())));
    value["arguments"] = Json::array({std::move(mode)});
    value["local_retention"] = {{"stdout", "on_failure"},
                                {"stderr", "on_failure"}};
    return value;
  }
  std::string captured(ex::SQLiteRunStore& store, const Json& attempt,
                       std::string_view name) {
    const auto ref =
        store.find_reference(attempt.at("attempt_id").get<std::string>(),
                             "local." + std::string(name));
    if (!ref || ref->artifact.size_bytes > 8192)
      throw std::runtime_error("Missing bounded log reference");
    ex::FileArtifactStore cas(store.artifact_root());
    auto input = cas.open(ref->artifact.digest);
    std::string value(static_cast<std::size_t>(input->size_bytes()), '\0');
    std::size_t offset = 0;
    while (offset != value.size()) {
      const auto got = input->read(
          std::as_writable_bytes(std::span(value.data(), value.size()))
              .subspan(offset));
      if (!got) throw std::runtime_error("Truncated log reference");
      offset += got;
    }
    EXPECT_EQ(value.size(), ref->artifact.size_bytes);
    return value;
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

TEST_F(LocalJobSdkTest, LogRetentionKeepsFailedRetryAndOmitsSuccessfulAttempt) {
  const auto root = temporary / "logs";
  auto value = log_spec("retry");
  value["max_attempts"] = 2;
  value["retry_exit_codes"] = Json::array({75});
  const auto result =
      ex::execute_local_job(root, "job", value.dump(), worker());
  ASSERT_EQ(result.status_code, 0) << result.json;
  const auto attempts = Json::parse(result.json).at("attempts");
  ASSERT_EQ(attempts.size(), 2);
  ex::SQLiteRunStore store(root);
  for (std::size_t i = 0; i != 2; ++i) {
    for (const auto name : {"stdout", "stderr"}) {
      const auto expected = "native " + std::string(name) + "\n";
      const auto& receipt =
          attempts[i].at("result").at("local_retention").at(name);
      EXPECT_EQ(receipt.at("retained"), i == 0);
      EXPECT_EQ(receipt.at("captured_bytes"), expected.size());
      EXPECT_EQ(receipt.at("captured_sha256"),
                jyppx::liuding::sha256_hex(std::as_bytes(
                    std::span(expected.data(), expected.size()))));
      EXPECT_EQ(captured(store, attempts[i], name), i == 0 ? expected : "");
    }
  }
  EXPECT_EQ(ex::execute_local_job(root, "job", value.dump(), {}).json,
            result.json);
  EXPECT_EQ(
      ex::control_local_job(ex::LocalJobControl::recover, root, "job").json,
      result.json);
  value["local_retention"]["stdout"] = "always";
  EXPECT_THROW(ex::execute_local_job(root, "job", value.dump(), worker()),
               jyppx::liuding::Error);
}

TEST_F(LocalJobSdkTest, LogRetentionStoresOnlySelectedStreamAndVerifiesItsCas) {
  const auto root = temporary / "streams";
  auto value = log_spec();
  value["local_retention"] = {{"stdout", "never"}, {"stderr", "always"}};
  const auto result =
      ex::execute_local_job(root, "job", value.dump(), worker());
  ASSERT_EQ(result.status_code, 0) << result.json;
  const auto attempt = Json::parse(result.json).at("attempts").at(0);
  ex::SQLiteRunStore store(root);
  EXPECT_EQ(captured(store, attempt, "stdout"), "");
  EXPECT_EQ(captured(store, attempt, "stderr"), "native stderr\n");
  const auto& receipt = attempt.at("result").at("local_retention");
  const auto dropped =
      receipt.at("stdout").at("captured_sha256").get<std::string>();
  const auto kept =
      receipt.at("stderr").at("captured_sha256").get<std::string>();
  const auto object = [&](const std::string& sha) {
    return store.artifact_root() / "objects" / sha.substr(0, 2) / sha.substr(2);
  };
  EXPECT_FALSE(fs::exists(object(dropped)));
  ASSERT_TRUE(fs::exists(object(kept)));
  std::ofstream(object(kept), std::ios::binary | std::ios::trunc) << "corrupt";
  EXPECT_THROW(ex::control_local_job(ex::LocalJobControl::recover, root, "job"),
               jyppx::liuding::Error);
}

TEST_F(LocalJobSdkTest, LogRetentionKeepsDiagnosticsAfterTypedOutputFailure) {
  const auto root = temporary / "typed";
  auto value = log_spec();
  ex::SQLiteRunStore store(root);
  store.start({.run_id = "inputs",
               .project_id = "sdk",
               .experiment_id = "contracts",
               .display_name = "Inputs",
               .status = ex::RunStatus::running,
               .created_at_unix_ms = 1,
               .updated_at_unix_ms = 1});
  const std::string payload = "{}";
  const auto pin = [&](std::string name) {
    const auto ref = store.put_artifact(
        "inputs", name, ex::RunArtifactKind::artifact, name,
        std::as_bytes(std::span(payload.data(), payload.size())),
        {.logical_name = name, .expected_size_bytes = payload.size()}, 1);
    return Json{{"run_id", "inputs"},
                {"reference_id", name},
                {"sha256", ref.artifact.digest.hex},
                {"size_bytes", payload.size()}};
  };
  value["training"] = {{"model_id", "fixture"},
                       {"configuration", pin("configuration")},
                       {"data_manifest", pin("data_manifest")},
                       {"resume", nullptr},
                       {"outputs", Json::array({{{"reference_id", "missing"},
                                                 {"kind", "artifact"},
                                                 {"role", "fixture"},
                                                 {"max_size_bytes", 4096}}})}};
  store.transition("inputs", ex::RunStatus::completed, 2);
  const auto result =
      ex::execute_local_job(root, "job", value.dump(), worker());
  ASSERT_EQ(result.status_code, 1) << result.json;
  const auto attempt = Json::parse(result.json).at("attempts").at(0);
  EXPECT_EQ(attempt.at("result").at("exit_code"), 0);
  EXPECT_EQ(attempt.at("result").at("training_outputs_confirmed"), false);
  EXPECT_EQ(captured(store, attempt, "stdout"), "native stdout\n");
  EXPECT_EQ(captured(store, attempt, "stderr"), "native stderr\n");
  EXPECT_EQ(
      ex::control_local_job(ex::LocalJobControl::recover, root, "job").json,
      result.json);
}

TEST_F(LocalJobSdkTest, LogRetentionNeverBypassesOutputLimitOrTimeout) {
  for (const auto mode : {"flood", "wait"}) {
    const auto root = temporary / mode;
    auto value = log_spec(mode);
    if (std::string_view(mode) == "flood") {
      value["max_stdout_bytes"] = 32;
      value["local_retention"] = {{"stdout", "never"}, {"stderr", "never"}};
    } else {
      value["timeout_ms"] = 250;
    }
    const auto result =
        ex::execute_local_job(root, "job", value.dump(), worker());
    ASSERT_EQ(result.status_code, 1) << result.json;
    const auto attempt = Json::parse(result.json).at("attempts").at(0);
    const auto& native = attempt.at("result");
    EXPECT_EQ(native.at("status"),
              std::string_view(mode) == "flood" ? "output_limit" : "timed_out");
    EXPECT_EQ(native.at("process_tree_cleanup_confirmed"), true);
    ex::SQLiteRunStore store(root);
    if (std::string_view(mode) == "flood") {
      EXPECT_EQ(native.at("local_retention").at("stdout").at("captured_bytes"),
                32);
      EXPECT_GT(native.at("stdout_bytes").get<std::uint64_t>(), 32);
      EXPECT_EQ(captured(store, attempt, "stdout"), "");
    } else {
      EXPECT_EQ(captured(store, attempt, "stdout"), "native stdout\n");
      EXPECT_EQ(captured(store, attempt, "stderr"), "native stderr\n");
    }
    EXPECT_EQ(
        ex::control_local_job(ex::LocalJobControl::recover, root, "job").json,
        result.json);
  }
}

TEST_F(LocalJobSdkTest, InvalidLogRetentionRejectsBeforeStoreCreation) {
  auto value = log_spec();
  for (const Json invalid :
       {Json(nullptr), Json::object(),
        Json{{"stdout", "unknown"}, {"stderr", "always"}},
        Json{{"stdout", true}, {"stderr", "always"}},
        Json{{"stdout", "never"}, {"stderr", "never"}, {"purge", true}}}) {
    value["local_retention"] = invalid;
    const auto root = temporary / "invalid-retention";
    EXPECT_THROW(ex::execute_local_job(root, "job", value.dump(), worker()),
                 jyppx::liuding::Error);
    EXPECT_FALSE(fs::exists(root));
  }
}
}  // namespace
