// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <signal.h>
#endif

#include "../../tools/cli/local_process.hpp"

#ifdef _WIN32
#include "windows_handle_diagnostics.hpp"
#endif

#include <jyppx/liuding/foundation/error.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace cli = jyppx::liuding::cli;
namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace std::chrono_literals;

std::string utf8(const fs::path& path) {
  const auto encoded = path.u8string();
  return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

bool await(const std::function<bool()>& predicate,
           std::chrono::milliseconds timeout = 5000ms) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

bool alive(std::uint64_t pid) {
  if (!pid) return false;
#ifdef _WIN32
  const auto process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
  if (!process) return false;
  const auto state = WaitForSingleObject(process, 0U);
  CloseHandle(process);
  return state == WAIT_TIMEOUT;
#else
  if (::kill(static_cast<pid_t>(pid), 0) < 0 && errno == ESRCH) return false;
#if defined(__linux__)
  std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (std::getline(stat, line)) {
    const auto position = line.rfind(')');
    if (position != std::string::npos && position + 2U < line.size() &&
        line[position + 2U] == 'Z')
      return false;
  }
#endif
  return true;
#endif
}

class ProcessDirectory {
 public:
  ProcessDirectory() {
    static std::atomic<unsigned> sequence{0U};
    root = fs::temp_directory_path() / fs::path(u8"liuding 本地 process") /
           (std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) +
            "-" + std::to_string(sequence.fetch_add(1U)));
    fs::create_directories(root.parent_path());
    if (!fs::create_directory(root))
      throw std::runtime_error("Unique cwd unavailable");
    fixture = root / fs::path(u8"进程 fixture.py");
    fs::copy_file(fs::u8path(JYPPX_LOCAL_PROCESS_FIXTURE), fixture);
  }
  ~ProcessDirectory() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  cli::LocalProcessOptions options(const std::string& mode) const {
    cli::LocalProcessOptions result;
    result.executable = fs::u8path(JYPPX_LOCAL_PROCESS_PYTHON);
    result.arguments = {"-I", "-u", utf8(fixture), mode, utf8(root)};
    result.working_directory = root;
    result.inherit_environment = true;
    result.timeout = 15000ms;
    result.cleanup_timeout = 2000ms;
    return result;
  }
  void expect_tree_stopped(std::uint64_t leader) const {
    EXPECT_FALSE(alive(leader));
    for (const auto* name : {"descendant.pid", "grandchild.pid"}) {
      std::ifstream file(root / name);
      std::uint64_t pid = 0;
      ASSERT_TRUE(static_cast<bool>(file >> pid)) << name;
      EXPECT_TRUE(await([&] { return !alive(pid); })) << name << ':' << pid;
    }
  }
  fs::path root;
  fs::path fixture;
};

TEST(LocalProcessTest, PreservesLiteralArgvUnicodeCwdEnvironmentAndStdinEof) {
  ProcessDirectory directory;
  auto options = directory.options("inspect");
  const std::vector<std::string> arguments{"",
                                           "with spaces",
                                           "a\"b",
                                           "tail\\",
                                           "\\\"\\\\",
                                           "literal & | ; $(x) `x`",
                                           "line\nbreak",
                                           utf8(fs::path(u8"参数/值"))};
  options.arguments.insert(options.arguments.end(), arguments.begin(),
                           arguments.end());
  options.environment["JYPPX_LOCAL_PROCESS_TEST_TOKEN"] = "child-only";
  const auto* original = std::getenv("JYPPX_LOCAL_PROCESS_TEST_TOKEN");
  const std::string before = original ? original : "";
  const auto cwd = fs::current_path();
  const auto result = cli::LocalProcessRunner{}.run(options);
  ASSERT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 0U);
  EXPECT_FALSE(result.signal);
  EXPECT_TRUE(result.process_tree_cleanup_confirmed);
  const auto payload = Json::parse(result.stdout_capture);
  EXPECT_EQ(payload.at("arguments").get<std::vector<std::string>>(), arguments);
  EXPECT_EQ(fs::u8path(payload.at("cwd").get<std::string>()), directory.root);
  EXPECT_EQ(payload.at("token"), "child-only");
  EXPECT_EQ(payload.at("stdin"), "");
  EXPECT_EQ(fs::current_path(), cwd);
  const auto* after = std::getenv("JYPPX_LOCAL_PROCESS_TEST_TOKEN");
  EXPECT_EQ(after ? after : "", before);
  EXPECT_FALSE(alive(result.process_id));
}

TEST(LocalProcessTest, NonzeroExitPreservesRawBinaryOutputAndNativeCode) {
  ProcessDirectory directory;
  auto options = directory.options("exit");
  options.arguments.push_back("37");
  const auto result = cli::LocalProcessRunner{}.run(options);
  EXPECT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 37U);
  EXPECT_TRUE(result.process_tree_cleanup_confirmed);
  EXPECT_EQ(result.stdout_capture, std::string("ordinary-output\0\xff", 17));
  EXPECT_EQ(result.stderr_capture, std::string("ordinary-error\0\xfe", 16));
  EXPECT_EQ(result.stdout_bytes, result.stdout_capture.size());
  EXPECT_EQ(result.stderr_bytes, result.stderr_capture.size());
  EXPECT_FALSE(alive(result.process_id));
}

TEST(LocalProcessTest, DrainsBothLargeStreamsAndFinalTailWithoutDeadlock) {
  ProcessDirectory directory;
  const auto result =
      cli::LocalProcessRunner{}.run(directory.options("volume"));
  ASSERT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 0U);
  EXPECT_EQ(result.stdout_capture, std::string(110000, 'o') + "last-stdout");
  EXPECT_EQ(result.stderr_capture, std::string(90000, 'e') + "last-stderr");
}

TEST(LocalProcessTest,
     LargeFinalStreamsDrainAfterLeaderExitWithLiveDescendants) {
  ProcessDirectory directory;
  auto options = directory.options("volume-leader-exits");
  options.max_stdout_bytes = 3U * 1024U * 1024U;
  options.max_stderr_bytes = 3U * 1024U * 1024U;
  const auto result = cli::LocalProcessRunner{}.run(options);
  ASSERT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 23U);
  EXPECT_EQ(result.stdout_capture,
            std::string(2U * 1024U * 1024U + 29U, 'o') + "final-stdout");
  EXPECT_EQ(result.stderr_capture,
            std::string(2U * 1024U * 1024U + 43U, 'e') + "final-stderr");
  EXPECT_EQ(result.stdout_bytes, result.stdout_capture.size());
  EXPECT_EQ(result.stderr_bytes, result.stderr_capture.size());
  directory.expect_tree_stopped(result.process_id);
}

TEST(LocalProcessTest, ClosedStdoutDoesNotDiscardLaterStderrOrExit) {
  ProcessDirectory directory;
  const auto result =
      cli::LocalProcessRunner{}.run(directory.options("stdout-eof"));
  EXPECT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 0U);
  EXPECT_TRUE(result.stdout_capture.empty());
  EXPECT_EQ(result.stderr_capture, "stderr-after-stdout-eof");
}

TEST(LocalProcessTest,
     DoesNotInheritParentEnvironmentUnlessExplicitlySelected) {
  ProcessDirectory directory;
  auto options = directory.options("inspect");
  options.inherit_environment = false;
  options.environment["JYPPX_LOCAL_PROCESS_TEST_TOKEN"] = "explicit";
  ASSERT_NE(std::getenv("PATH"), nullptr);
  const auto result = cli::LocalProcessRunner{}.run(options);
  ASSERT_EQ(result.exit_code, 0U);
  EXPECT_EQ(Json::parse(result.stdout_capture).at("token"), "explicit");
  EXPECT_TRUE(Json::parse(result.stdout_capture).at("parent_path").is_null());
}

TEST(LocalProcessTest, PrecancelledInvocationDoesNotLaunch) {
  ProcessDirectory directory;
  std::stop_source stop;
  stop.request_stop();
  const auto result = cli::LocalProcessRunner{}.run(directory.options("sleep"),
                                                    stop.get_token());
  EXPECT_EQ(result.status, cli::LocalProcessStatus::cancelled);
  EXPECT_EQ(result.process_id, 0U);
  EXPECT_TRUE(result.process_tree_cleanup_confirmed);
  EXPECT_FALSE(result.exit_code);
  EXPECT_FALSE(fs::exists(directory.root / "ready"));
}

TEST(LocalProcessTest, CancellationKillsLeaderDescendantAndGrandchild) {
  ProcessDirectory directory;
  std::stop_source stop;
  std::atomic<bool> ready{false};
  std::jthread canceller([&] {
    ready =
        await([&] { return fs::is_regular_file(directory.root / "ready"); });
    stop.request_stop();
  });
  const auto result = cli::LocalProcessRunner{}.run(directory.options("tree"),
                                                    stop.get_token());
  canceller.join();
  ASSERT_TRUE(ready);
  EXPECT_EQ(result.status, cli::LocalProcessStatus::cancelled);
  EXPECT_FALSE(result.exit_code);
  EXPECT_FALSE(result.signal);
  directory.expect_tree_stopped(result.process_id);
}

TEST(LocalProcessTest, TimeoutKillsLeaderDescendantAndGrandchild) {
  ProcessDirectory directory;
  auto options = directory.options("tree");
  options.timeout = 3000ms;
  const auto result = cli::LocalProcessRunner{}.run(options);
  EXPECT_EQ(result.status, cli::LocalProcessStatus::timed_out);
  EXPECT_FALSE(result.exit_code);
  EXPECT_GE(result.elapsed, options.timeout);
  EXPECT_LT(result.elapsed, options.timeout + options.cleanup_timeout + 3000ms);
  directory.expect_tree_stopped(result.process_id);
}

TEST(LocalProcessTest, OrdinaryLeaderExitAlsoStopsRemainingDescendants) {
  ProcessDirectory directory;
  const auto result =
      cli::LocalProcessRunner{}.run(directory.options("leader-exits"));
  EXPECT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 23U);
  EXPECT_EQ(result.stdout_capture, "leader-done");
  directory.expect_tree_stopped(result.process_id);
}

TEST(LocalProcessTest, OutputQuotaKillsEntireTreeAndBoundsCapture) {
  ProcessDirectory directory;
  auto options = directory.options("tree-flood");
  options.max_stdout_bytes = 513;
  const auto result = cli::LocalProcessRunner{}.run(options);
  EXPECT_EQ(result.status, cli::LocalProcessStatus::output_limit);
  EXPECT_FALSE(result.exit_code);
  EXPECT_EQ(result.stdout_capture, std::string(513, 'x'));
  EXPECT_GT(result.stdout_bytes, 513U);
  EXPECT_LE(result.stdout_bytes, 513U + 4096U);
  directory.expect_tree_stopped(result.process_id);
}

TEST(LocalProcessTest, StderrHasIndependentCumulativeQuota) {
  ProcessDirectory directory;
  auto options = directory.options("flood");
  options.arguments.push_back("2");
  options.max_stderr_bytes = 129;
  const auto result = cli::LocalProcessRunner{}.run(options);
  EXPECT_EQ(result.status, cli::LocalProcessStatus::output_limit);
  EXPECT_EQ(result.stderr_capture, std::string(129, 'x'));
  EXPECT_TRUE(result.stdout_capture.empty());
  EXPECT_FALSE(alive(result.process_id));
}

TEST(LocalProcessTest, InvalidPathsTextAndBoundsNeverExecute) {
  ProcessDirectory directory;
  const auto valid = directory.options("sleep");
  auto rejected = [&](auto mutate) {
    auto options = valid;
    mutate(options);
    EXPECT_THROW(cli::LocalProcessRunner{}.run(options),
                 jyppx::liuding::InvalidArgument);
    EXPECT_FALSE(fs::exists(directory.root / "ready"));
  };
  rejected([](auto& o) { o.executable = "python"; });
  rejected([&](auto& o) { o.executable = directory.root / "missing.exe"; });
  rejected([&](auto& o) { o.working_directory = directory.root / "missing"; });
  rejected([](auto& o) { o.timeout = 0ms; });
  rejected([](auto& o) { o.timeout = std::chrono::hours(24 * 8); });
  rejected([](auto& o) { o.cleanup_timeout = 11000ms; });
  rejected([](auto& o) { o.startup_timeout = 0ms; });
  rejected([](auto& o) { o.max_stdout_bytes = 0; });
  rejected([](auto& o) { o.environment["bad=key"] = "value"; });
  rejected([](auto& o) { o.arguments.push_back(std::string("nul\0byte", 8)); });
  rejected([](auto& o) { o.arguments.push_back(std::string("\xff", 1)); });
}

TEST(LocalProcessTest, LaunchQuotaFailsBeforeChildStarts) {
  ProcessDirectory directory;
  auto options = directory.options("sleep");
  options.max_launch_bytes = 32;
  EXPECT_THROW(cli::LocalProcessRunner{}.run(options),
               jyppx::liuding::ResourceExhausted);
  EXPECT_FALSE(fs::exists(directory.root / "ready"));
}

TEST(LocalProcessTest, ExistingNonexecutableFailsWithLaunchError) {
  ProcessDirectory directory;
  auto options = directory.options("sleep");
  options.executable = directory.root / "not-native.bin";
  std::ofstream(options.executable, std::ios::binary) << "not an executable";
  EXPECT_THROW(cli::LocalProcessRunner{}.run(options),
               jyppx::liuding::BackendError);
  EXPECT_FALSE(fs::exists(directory.root / "ready"));
}

TEST(LocalProcessTest, RunnerCanExecuteAgainAfterCompletedAndCancelledCalls) {
  ProcessDirectory directory;
  cli::LocalProcessRunner runner;
  std::stop_source stop;
  stop.request_stop();
  EXPECT_EQ(runner.run(directory.options("inspect"), stop.get_token()).status,
            cli::LocalProcessStatus::cancelled);
  for (int i = 0; i < 3; ++i) {
    const auto result = runner.run(directory.options("inspect"));
    EXPECT_EQ(result.status, cli::LocalProcessStatus::exited);
    EXPECT_EQ(result.exit_code, 0U);
    EXPECT_FALSE(alive(result.process_id));
  }
}

#ifdef _WIN32
TEST(LocalProcessTest, RepeatedCompletionReleasesProcessJobAndPipeHandles) {
  ProcessDirectory directory;
  auto options = directory.options("exit");
  options.arguments.push_back("37");
  ASSERT_EQ(cli::LocalProcessRunner{}.run(options).exit_code, 37U);
  jyppx::liuding::test::WindowsHandleSnapshot before_framework;
  ASSERT_NO_THROW(before_framework =
                      jyppx::liuding::test::CaptureWindowsSelfHandles());
  {
    // Measure GoogleTest's first trace-stack/ThreadLocal initialization apart
    // from product operations. This performs no additional process warmup.
    SCOPED_TRACE("initialize the measurement framework before its baseline");
  }
  jyppx::liuding::test::WindowsHandleSnapshot baseline_metadata;
  ASSERT_NO_THROW(baseline_metadata =
                      jyppx::liuding::test::CaptureWindowsSelfHandles());
  const DWORD baseline = baseline_metadata.count;
  std::vector<jyppx::liuding::test::WindowsHandleSnapshot> round_metadata;
  round_metadata.reserve(8U);
  for (unsigned index = 0U; index < 8U; ++index) {
    SCOPED_TRACE(index);
    const auto result = cli::LocalProcessRunner{}.run(options);
    EXPECT_EQ(result.status, cli::LocalProcessStatus::exited);
    EXPECT_EQ(result.exit_code, 37U);
    EXPECT_FALSE(alive(result.process_id));
    jyppx::liuding::test::WindowsHandleSnapshot current;
    ASSERT_NO_THROW(current =
                        jyppx::liuding::test::CaptureWindowsSelfHandles());
    const DWORD observed = current.count;
    EXPECT_EQ(observed, baseline)
        << jyppx::liuding::test::WindowsHandleDifference(baseline_metadata,
                                                         current);
    round_metadata.push_back(std::move(current));
  }
  // Print after all measurements, so output/reporting cannot initialize a
  // handle between the baseline and a successful measured product operation.
  std::cout << "GoogleTest trace initialization: "
            << jyppx::liuding::test::WindowsHandleDifference(before_framework,
                                                             baseline_metadata)
            << '\n';
  if (::testing::Test::HasFailure())
    for (std::size_t index = 0; index < round_metadata.size(); ++index)
      std::cout << "handle round=" << index << ' '
                << jyppx::liuding::test::WindowsHandleDifference(
                       baseline_metadata, round_metadata[index])
                << '\n';
}

TEST(LocalProcessTest, ExitCode259IsAnExitRatherThanStillRunning) {
  ProcessDirectory directory;
  auto options = directory.options("exit");
  options.arguments.push_back("259");
  const auto result = cli::LocalProcessRunner{}.run(options);
  EXPECT_EQ(result.status, cli::LocalProcessStatus::exited);
  EXPECT_EQ(result.exit_code, 259U);
}
#else
TEST(LocalProcessTest, SignalTerminationRetainsNativeSignalWithoutExitCode) {
  ProcessDirectory directory;
  const auto result =
      cli::LocalProcessRunner{}.run(directory.options("signal"));
  EXPECT_EQ(result.status, cli::LocalProcessStatus::signalled);
  EXPECT_FALSE(result.exit_code);
  EXPECT_EQ(result.signal, SIGTERM);
}
#endif

}  // namespace
