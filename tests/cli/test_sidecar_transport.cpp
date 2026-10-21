// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "../../tools/cli/sidecar_process.hpp"

#include <jyppx/foundation/error.hpp>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// The registering CMake target should define these from Python3_EXECUTABLE and
// the absolute tests/cli/sidecar_fixture.py path. Environment fallback supports
// direct test invocation; absent prerequisites fail, rather than silently skip.
#ifndef JYPPX_SIDECAR_TEST_PYTHON
#define JYPPX_SIDECAR_TEST_PYTHON ""
#endif
#ifndef JYPPX_SIDECAR_TEST_FIXTURE
#define JYPPX_SIDECAR_TEST_FIXTURE ""
#endif

namespace {
namespace jy = jyppx;
namespace cli = jyppx::cli;
namespace fs = std::filesystem;
using Json = nlohmann::json;
using namespace std::chrono_literals;

std::string utf8(const fs::path& path) {
  const auto encoded = path.u8string();
  return {reinterpret_cast<const char*>(encoded.data()), encoded.size()};
}

fs::path configured_path(const char* compiled, const char* variable) {
  const auto* configured = *compiled ? compiled : std::getenv(variable);
  if (!configured || !*configured) {
    throw std::runtime_error(
        "Sidecar process tests require explicit Python and fixture absolute "
        "paths");
  }
  const fs::path path = fs::u8path(configured);
  if (!path.is_absolute() || !fs::is_regular_file(path)) {
    throw std::runtime_error(
        "Sidecar process test prerequisite is not an absolute regular file");
  }
  return path;
}

class ProcessDirectory {
 public:
  ProcessDirectory() {
    static std::atomic<unsigned> sequence{0U};
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    root =
        fs::temp_directory_path() / fs::path(u8"jyppx 子进程 space") /
        (std::to_string(stamp) + "-" + std::to_string(sequence.fetch_add(1U)));
    fs::create_directories(root.parent_path());
    if (!fs::create_directory(root))
      throw std::runtime_error("Unique subprocess test directory unavailable");
    const auto original = configured_path(JYPPX_SIDECAR_TEST_FIXTURE,
                                          "JYPPX_SIDECAR_TEST_FIXTURE");
    fixture = root / fs::path(u8"故障 fixture.py");
    fs::copy_file(original, fixture);
  }
  ~ProcessDirectory() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
  cli::SidecarProcessOptions options(const std::string& mode) const {
    cli::SidecarProcessOptions configured;
    configured.executable =
        configured_path(JYPPX_SIDECAR_TEST_PYTHON, "JYPPX_SIDECAR_TEST_PYTHON");
    configured.arguments = {"-I", "-u", utf8(fixture), "--mode", mode};
    configured.working_directory = root;
    configured.inherit_environment = true;
    configured.operation_timeout = 2000ms;
    configured.startup_timeout = 5000ms;
    configured.cleanup_timeout = 1000ms;
    return configured;
  }
  fs::path root;
  fs::path fixture;
};

bool await(const std::function<bool()>& predicate,
           std::chrono::milliseconds duration = 3000ms) {
  const auto deadline = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

std::optional<unsigned long> pid_file(const fs::path& path) {
  std::ifstream input(path);
  unsigned long pid = 0U;
  if (input >> pid && pid != 0U) return pid;
  return std::nullopt;
}

bool alive(unsigned long pid) {
#ifdef _WIN32
  HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(pid));
  if (!process) return false;
  const bool running = WaitForSingleObject(process, 0U) == WAIT_TIMEOUT;
  CloseHandle(process);
  return running;
#else
  if (::kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH) return false;
#if defined(__linux__)
  // Descendants belong to their actual parent/system reaper. A killed zombie
  // is no longer executing and must not turn this process-group gate flaky.
  std::ifstream input("/proc/" + std::to_string(pid) + "/stat");
  std::string line;
  if (std::getline(input, line)) {
    const auto name_end = line.rfind(')');
    if (name_end != std::string::npos && name_end + 2U < line.size() &&
        line[name_end + 2U] == 'Z')
      return false;
  }
#endif
  return true;
#endif
}

TEST(SidecarProcessTest, RealPersistentChildEchoesMultipleBoundedRequests) {
  ProcessDirectory directory;
  cli::SidecarProcessTransport transport(directory.options("echo"));
  for (unsigned index = 0U; index < 4U; ++index) {
    const auto request =
        Json{{"request_id", index}, {"text", "redacted"}}.dump();
    EXPECT_EQ(transport.transact(request), request);
  }
  EXPECT_FALSE(transport.poisoned());
  EXPECT_EQ(transport.stats().completed_requests, 4U);
  EXPECT_GT(transport.stats().stdout_bytes, 0U);
}

TEST(SidecarProcessTest, ArgvQuotingUnicodeCwdAndChildOnlyEnvironmentAreExact) {
  ProcessDirectory directory;
  auto options = directory.options("inspect");
  const std::vector<std::string> arguments{"",
                                           "with spaces",
                                           "a\"b",
                                           "trailing\\",
                                           "\\\"\\\\",
                                           "$(must-never-run) & | ; `literal`",
                                           "line\nbreak",
                                           utf8(fs::path(u8"参数/测试"))};
  options.arguments.push_back("--");
  options.arguments.insert(options.arguments.end(), arguments.begin(),
                           arguments.end());
  options.environment["JYPPX_CHILD_TOKEN"] = "child-only-token";
  const auto* original = std::getenv("JYPPX_CHILD_TOKEN");
  const std::optional<std::string> before =
      original ? std::optional<std::string>(original) : std::nullopt;
  const auto cwd = fs::current_path();
  cli::SidecarProcessTransport transport(options);
  const auto result = Json::parse(transport.transact("{}"));
  EXPECT_EQ(result.at("arguments").get<std::vector<std::string>>(), arguments);
  EXPECT_TRUE(fs::equivalent(fs::u8path(result.at("cwd").get<std::string>()),
                             directory.root));
  EXPECT_EQ(result.at("environment"), "child-only-token");
  EXPECT_EQ(fs::current_path(), cwd);
  const auto* after = std::getenv("JYPPX_CHILD_TOKEN");
  EXPECT_EQ(after ? std::optional<std::string>(after) : std::nullopt, before);
}

TEST(SidecarProcessTest,
     AbsoluteDeadlineAndDefaultTimeoutPoisonInsteadOfHanging) {
  ProcessDirectory directory;
  for (bool explicit_deadline : {false, true}) {
    auto options = directory.options("delay");
    options.operation_timeout = explicit_deadline ? 2000ms : 150ms;
    cli::SidecarProcessTransport transport(options);
    const jy::TelemetryDeadline deadline =
        explicit_deadline
            ? jy::TelemetryDeadline(std::chrono::steady_clock::now() + 150ms)
            : std::nullopt;
    const auto begin = std::chrono::steady_clock::now();
    EXPECT_THROW(transport.transact("{}", {}, deadline), jy::DeadlineExceeded);
    EXPECT_LT(std::chrono::steady_clock::now() - begin, 2500ms);
    EXPECT_TRUE(transport.poisoned());
    EXPECT_THROW(transport.transact("{}"), jy::NotReady);
  }
}

TEST(SidecarProcessTest, StopTokenInterruptsARealBlockedExchange) {
  ProcessDirectory directory;
  cli::SidecarProcessTransport transport(directory.options("delay"));
  std::atomic<jy::ErrorCode> error{jy::ErrorCode::internal};
  std::jthread exchange([&](std::stop_token token) {
    try {
      transport.transact("{}", token);
    } catch (const jy::Error& failure) {
      error.store(failure.code());
    }
  });
  std::this_thread::sleep_for(30ms);
  exchange.request_stop();
  exchange.join();
  EXPECT_EQ(error.load(), jy::ErrorCode::cancelled);
  EXPECT_TRUE(transport.poisoned());
}

TEST(SidecarProcessTest, BlockedStdinWriteHasTheSameAbsoluteDeadline) {
  ProcessDirectory directory;
  auto options = directory.options("no_read");
  options.max_request_bytes = 1024U * 1024U;
  cli::SidecarProcessTransport transport(options);
  const auto request = Json{{"large", std::string(512U * 1024U, 'x')}}.dump();
  EXPECT_THROW(
      transport.transact(request, {}, std::chrono::steady_clock::now() + 150ms),
      jy::DeadlineExceeded);
  EXPECT_TRUE(transport.poisoned());
}

TEST(SidecarProcessTest, EofPartialFrameCrashAndExtraLinesFailAndPoison) {
  ProcessDirectory directory;
  for (const auto* mode :
       {"eof", "partial", "crash", "extra", "invalid_utf8", "nul", "cr"}) {
    SCOPED_TRACE(mode);
    auto options = directory.options(mode);
    options.operation_timeout = 250ms;
    cli::SidecarProcessTransport transport(options);
    EXPECT_THROW(transport.transact("{}"), jy::Error);
    EXPECT_TRUE(transport.poisoned());
    EXPECT_EQ(transport.stats().completed_requests, 0U);
    EXPECT_THROW(transport.transact("{}"), jy::NotReady);
  }
}

TEST(SidecarProcessTest,
     ReplyAndStderrFloodsAreBoundedAndNeverEchoDiagnostics) {
  ProcessDirectory directory;
  for (const auto* mode : {"flood", "stderr_flood"}) {
    auto options = directory.options(mode);
    options.max_reply_bytes = 1024U;
    options.max_stderr_bytes = 1024U;
    cli::SidecarProcessTransport transport(options);
    try {
      transport.transact("{}");
      FAIL() << "Expected an explicit byte quota failure";
    } catch (const jy::ResourceExhausted& failure) {
      EXPECT_EQ(std::string(failure.what()).find("credential-do-not-echo"),
                std::string::npos);
      EXPECT_TRUE(transport.poisoned());
    }
  }
  cli::SidecarProcessTransport harmless(directory.options("stderr"));
  EXPECT_EQ(harmless.transact("{}"), "{}");
  EXPECT_GT(harmless.stats().stderr_bytes, 0U);
}

TEST(SidecarProcessTest, DelayedUnsolicitedOutputFailsBeforeTheNextRequest) {
  ProcessDirectory directory;
  auto options = directory.options("late_extra");
  const auto prefix = directory.root / "late";
  options.arguments.insert(options.arguments.end(),
                           {"--marker-prefix", utf8(prefix)});
  cli::SidecarProcessTransport transport(options);
  EXPECT_EQ(transport.transact("{}"), "{}");
  {
    std::ofstream release(fs::u8path(utf8(prefix) + ".late.release"));
    release << "release";
  }
  ASSERT_TRUE(await(
      [&] { return fs::exists(fs::u8path(utf8(prefix) + ".late.ready")); }));
  EXPECT_THROW(transport.transact("{}"), jy::SerializationError);
  EXPECT_TRUE(transport.poisoned());
}

TEST(SidecarProcessTest,
     InvalidRequestAndQuotaFailureRequireExplicitNewTransport) {
  ProcessDirectory directory;
  for (const auto& request :
       {std::string(""), std::string("{}\n{}"), std::string("{}\r"),
        std::string("\xff", 1U), std::string("{\0}", 3U)}) {
    cli::SidecarProcessTransport transport(directory.options("echo"));
    EXPECT_THROW(transport.transact(request), jy::InvalidArgument);
    EXPECT_TRUE(transport.poisoned());
  }
  auto options = directory.options("echo");
  options.max_request_bytes = 1U;
  cli::SidecarProcessTransport exceeded(options);
  EXPECT_THROW(exceeded.transact("{}"), jy::ResourceExhausted);
  EXPECT_TRUE(exceeded.poisoned());
  cli::SidecarProcessTransport restarted(directory.options("echo"));
  EXPECT_EQ(restarted.transact("{}"), "{}");
}

TEST(SidecarProcessTest, PreCancelledOrExpiredLaunchDoesNotCreateAChild) {
  ProcessDirectory directory;
  const auto options = directory.options("echo");
  std::stop_source source;
  source.request_stop();
  EXPECT_THROW(cli::SidecarProcessTransport(options, source.get_token()),
               jy::Cancelled);
  EXPECT_THROW(cli::SidecarProcessTransport(
                   options, {}, std::chrono::steady_clock::now() - 1ms),
               jy::DeadlineExceeded);
}

TEST(SidecarProcessTest,
     NonExecutableAndRelativePathsFailWithoutLeakingLaunchData) {
  ProcessDirectory directory;
  auto options = directory.options("echo");
  options.executable = "relative-python";
  EXPECT_THROW(cli::SidecarProcessTransport{options}, jy::InvalidArgument);
  options = directory.options("echo");
  options.executable = directory.root / "not-an-executable.bin";
  {
    std::ofstream invalid(options.executable, std::ios::binary);
    invalid << "invalid executable bytes";
  }
  EXPECT_THROW(cli::SidecarProcessTransport{options}, jy::BackendError);
  options = directory.options("echo");
  options.environment["INVALID=KEY"] = "secret-do-not-echo";
  try {
    cli::SidecarProcessTransport rejected(options);
    FAIL();
  } catch (const jy::InvalidArgument& failure) {
    EXPECT_EQ(std::string(failure.what()).find("secret-do-not-echo"),
              std::string::npos);
  }
}

TEST(SidecarProcessTest,
     CancellationTerminatesTheLeaderAndMultipleDescendantLevels) {
  ProcessDirectory directory;
  auto options = directory.options("child");
  options.operation_timeout = 5000ms;
  const auto prefix = directory.root / "descendant";
  options.arguments.insert(options.arguments.end(),
                           {"--marker-prefix", utf8(prefix), "--levels", "1"});
  cli::SidecarProcessTransport transport(options);
  std::atomic<jy::ErrorCode> error{jy::ErrorCode::internal};
  std::jthread exchange([&](std::stop_token token) {
    try {
      transport.transact("{}", token);
    } catch (const jy::Error& failure) {
      error.store(failure.code());
    }
  });
  const auto leaf_path = fs::u8path(utf8(prefix) + ".0.pid");
  const auto middle_path = fs::u8path(utf8(prefix) + ".1.pid");
  const auto parent_path = fs::u8path(utf8(prefix) + ".parent.pid");
  const bool ready = await([&] {
    return pid_file(leaf_path) && pid_file(middle_path) &&
           pid_file(parent_path);
  });
  exchange.request_stop();
  exchange.join();
  ASSERT_TRUE(ready);
  EXPECT_EQ(error.load(), jy::ErrorCode::cancelled);
  for (const auto& path : {parent_path, middle_path, leaf_path}) {
    const auto pid = pid_file(path);
    ASSERT_TRUE(pid.has_value());
    EXPECT_TRUE(await([&] { return !alive(*pid); }));
  }
}

TEST(SidecarProcessTest, DestructorTerminatesAHealthyChildAndItsDescendants) {
  ProcessDirectory directory;
  auto options = directory.options("child_reply");
  const auto prefix = directory.root / "owned";
  options.arguments.insert(options.arguments.end(),
                           {"--marker-prefix", utf8(prefix), "--levels", "0"});
  const auto leaf_path = fs::u8path(utf8(prefix) + ".0.pid");
  std::optional<unsigned long> child;
  {
    cli::SidecarProcessTransport transport(options);
    EXPECT_TRUE(
        Json::parse(transport.transact("{}")).at("spawned").get<bool>());
    ASSERT_TRUE(await([&] {
      child = pid_file(leaf_path);
      return child.has_value();
    }));
  }
  ASSERT_TRUE(child.has_value());
  EXPECT_TRUE(await([&] { return !alive(*child); }));
}

TEST(SidecarProcessTest, ArbitraryParentHandlesAreNotInherited) {
  ProcessDirectory directory;
  auto options = directory.options("handle");
#ifdef _WIN32
  SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  const auto name =
      "Local\\jyppx.sidecar.test." + std::to_string(GetCurrentProcessId()) +
      "." +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());
  const std::wstring wide_name(name.begin(), name.end());
  HANDLE handle = CreateEventW(&security, TRUE, FALSE, wide_name.c_str());
  ASSERT_NE(handle, nullptr);
  const auto number =
      static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(handle));
  options.arguments.insert(
      options.arguments.end(),
      {"--handle", std::to_string(number), "--object-name", name});
  try {
    cli::SidecarProcessTransport transport(options);
    EXPECT_FALSE(
        Json::parse(transport.transact("{}")).at("inherited").get<bool>());
  } catch (...) {
    CloseHandle(handle);
    throw;
  }
  CloseHandle(handle);
#else
  const auto parent_file = directory.root / "parent-only-handle.bin";
  {
    std::ofstream output(parent_file);
    output << "parent-owned";
  }
  const auto descriptor = ::open(parent_file.c_str(), O_RDONLY);
  ASSERT_GE(descriptor, 0);
  ASSERT_EQ(fcntl(descriptor, F_SETFD, 0), 0);
  struct stat identity {};
  ASSERT_EQ(fstat(descriptor, &identity), 0);
  const auto file_identity =
      std::to_string(identity.st_dev) + ":" + std::to_string(identity.st_ino);
  options.arguments.insert(options.arguments.end(),
                           {"--handle", std::to_string(descriptor),
                            "--file-identity", file_identity});
  try {
    cli::SidecarProcessTransport transport(options);
    EXPECT_FALSE(
        Json::parse(transport.transact("{}")).at("inherited").get<bool>());
  } catch (...) {
    ::close(descriptor);
    throw;
  }
  ::close(descriptor);
#endif
}

#ifdef _WIN32
TEST(SidecarProcessTest, JobObjectContainsExplicitBreakawayDescendants) {
  ProcessDirectory directory;
  std::optional<unsigned long> child;
  {
    cli::SidecarProcessTransport transport(directory.options("breakaway"));
    const auto reply = Json::parse(transport.transact("{}"));
    if (!reply.at("blocked").get<bool>()) {
      child = reply.at("child_pid").get<unsigned long>();
      ASSERT_NE(*child, 0U);
      EXPECT_TRUE(alive(*child));
    }
  }
  if (child) {
    const bool terminated = await([&] { return !alive(*child); });
    // Clean up even when containment fails so a failed gate cannot leak work.
    if (!terminated) {
      HANDLE escaped = OpenProcess(PROCESS_TERMINATE, FALSE, *child);
      if (escaped) {
        TerminateProcess(escaped, 1U);
        CloseHandle(escaped);
      }
    }
    EXPECT_TRUE(terminated);
  }
}
#endif

}  // namespace
