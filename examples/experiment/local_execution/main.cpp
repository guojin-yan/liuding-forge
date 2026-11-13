// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/local_execution/json_lines.hpp>
#include <jyppx/liuding/experiment/local_execution/process.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#endif

namespace {
namespace exp = jyppx::liuding::experiment;
using namespace std::chrono_literals;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int child(int argc, char** argv) {
  require(argc == 3 && std::string(argv[2]) == "literal $() ; value",
          "Literal argument changed");
  require(std::getenv("LIUDING_EXAMPLE_VALUE") &&
              std::string(std::getenv("LIUDING_EXAMPLE_VALUE")) == "child-only",
          "Child environment changed");
  require(std::cin.get() == std::char_traits<char>::eof(),
          "Child stdin must be EOF");
  std::cout << "native child\n";
  std::cerr << "native diagnostic\n";
  return 23;
}

void example(const std::filesystem::path& executable) {
  exp::LocalProcessOptions options;
  options.executable = executable;
  options.working_directory = std::filesystem::current_path();
  options.arguments = {"--child", "literal $() ; value"};
  // Preserve the configured runtime lookup paths; overrides stay child-only.
  options.inherit_environment = true;
  options.environment = {{"LIUDING_EXAMPLE_VALUE", "child-only"}};
  options.timeout = 5s;
  options.max_stdout_bytes = options.max_stderr_bytes = 1024;
  const auto cpus = exp::local_process_available_cpu_ids();
  options.cpu_affinity = {cpus.front()};
  options.environment_sha256 = exp::local_process_environment_sha256(options);
  const auto result = exp::LocalProcessRunner{}.run(options);
  require(result.status == exp::LocalProcessStatus::exited &&
              result.exit_code == 23 && !result.signal &&
              result.stdout_capture == "native child\n" &&
              result.stderr_capture == "native diagnostic\n" &&
              result.process_tree_cleanup_confirmed &&
              result.cpu_affinity == options.cpu_affinity &&
              result.resource_usage.has_value(),
          "Native process result differs");

  auto cancelled = std::stop_source{};
  cancelled.request_stop();
  const auto stopped =
      exp::LocalProcessRunner{}.run(options, cancelled.get_token());
  require(stopped.status == exp::LocalProcessStatus::cancelled &&
              stopped.process_id == 0 && stopped.process_tree_cleanup_confirmed,
          "Pre-cancelled process started");

  options.arguments = {"--wait"};
  options.timeout = 150ms;
  const auto timeout = exp::LocalProcessRunner{}.run(options);
  require(timeout.status == exp::LocalProcessStatus::timed_out &&
              timeout.process_tree_cleanup_confirmed,
          "Timed-out process was not reaped");

  exp::JsonLineProcessOptions channel;
  channel.executable = executable;
  channel.working_directory = options.working_directory;
  channel.inherit_environment = true;
  channel.arguments = {"--json-lines"};
  channel.max_request_bytes = channel.max_reply_bytes = 1024;
  exp::JsonLineProcessTransport transport(channel);
  require(transport.transact("{\"request\":1}") == "{\"request\":1}",
          "First native reply differs");
  require(transport.transact("{\"request\":2}") == "{\"request\":2}" &&
              transport.stats().completed_requests == 2,
          "Second native reply differs");
  transport.terminate();
  require(transport.poisoned() && transport.process_tree_cleanup_confirmed(),
          "Native channel did not release ownership");
  std::cout << "{\"passed\":true,\"native_cpp\":true,\"ordinary_exit\":23,"
               "\"cpu_placement\":true,\"environment_binding\":true,"
               "\"cancel_before_launch\":true,\"timeout_cleanup\":true,"
               "\"json_line_replies\":2,\"training_updates\":0}\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
#if defined(_WIN32)
    require(_setmode(_fileno(stdout), _O_BINARY) != -1 &&
                _setmode(_fileno(stderr), _O_BINARY) != -1,
            "Binary output mode unavailable");
#endif
    if (argc > 1) {
      const std::string mode = argv[1];
      if (mode == "--child") return child(argc, argv);
      if (mode == "--wait") {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        return 0;
      }
      if (mode == "--json-lines") {
        for (std::string line; std::getline(std::cin, line);)
          std::cout << line << std::endl;
        return 0;
      }
      throw std::runtime_error("Unknown example mode");
    }
    example(std::filesystem::absolute(argv[0]));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
