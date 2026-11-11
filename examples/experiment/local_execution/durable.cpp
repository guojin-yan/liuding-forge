// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/local_execution/process.hpp>
#include <jyppx/liuding/experiment/local_execution/runner.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <cstdio>
#include <fcntl.h>
#include <io.h>
#endif

namespace {
namespace ex = jyppx::liuding::experiment;
namespace fs = std::filesystem;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
std::string quote(std::string_view text) {
  std::string result = "\"";
  constexpr char digits[] = "0123456789abcdef";
  for (const auto raw : text) {
    const auto c = static_cast<unsigned char>(raw);
    if (c == '"' || c == '\\') {
      result += '\\';
      result += raw;
    } else if (c < 32) {
      result += "\\u00";
      result += digits[c >> 4];
      result += digits[c & 15];
    } else
      result += raw;
  }
  return result + '"';
}
std::string path_text(const fs::path& path) {
  const auto value = path.generic_u8string();
  return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::string digest_file(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "Cannot open executable");
  jyppx::liuding::Sha256 digest;
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    digest.update(
        bytes({buffer.data(), static_cast<std::size_t>(input.gcount())}));
  }
  require(input.eof(), "Cannot hash executable");
  return jyppx::liuding::sha256_digest_hex(digest.finish());
}
std::string environment(const char* name) {
  const auto* value = std::getenv(name);
  require(value != nullptr, "Missing worker context");
  return value;
}
int task() {
  if (environment("JYPPX_LIUDING_LOCAL_ATTEMPT_INDEX") == "1") return 75;
  ex::SQLiteRunStore store(
      fs::u8path(environment("JYPPX_LIUDING_LOCAL_STORE_ROOT")));
  const auto id = environment("JYPPX_LIUDING_LOCAL_ATTEMPT_ID");
  const auto owner = store.find(id);
  require(owner.has_value(), "Missing owned attempt");
  const std::string payload = "native C++ result\n";
  store.put_artifact(id, "native.result", ex::RunArtifactKind::artifact,
                     "fixture", bytes(payload),
                     {.logical_name = "result.txt",
                      .media_type = "text/plain",
                      .producer = "native-sdk-example",
                      .expected_size_bytes = payload.size()},
                     owner->created_at_unix_ms);
  return 0;
}
int replay(const fs::path& root, const fs::path& spec) {
  std::ifstream input(spec, std::ios::binary);
  const std::string text{std::istreambuf_iterator<char>(input), {}};
  const auto result = ex::execute_local_job(root, "example", text, {});
  require(result.status_code == 0, "Confirmed terminal replay failed");
  require(ex::control_local_job(ex::LocalJobControl::recover, root, "example")
                  .json == result.json,
          "Terminal recovery differs");
  std::cout << result.json << '\n';
  return 0;
}
void example(const fs::path& executable, const fs::path& root) {
  require(!fs::exists(root), "Use an unused output directory");
  const auto sha = digest_file(executable);
  ex::SQLiteRunStore store(root);
  const auto stamp = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  store.start({.run_id = "inputs",
               .project_id = "demo",
               .experiment_id = "native-sdk",
               .display_name = "Native SDK inputs",
               .status = ex::RunStatus::running,
               .created_at_unix_ms = stamp,
               .updated_at_unix_ms = stamp,
               .lineage = {},
               .tags = {}});
  const auto pin = [&](const std::string& name) {
    const std::string value = "{}";
    const auto ref = store.put_artifact(
        "inputs", name, ex::RunArtifactKind::artifact, name, bytes(value),
        {.logical_name = name + ".json",
         .media_type = "application/json",
         .producer = "native-sdk-example",
         .expected_size_bytes = value.size()},
        stamp);
    return "{\"run_id\":\"inputs\",\"reference_id\":" + quote(name) +
           ",\"sha256\":" + quote(ref.artifact.digest.hex) +
           ",\"size_bytes\":2}";
  };
  const auto configuration = pin("configuration"), data = pin("data_manifest");
  store.transition("inputs", ex::RunStatus::completed, stamp + 1);
  ex::validate_local_training_source_json(store, "demo", configuration,
                                          ex::RunArtifactKind::artifact);
  const std::string specification =
      "{\"format\":\"jyppx.liuding.local_run_spec\",\"schema_version\":1,"
      "\"project_id\":\"demo\",\"experiment_id\":\"native-sdk\","
      "\"executable\":" +
      quote(path_text(executable)) + ",\"executable_sha256\":" + quote(sha) +
      ",\"arguments\":[\"--task\"],\"working_directory\":" +
      quote(path_text(root.parent_path())) +
      ",\"environment\":{},\"inherit_environment\":true,\"timeout_ms\":5000,"
      "\"max_attempts\":2,\"retry_exit_codes\":[75],\"training\":{\"model_id\":"
      "\"native-fixture\",\"configuration\":" +
      configuration + ",\"data_manifest\":" + data +
      ",\"resume\":null,\"outputs\":[{\"reference_id\":\"native.result\","
      "\"kind\":\"artifact\",\"role\":\"fixture\",\"max_size_bytes\":1024}]}}";
  require(ex::local_execution_spec_sha256(specification, true).size() == 64,
          "Invalid frozen identity");
  ex::LocalJobWorkerLaunch worker{executable, sha, {"--worker"}};
  const auto result =
      ex::execute_local_job(root, "example", specification, worker);
  require(result.status_code == 0, "Native durable execution failed");
  const auto prefix =
      "local-" + jyppx::liuding::sha256_hex(bytes("example")) + ".attempt-";
  const auto first = store.find(prefix + "1"),
             second = store.find(prefix + "2");
  require(first && second && first->status == ex::RunStatus::failed &&
              second->status == ex::RunStatus::completed,
          "Finite retry lifecycle differs");
  require(store.find_reference(prefix + "2", "native.result").has_value() &&
              store.find_reference(prefix + "2", "local.training.outputs")
                  .has_value(),
          "Typed output not confirmed");
  const auto spec_path = root / "example-spec.json";
  {
    std::ofstream output(spec_path, std::ios::binary);
    output << specification;
    require(static_cast<bool>(output), "Cannot save example spec");
  }
  ex::LocalProcessOptions reopen;
  reopen.executable = executable;
  reopen.arguments = {"--replay", path_text(root), path_text(spec_path)};
  reopen.working_directory = root;
  reopen.inherit_environment = true;
  reopen.timeout = std::chrono::seconds(10);
  const auto reopened = ex::LocalProcessRunner{}.run(reopen);
  require(reopened.exit_code == 0 && reopened.process_tree_cleanup_confirmed &&
              reopened.stdout_capture == result.json + "\n",
          "Fresh-process replay differs");
  std::cout << "{\"passed\":true,\"native_cpp\":true,\"attempts\":2,\"typed_"
               "output_confirmed\":true,\"fresh_process_replay\":true,\"CLI_"
               "dependencies\":0,\"training_updates\":0}\n";
}
}  // namespace
int main(int argc, char** argv) {
  try {
#if defined(_WIN32)
    require(_setmode(_fileno(stdout), _O_BINARY) != -1 &&
                _setmode(_fileno(stderr), _O_BINARY) != -1,
            "Binary output mode unavailable");
#endif
    if (argc == 10 && std::string(argv[1]) == "--worker" &&
        std::string(argv[2]) == "--root" && std::string(argv[4]) == "--id" &&
        std::string(argv[6]) == "--attempt" &&
        std::string(argv[8]) == "--intent-sha256") {
      const auto result = ex::run_local_job_worker(
          fs::u8path(argv[3]), argv[5], std::stoull(argv[7]), argv[9]);
      std::cout << result.json << '\n';
      return result.status_code;
    }
    if (argc == 2 && std::string(argv[1]) == "--task") return task();
    if (argc > 1 && std::string(argv[1]) == "--reject-worker") return 9;
    if (argc == 2 && std::string(argv[1]) == "--wait") {
      std::this_thread::sleep_for(std::chrono::seconds(30));
      return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--replay")
      return replay(fs::u8path(argv[2]), fs::u8path(argv[3]));
    require(argc == 2, "Usage: durable-example UNUSED_OUTPUT_DIRECTORY");
    example(fs::absolute(fs::u8path(argv[0])),
            fs::absolute(fs::u8path(argv[1])));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
