// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/experiment/local_execution/runner.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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
std::span<const std::byte> bytes(std::string_view value) {
  return std::as_bytes(std::span(value.data(), value.size()));
}
std::string digest(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "Cannot open worker executable");
  jyppx::liuding::Sha256 hash;
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    hash.update(
        bytes({buffer.data(), static_cast<std::size_t>(input.gcount())}));
  }
  require(input.eof(), "Cannot hash worker executable");
  return jyppx::liuding::sha256_digest_hex(hash.finish());
}
std::string specification(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "Cannot open specification");
  std::array<char, 65537> buffer{};
  input.read(buffer.data(), buffer.size());
  const auto count = static_cast<std::size_t>(input.gcount());
  require(input.eof() && count <= 65536,
          "Specification exceeds 64 KiB or cannot be read");
  return {buffer.data(), count};
}
std::uint64_t attempt(std::string_view text) {
  std::uint64_t value = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value);
  require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
              value != 0 && value <= 16,
          "Worker requires attempt 1 through 16");
  return value;
}
ex::LocalJobWorkerLaunch worker(const fs::path& executable) {
  return {executable, digest(executable), {"--worker"}};
}
}  // namespace

int main(int argc, char** argv) {
  try {
#if defined(_WIN32)
    require(_setmode(_fileno(stdout), _O_BINARY) != -1 &&
                _setmode(_fileno(stderr), _O_BINARY) != -1,
            "Binary output mode unavailable");
#endif
    if (argc == 3 && std::string_view(argv[1]) == "environment") {
      const auto value = ex::local_execution_environment_sha256(
          specification(fs::u8path(argv[2])));
      std::cout << "{\"environment_sha256\":\"" << value << "\"}\n";
      return 0;
    }
    ex::LocalJobResult result;
    if (argc == 10 && std::string_view(argv[1]) == "--worker" &&
        std::string_view(argv[2]) == "--root" &&
        std::string_view(argv[4]) == "--id" &&
        std::string_view(argv[6]) == "--attempt" &&
        std::string_view(argv[8]) == "--intent-sha256") {
      result = ex::run_local_job_worker(fs::u8path(argv[3]), argv[5],
                                        attempt(argv[7]), argv[9]);
    } else {
      require(argc == 4 || argc == 5,
              "Usage: local-job execute STORE ID SPEC | show/recover/cancel "
              "STORE ID");
      const std::string_view action(argv[1]);
      const auto root = fs::absolute(fs::u8path(argv[2]));
      const auto executable = fs::canonical(fs::u8path(argv[0]));
      if (action == "execute" && argc == 5) {
        result = ex::execute_local_job(root, argv[3],
                                       specification(fs::u8path(argv[4])),
                                       worker(executable));
      } else {
        require(argc == 4 && (action == "show" || action == "recover" ||
                              action == "cancel"),
                "Control requires show/recover/cancel STORE ID");
        const auto control = action == "show" ? ex::LocalJobControl::show
                             : action == "recover"
                                 ? ex::LocalJobControl::recover
                                 : ex::LocalJobControl::cancel;
        result = ex::control_local_job(control, root, argv[3],
                                       action == "recover"
                                           ? worker(executable)
                                           : ex::LocalJobWorkerLaunch{});
      }
    }
    std::cout << result.json << '\n';
    return result.status_code;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
