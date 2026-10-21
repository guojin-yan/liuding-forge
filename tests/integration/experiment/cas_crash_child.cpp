// Copyright 2026 JYPPX Training Framework Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

// Link against the static experiment archive. This executable supplies only
// its private forwarding symbol; the native publication remains product code.
#include "jyppx/experiment/detail/artifact_publish.hpp"

#include <jyppx/experiment/artifact_store.hpp>
#include <jyppx/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
namespace ex = jyppx::experiment;
using Json = nlohmann::json;
constexpr auto kConfirmed =
    "Confirmed CAS object predating the interrupted upload";
constexpr std::size_t kTargetBytes = 2U * 1024U * 1024U + 257U;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

std::string utf8(const std::filesystem::path& path) {
  const auto text = path.u8string();
  return {reinterpret_cast<const char*>(text.data()), text.size()};
}

std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

std::vector<std::byte> target_content() {
  std::vector<std::byte> content(kTargetBytes);
  for (std::size_t index = 0U; index < content.size(); ++index)
    content[index] = static_cast<std::byte>((index * 37U + 11U) & 255U);
  return content;
}

jyppx::ArtifactDigest confirmed_digest() {
  return {.hex = jyppx::sha256_hex(bytes(kConfirmed))};
}

jyppx::ArtifactDigest target_digest() {
  const auto content = target_content();
  return {.hex = jyppx::sha256_hex(content)};
}

std::filesystem::path object_path(const std::filesystem::path& root,
                                  const jyppx::ArtifactDigest& digest) {
  return root / "cas" / "objects" / digest.hex.substr(0U, 2U) /
         digest.hex.substr(2U);
}

std::uint64_t process_id() {
#ifdef _WIN32
  return static_cast<std::uint64_t>(::_getpid());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

void report(const Json& value) {
  const auto encoded = value.dump();
  require(encoded.size() <= 64U * 1024U, "CAS report exceeded bound");
  std::cout << encoded << '\n' << std::flush;
  require(static_cast<bool>(std::cout), "could not publish CAS report");
}

Json file_evidence(const std::filesystem::path& path) {
  Json result{{"path", utf8(path)}, {"exists", std::filesystem::exists(path)}};
  if (!result.at("exists").get<bool>()) return result;
  const auto size = std::filesystem::file_size(path);
  require(size <= 4U * 1024U * 1024U, "fixture file exceeded its bound");
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "could not read CAS fixture file");
  jyppx::Sha256 hash;
  std::array<std::byte, 8192U> buffer{};
  std::uint64_t read = 0U;
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      hash.update(std::span<const std::byte>(buffer.data(),
                                             static_cast<std::size_t>(count)));
      read += static_cast<std::uint64_t>(count);
    }
  }
  require(input.eof() && read == size, "CAS fixture changed while reading");
  result["bytes"] = size;
  result["sha256"] = jyppx::sha256_digest_hex(hash.finish());
  return result;
}

struct PublicationState {
  bool armed = false;
  std::string boundary;
  std::filesystem::path root;
  std::filesystem::path source;
  std::filesystem::path destination;
  ex::ArtifactUploadSession* session = nullptr;
  unsigned native_calls = 0U;
};

PublicationState publication;

[[noreturn]] void kill_fence(std::string_view boundary) {
  require(publication.session != nullptr,
          "CAS fence requires an active upload");
  report({{"schema_version", 1U},
          {"phase", "ready-for-forced-kill"},
          {"process_id", process_id()},
          {"boundary", boundary},
          {"private_seam_observed", true},
          {"native_calls", publication.native_calls},
          {"session_closed", publication.session->closed()},
          {"source", file_evidence(publication.source)},
          {"destination", file_evidence(publication.destination)},
          {"confirmed",
           file_evidence(object_path(publication.root, confirmed_digest()))}});
  char input = 0;
  std::cin.get(input);
  throw std::runtime_error("CAS fence must end only by supervisor forced kill");
}

std::error_code supervised_publish(const std::filesystem::path& source,
                                   const std::filesystem::path& destination) {
  if (!publication.armed)
    return ex::detail::native_publish_artifact_file(source, destination);
  publication.source = source;
  publication.destination = destination;
  require(destination == object_path(publication.root, target_digest()),
          "private publication seam observed an unexpected object");
  if (publication.boundary == "before-rename") kill_fence("before-rename");
  const auto error =
      ex::detail::native_publish_artifact_file(source, destination);
  if (error) return error;
  ++publication.native_calls;
  if (publication.boundary == "after-rename") kill_fence("after-rename");
  return {};
}

class CleanupMarker final {
 public:
  explicit CleanupMarker(std::filesystem::path path) : path_(std::move(path)) {}
  ~CleanupMarker() {
    std::ofstream marker(path_, std::ios::binary);
    marker << "CAS stack unwound\n";
  }
  CleanupMarker(const CleanupMarker&) = delete;
  CleanupMarker& operator=(const CleanupMarker&) = delete;

 private:
  std::filesystem::path path_;
};

ex::ArtifactDescriptor descriptor(std::size_t size, bool confirmed = false) {
  return {.logical_name = confirmed ? "confirmed.bin" : "target.bin",
          .media_type = "application/octet-stream",
          .producer = "cas-crash-fixture",
          .expected_size_bytes = size};
}

void check_reader(ex::FileArtifactStore& store,
                  const jyppx::ArtifactDigest& digest,
                  std::span<const std::byte> expected) {
  auto reader = store.open(digest);
  require(reader->size_bytes() == expected.size(), "CAS reader size changed");
  std::vector<std::byte> actual(expected.size());
  std::size_t offset = 0U;
  while (offset < actual.size()) {
    const auto count =
        reader->read(std::span<std::byte>(actual).subspan(offset));
    require(count > 0U, "CAS reader ended before its confirmed size");
    offset += count;
  }
  std::array<std::byte, 1U> tail{};
  require(reader->read(tail) == 0U &&
              std::equal(actual.begin(), actual.end(), expected.begin()),
          "CAS reader did not preserve exact immutable content");
}

Json inspect(ex::FileArtifactStore& store) {
  require(store.exists(confirmed_digest()),
          "previously confirmed object disappeared");
  check_reader(store, confirmed_digest(), bytes(kConfirmed));
  const auto target = target_digest();
  const bool target_present = store.exists(target);
  if (target_present) {
    const auto content = target_content();
    check_reader(store, target, content);
  }
  const auto verified = store.verify();
  return {{"confirmed_exists", true},
          {"target_exists", target_present},
          {"verified_files", verified.checked_files},
          {"verified_bytes", verified.checked_bytes},
          {"integrity_ok", verified.ok()}};
}

int execute(std::string_view mode, const std::filesystem::path& root,
            std::string_view boundary) {
  require(mode == "seed" || mode == "crash" || mode == "inspect" ||
              mode == "recover" || mode == "retry" || mode == "gc-confirmed" ||
              mode == "gc-both",
          "unknown CAS mode");
  require(std::filesystem::is_directory(root), "CAS fixture root must exist");
  if (mode == "seed")
    require(!std::filesystem::exists(root / "cas"), "CAS seed must be fresh");
  else
    require(std::filesystem::is_directory(root / "cas"),
            "seeded CAS is missing");
  ex::FileArtifactStore store(root / "cas");
  Json result{{"schema_version", 1U}, {"phase", "completed"}, {"mode", mode}};
  if (mode == "seed") {
    store.put(bytes(kConfirmed),
              descriptor(std::string_view(kConfirmed).size(), true),
              confirmed_digest());
    result["confirmed_sha256"] = confirmed_digest().hex;
    result["confirmed_bytes"] = std::string_view(kConfirmed).size();
    result["target_sha256"] = target_digest().hex;
    result["target_bytes"] = kTargetBytes;
  }
  if (mode == "crash") {
    require(boundary == "before-rename" || boundary == "after-rename" ||
                boundary == "after-commit",
            "unknown CAS crash boundary");
    CleanupMarker cleanup(root / "cas-cleanup.marker");
    const auto content = target_content();
    auto session = store.begin(descriptor(content.size()));
    constexpr std::size_t chunk = 777777U;
    for (std::size_t offset = 0U; offset < content.size();) {
      const auto count = std::min(chunk, content.size() - offset);
      session->write(
          std::span<const std::byte>(content).subspan(offset, count));
      offset += count;
    }
    publication.root = root;
    publication.boundary = std::string(boundary);
    publication.session = session.get();
    publication.armed = true;
    store.commit(*session, target_digest());
    require(publication.native_calls == 1U && !publication.source.empty(),
            "private CAS seam was not linked; unsupported linkage must fail");
    if (boundary == "after-commit") kill_fence("after-commit");
    throw std::runtime_error(
        "CAS operation escaped its required publication fence");
  }
  if (mode == "recover") {
    const auto recovered = store.recover();
    result["recovery"] = {{"removed_files", recovered.removed_files},
                          {"retained_files", recovered.retained_files},
                          {"removed_bytes", recovered.removed_bytes}};
  }
  if (mode == "gc-confirmed" || mode == "gc-both") {
    std::vector<jyppx::ArtifactDigest> live{confirmed_digest()};
    if (mode == "gc-both") live.push_back(target_digest());
    const auto gc = store.collect_garbage(live, std::chrono::seconds::zero());
    result["gc"] = {{"removed_files", gc.removed_files},
                    {"retained_files", gc.retained_files},
                    {"removed_bytes", gc.removed_bytes}};
  }
  if (mode == "retry") {
    const auto content = target_content();
    const auto first =
        store.put(content, descriptor(content.size()), target_digest());
    const auto second =
        store.put(content, descriptor(content.size()), target_digest());
    require(
        first.digest == second.digest && first.size_bytes == second.size_bytes,
        "CAS duplicate upload changed its immutable identity");
    result["repeated_digest"] = first.digest.hex;
  }
  result["store"] = inspect(store);
  report(result);
  return 0;
}

int guarded_execute(std::string_view mode, const std::filesystem::path& root,
                    std::string_view boundary) {
  try {
    return execute(mode, std::filesystem::absolute(root), boundary);
  } catch (const std::exception& error) {
    std::cerr << "cas-crash-child: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace

namespace jyppx::experiment::detail {
// Resolves the production caller's private forwarding symbol from this object.
// The default forwarding object must stay a separate static archive member.
std::error_code publish_artifact_file(
    const std::filesystem::path& source,
    const std::filesystem::path& destination) {
  return ::supervised_publish(source, destination);
}
}  // namespace jyppx::experiment::detail

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
  if (argc != 3 && argc != 4) return 2;
  return guarded_execute(utf8(std::filesystem::path(argv[1])),
                         std::filesystem::path(argv[2]),
                         argc == 4 ? utf8(std::filesystem::path(argv[3])) : "");
}
#else
int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) return 2;
  return guarded_execute(argv[1], std::filesystem::path(argv[2]),
                         argc == 4 ? argv[3] : "");
}
#endif
