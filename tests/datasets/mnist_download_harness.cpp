// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "../../tools/datasets/mnist_download_detail.hpp"

#include <jyppx/liuding/foundation/error.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace {
namespace fs = std::filesystem;
namespace data = jyppx::liuding::datasets::detail;
int run(const fs::path& contract_path, const fs::path& source,
        const fs::path& root, const std::string& mode) {
  try {
    if (!fs::is_regular_file(contract_path) ||
        fs::file_size(contract_path) > 65536U)
      throw jyppx::liuding::InvalidArgument("Invalid private MNIST contract");
    std::ifstream input(contract_path);
    const auto json = nlohmann::json::parse(input);
    data::MnistContract contract{
        json.at("train_samples").get<std::uint32_t>(),
        json.at("test_samples").get<std::uint32_t>(),
        json.at("max_output_bytes").get<std::uint64_t>(),
        {}};
    const auto& files = json.at("files");
    if (!files.is_array() || files.size() != contract.files.size())
      throw jyppx::liuding::InvalidArgument("Four fixture files required");
    for (std::size_t i = 0U; i < files.size(); ++i) {
      const auto& file = files[i];
      contract.files[i] = {file.at("name").get<std::string>(),
                           file.at("compressed_bytes").get<std::uint64_t>(),
                           file.at("compressed_sha256").get<std::string>(),
                           file.at("decoded_bytes").get<std::uint64_t>(),
                           file.at("decoded_sha256").get<std::string>()};
    }
    std::size_t request = 0U;
    const data::MnistHttpsTransfer fake =
        [&](std::string_view url, std::uint64_t expected,
            const data::MnistHttpsCallbacks& callbacks) {
          const auto index = request++;
          if (index >= contract.files.size() ||
              url != std::string(data::kMnistHttpsSource) +
                         contract.files[index].name ||
              expected != contract.files[index].compressed_bytes)
            throw jyppx::liuding::InvalidArgument("Unexpected fake request");
          std::cout << "GET " << url << ' ' << expected << '\n';
          if (mode == "transport" || (mode == "fail-third" && index == 2U))
            throw jyppx::liuding::IoError("Injected transport failure");
          const auto status =
              mode == "redirect" || mode == "late-redirect"   ? 302U
              : mode == "notfound" || mode == "late-notfound" ? 404U
                                                              : 200U;
          std::optional<std::uint64_t> length = expected;
          if (mode == "chunked" || mode == "late-chunked")
            length = std::nullopt;
          if (mode == "length-large" || mode == "late-length-large")
            length = expected + 1U;
          if (mode == "length-small" || mode == "late-length-small")
            length = expected - 1U;
          if (mode == "missing-headers") return;
          const bool final_metadata =
              mode == "body-before-head" || mode.starts_with("late-");
          if (!final_metadata && mode != "missing-metadata-after-body")
            callbacks.headers(status, length);
          if (mode == "repeated-head") callbacks.headers(status, length);
          if (mode == "empty-body") return;
          if (mode == "oversized-callback") {
            const std::array<std::byte, data::kMnistDownloadChunkBytes + 1U>
                oversized{};
            callbacks.bytes(oversized);
            return;
          }
          std::ifstream archive(source / contract.files[index].name,
                                std::ios::binary);
          if (!archive) throw jyppx::liuding::IoError("Fixture missing");
          std::array<std::byte, data::kMnistDownloadChunkBytes> block{};
          // Tiny writes force fragmentation independently of gzip boundaries.
          const std::size_t chunk = mode == "large-block" ? block.size() : 17U;
          const auto target = mode == "short-body" ? expected - 1U : expected;
          std::uint64_t copied = 0U;
          while (copied < target) {
            const auto count = static_cast<std::size_t>(
                std::min<std::uint64_t>(target - copied, chunk));
            archive.read(reinterpret_cast<char*>(block.data()),
                         static_cast<std::streamsize>(count));
            if (archive.gcount() != static_cast<std::streamsize>(count))
              throw jyppx::liuding::IoError("Fixture shorter than contract");
            if (mode == "tampered" && copied == 0U) block[0] ^= std::byte{1U};
            callbacks.bytes(std::span(block).first(count));
            copied += count;
          }
          if (mode == "extra-body") callbacks.bytes(std::span(block).first(1U));
          if (final_metadata) callbacks.headers(status, length);
          if (mode == "late-repeated-head") callbacks.headers(status, length);
          if (mode == "throw-after-body" || mode == "late-throw-after-body")
            throw jyppx::liuding::IoError("Injected completion failure");
          if (mode == "late-destination" && index == 3U) {
            fs::create_directory(root / "mnist");
            std::ofstream marker(root / "mnist" / "winner");
            marker << "preserved\n";
          }
        };
    data::download_prepare_mnist_contract(root, contract, fake);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
  if (argc != 5) return 2;
  return run(fs::path(argv[1]), fs::path(argv[2]), fs::path(argv[3]),
             fs::path(argv[4]).string());
}
#else
int main(int argc, char* argv[]) {
  if (argc != 5) return 2;
  return run(fs::u8path(argv[1]), fs::u8path(argv[2]), fs::u8path(argv[3]),
             argv[4]);
}
#endif
