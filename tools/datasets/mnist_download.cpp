// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "mnist_download.hpp"

#include "mnist_download_detail.hpp"
#include "preparation_support.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <fstream>
#include <string>

namespace jyppx::liuding::datasets {
namespace fs = std::filesystem;
void detail::require_mnist_https_url(std::string_view url) {
  // Exact full-URL allowlist: no credentials, query, fragment, alternate
  // endpoint, scheme, host, path normalization, or redirect destination.
  for (const auto& file : official_mnist_contract().files)
    if (url == std::string(kMnistHttpsSource) + file.name) return;
  throw InvalidArgument("Only the four pinned MNIST HTTPS URLs are allowed");
}

fs::path detail::download_prepare_mnist_contract(
    const fs::path& data_root, const MnistContract& contract,
    const MnistHttpsTransfer& transfer) {
  try {
    validate_mnist_contract(contract);
    if (!transfer) throw InvalidArgument("MNIST HTTPS transport is required");
    const auto root = external_root(data_root);
    if (present(root / "mnist"))
      throw IoError("Refusing to replace prepared MNIST directory");
    // This transaction owns compressed inputs only and is never published.
    // The offline preparer creates/publishes its separate IDX transaction.
    StagingDirectory downloads(root, "mnist", ".mnist-downloading-");
    for (const auto& file : contract.files) {
      const auto url = std::string(kMnistHttpsSource) + file.name;
      require_mnist_https_url(url);
      const auto path = downloads.path() / file.name;
      if (present(path))
        throw IoError("Refusing to replace a staged MNIST archive");
      std::ofstream output(path, std::ios::binary | std::ios::trunc);
      if (!output) throw IoError("Could not create staged MNIST archive");
      Sha256 hash;
      std::uint64_t total = 0U;
      bool received_headers = false;
      const MnistHttpsCallbacks callbacks{
          [&](std::uint32_t status, std::optional<std::uint64_t> length) {
            if (received_headers)
              throw InvalidArgument("Repeated MNIST HTTP response headers");
            if (status != 200U)
              throw IoError(
                  "MNIST HTTPS response must be 200; redirects are refused");
            if (length && *length != file.compressed_bytes)
              throw InvalidArgument("MNIST HTTP Content-Length mismatch");
            received_headers = true;
          },
          [&](std::span<const std::byte> bytes) {
            // libcurl reports final response metadata after the transfer.
            // Only this private transaction receives bytes before validation;
            // metadata, exact length and SHA must all pass before publication.
            if (bytes.size() > kMnistDownloadChunkBytes ||
                bytes.size() > file.compressed_bytes - total)
              throw ResourceExhausted("MNIST download exceeded byte budget");
            if (bytes.empty()) return;
            output.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size()));
            if (!output) throw IoError("Could not write MNIST archive");
            hash.update(bytes);
            total += bytes.size();
          }};
      transfer(url, file.compressed_bytes, callbacks);
      if (!received_headers || total != file.compressed_bytes)
        throw InvalidArgument("Incomplete MNIST HTTPS response");
      if (sha256_digest_hex(hash.finish()) != file.compressed_sha256)
        throw InvalidArgument("MNIST HTTPS SHA-256 mismatch");
      close_checked(output);
    }
    // The owner attempts stage cleanup on ordinary return/exception. The
    // offline path verifies original-byte IDX, gzip integrity and the report.
    return prepare_mnist_contract(downloads.path(), root, contract);
  } catch (const Error&) {
    throw;
  } catch (const fs::filesystem_error& error) {
    throw IoError(std::string("MNIST download filesystem operation failed: ") +
                  error.what());
  }
}

fs::path download_prepare_mnist(const fs::path& data_root) {
  return detail::download_prepare_mnist_contract(
      data_root, detail::official_mnist_contract(),
      detail::transfer_mnist_https);
}
}  // namespace jyppx::liuding::datasets
