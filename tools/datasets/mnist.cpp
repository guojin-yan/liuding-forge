// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "mnist.hpp"

#include "codec_support.hpp"
#include "mnist_detail.hpp"
#include "preparation_support.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>

namespace jyppx::liuding::datasets {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;
constexpr std::uint64_t kMaxOutput = 54950048U;
constexpr std::array<std::string_view, 4> kNames{
    "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};
constexpr std::array<std::uint64_t, 4> kMaxCompressed{9912422U, 28881U,
                                                      1648877U, 4542U};

bool digest_text(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

void validate_contract(const detail::MnistContract& contract) {
  if (contract.train_samples == 0U || contract.train_samples > 60000U ||
      contract.test_samples == 0U || contract.test_samples > 10000U ||
      contract.max_output_bytes == 0U || contract.max_output_bytes > kMaxOutput)
    throw InvalidArgument("Invalid private MNIST counts/output budget");
  std::uint64_t total = 0U;
  for (std::size_t index = 0U; index < contract.files.size(); ++index) {
    const auto& file = contract.files[index];
    const auto samples =
        index < 2U ? contract.train_samples : contract.test_samples;
    const bool image = index % 2U == 0U;
    const auto decoded = image ? 16ULL + 784ULL * samples : 8ULL + samples;
    if (file.name != kNames[index] || file.compressed_bytes == 0U ||
        file.compressed_bytes > kMaxCompressed[index] ||
        !digest_text(file.compressed_sha256) ||
        !digest_text(file.decoded_sha256) || file.decoded_bytes != decoded)
      throw InvalidArgument("Invalid private MNIST file identity/IDX bounds");
    if (decoded > contract.max_output_bytes - total)
      throw ResourceExhausted("MNIST complete output exceeds byte budget");
    total += decoded;
  }
}

void verify_sources(const fs::path& source,
                    const detail::MnistContract& contract) {
  if (!fs::is_directory(source))
    throw InvalidArgument("MNIST source must be a local directory");
  std::array<unsigned char, detail::kCodecChunkBytes> block{};
  for (const auto& file : contract.files) {
    const auto path = source / file.name;
    detail::reject_links(path);
    if (!fs::is_regular_file(path) ||
        fs::file_size(path) != file.compressed_bytes)
      throw InvalidArgument("MNIST source size mismatch: " + file.name);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw IoError("Could not open MNIST source: " + file.name);
    Sha256 hash;
    std::uint64_t total = 0U;
    while (input) {
      input.read(reinterpret_cast<char*>(block.data()),
                 static_cast<std::streamsize>(block.size()));
      const auto count = static_cast<std::size_t>(input.gcount());
      if (count > file.compressed_bytes - total)
        throw InvalidArgument("MNIST source grew while reading");
      hash.update(std::as_bytes(std::span(block).first(count)));
      total += count;
    }
    if (!input.eof())
      throw IoError("Could not read MNIST source: " + file.name);
    if (total != file.compressed_bytes ||
        sha256_digest_hex(hash.finish()) != file.compressed_sha256)
      throw InvalidArgument("MNIST source SHA-256 mismatch: " + file.name);
  }
}

std::uint32_t be32(std::span<const unsigned char> bytes) {
  std::uint32_t value = 0U;
  for (const auto byte : bytes) value = (value << 8U) | byte;
  return value;
}

Json decode_file(const fs::path& source, const fs::path& staging,
                 const detail::MnistFile& file, std::uint32_t samples,
                 bool image, std::array<std::uint64_t, 10>& labels) {
  detail::GzipReader gzip(source / file.name, file.compressed_bytes,
                          file.compressed_sha256);
  std::array<unsigned char, 16> header{};
  const std::size_t header_bytes = image ? 16U : 8U;
  const auto extent = std::span(header).first(header_bytes);
  gzip.exact(extent);
  if (be32(extent.first(4U)) != (image ? 2051U : 2049U) ||
      be32(extent.subspan(4U, 4U)) != samples ||
      (image && (be32(extent.subspan(8U, 4U)) != 28U ||
                 be32(extent.subspan(12U, 4U)) != 28U)))
    throw InvalidArgument("MNIST IDX magic/count/shape mismatch: " + file.name);

  const auto name = file.name.substr(0U, file.name.size() - 3U);
  if (detail::present(staging / name))
    throw IoError("Refusing to replace a staged MNIST IDX file");
  std::ofstream output(staging / name, std::ios::binary | std::ios::trunc);
  if (!output) throw IoError("Could not create staged MNIST IDX: " + name);
  output.write(reinterpret_cast<const char*>(extent.data()),
               static_cast<std::streamsize>(extent.size()));
  Sha256 hash;
  hash.update(std::as_bytes(extent));
  std::array<unsigned char, detail::kCodecChunkBytes> block{};
  auto remaining = file.decoded_bytes - header_bytes;
  while (remaining != 0U) {
    const auto count = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, block.size()));
    const auto bytes = std::span(block).first(count);
    gzip.exact(bytes);
    if (!image)
      for (const auto label : bytes) {
        if (label >= labels.size())
          throw InvalidArgument("MNIST IDX label is outside 0..9");
        ++labels[label];
      }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) throw IoError("Could not write complete MNIST IDX: " + name);
    hash.update(std::as_bytes(bytes));
    remaining -= count;
  }
  gzip.finish();
  const auto decoded_sha = sha256_digest_hex(hash.finish());
  if (decoded_sha != file.decoded_sha256)
    throw InvalidArgument("MNIST decoded IDX SHA-256 mismatch: " + file.name);
  detail::close_checked(output);
  return {{"source_file", file.name},
          {"source_bytes", file.compressed_bytes},
          {"source_sha256", file.compressed_sha256},
          {"idx_file", name},
          {"idx_bytes", file.decoded_bytes},
          {"idx_sha256", decoded_sha}};
}
}  // namespace

void detail::validate_mnist_contract(const MnistContract& contract) {
  validate_contract(contract);
}

const detail::MnistContract& detail::official_mnist_contract() {
  static const MnistContract contract{
      60000U,
      10000U,
      kMaxOutput,
      {{{"train-images-idx3-ubyte.gz", 9912422U,
         "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
         47040016U,
         "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db"},
        {"train-labels-idx1-ubyte.gz", 28881U,
         "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
         60008U,
         "65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5"},
        {"t10k-images-idx3-ubyte.gz", 1648877U,
         "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
         7840016U,
         "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7"},
        {"t10k-labels-idx1-ubyte.gz", 4542U,
         "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
         10008U,
         "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2"}}}};
  return contract;
}

fs::path detail::prepare_mnist_contract(const fs::path& source_dir,
                                        const fs::path& data_root,
                                        const MnistContract& contract) {
  try {
    validate_mnist_contract(contract);
    if (source_dir.empty())
      throw InvalidArgument("MNIST local gzip source directory is required");
    const auto absolute = fs::absolute(source_dir);
    reject_links(absolute);
    const auto source = absolute.lexically_normal();
    const auto root = external_root(data_root);
    if (present(root / "mnist"))
      throw IoError("Refusing to replace prepared MNIST directory");
    verify_sources(source, contract);
    StagingDirectory transaction(root, "mnist", ".mnist-preparing-");
    Json files = Json::array();
    std::array<std::array<std::uint64_t, 10>, 2> counts{};
    for (std::size_t index = 0U; index < contract.files.size(); ++index) {
      const auto split = index < 2U ? 0U : 1U;
      files.push_back(decode_file(
          source, transaction.path(), contract.files[index],
          split == 0U ? contract.train_samples : contract.test_samples,
          index % 2U == 0U, counts[split]));
    }
    const Json report{
        {"schema_version", 1U},
        {"dataset_id", "jyppx.liuding.mnist.idx/v1"},
        {"source", "https://storage.googleapis.com/cvdf-datasets/mnist/"},
        {"checksum_reference",
         "https://github.com/tensorflow/datasets/blob/master/"
         "tensorflow_datasets/url_checksums/mnist.txt"},
        {"samples",
         {{"train", contract.train_samples}, {"test", contract.test_samples}}},
        {"label_counts", {{"train", counts[0]}, {"test", counts[1]}}},
        {"pixels", "28x28 grayscale uint8; unchanged IDX bytes"},
        {"split_policy", "complete official train/test; no subset"},
        {"files", files}};
    // A new report contract: binary LF makes it identical across hosts.
    std::ofstream output(transaction.path() / "preparation.json",
                         std::ios::binary | std::ios::trunc);
    if (!output) throw IoError("Could not create MNIST preparation report");
    output << report.dump(2) << '\n';
    close_checked(output);
    const auto destination = transaction.destination();
    transaction.publish();
    return destination;
  } catch (const Error&) {
    throw;
  } catch (const fs::filesystem_error& error) {
    throw IoError(std::string("MNIST filesystem operation failed: ") +
                  error.what());
  }
}

fs::path prepare_mnist(const fs::path& source_dir, const fs::path& data_root) {
  return detail::prepare_mnist_contract(source_dir, data_root,
                                        detail::official_mnist_contract());
}
}  // namespace jyppx::liuding::datasets
