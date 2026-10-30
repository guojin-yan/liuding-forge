// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "fashion_mnist.hpp"

#include "codec_support.hpp"
#include "fashion_mnist_detail.hpp"
#include "fashion_mnist_license.hpp"
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
#include <vector>

#if !defined(JYPPX_LIUDING_FASHION_ZLIB_SOURCE_SHA256) || \
    !defined(JYPPX_LIUDING_FASHION_ZLIB_LIBRARY_SHA256)
#error "Explicit audited zlib SDK provenance is required"
#endif

namespace jyppx::liuding::datasets {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using detail::FashionArchive;
using detail::FashionContract;
using detail::FashionSplit;
constexpr std::size_t kSide = 28U;
constexpr std::size_t kPixels = kSide * kSide;
constexpr auto kChunk = detail::kCodecChunkBytes;
constexpr std::uint64_t kMaxSource = 32U * 1024U * 1024U;
constexpr std::uint64_t kMaxOutput = 768U * 1024U * 1024U;
constexpr auto kMaxPng = detail::kMaxDatasetPngBytes;
constexpr std::size_t kMaxIndexLine = 512U;
constexpr auto kCodecBudget = detail::kCodecAllocationBudgetBytes;
constexpr std::string_view kDataset = "jyppx.liuding.fashion-mnist/v1";
constexpr std::string_view kLicenseSha =
    "13ef4788476d292858fa60eb9a5f74aeca5c65770bc885ccaa05823a17ef7be1";
constexpr std::array<std::string_view, 10> kNames{
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"};
constexpr std::array<std::string_view, 10> kDirs{
    "00-t-shirt-top", "01-trouser", "02-pullover", "03-dress", "04-coat",
    "05-sandal",      "06-shirt",   "07-sneaker",  "08-bag",   "09-ankle-boot"};

std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}
template <typename T, std::size_t N>
std::span<const std::byte> bytes(const std::array<T, N>& data) {
  return std::as_bytes(std::span(data));
}
std::string selection_name(std::string_view selection) {
  if (selection != "full" && selection != "first100-per-class")
    throw InvalidArgument(
        "Fashion-MNIST selection must be full or first100-per-class");
  return std::string(selection);
}
void domain(Sha256& hash, std::string_view value) {
  hash.update(bytes(value));
  constexpr std::array<std::byte, 1> zero{};
  hash.update(zero);
}

void verify_archive(const fs::path& path, const FashionArchive& expected) {
  detail::reject_links(path);
  if (!fs::is_regular_file(path) || fs::file_size(path) != expected.bytes)
    throw CompatibilityError("Fashion-MNIST compressed archive size mismatch");
  std::ifstream input(path, std::ios::binary);
  if (!input) throw IoError("Could not open Fashion-MNIST compressed source");
  std::array<char, kChunk> buffer{};
  Sha256 hash;
  std::uint64_t total = 0U;
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = static_cast<std::size_t>(input.gcount());
    if (count > expected.bytes - total)
      throw CompatibilityError(
          "Fashion-MNIST compressed source grew while hashing");
    hash.update(bytes(std::string_view(buffer.data(), count)));
    total += count;
  }
  if (!input.eof())
    throw IoError("Could not read Fashion-MNIST compressed source");
  if (total != expected.bytes ||
      sha256_digest_hex(hash.finish()) != expected.sha256)
    throw CompatibilityError(
        "Fashion-MNIST compressed source SHA-256 mismatch");
}

std::uint32_t be32(const unsigned char* data) noexcept {
  return (static_cast<std::uint32_t>(data[0]) << 24U) |
         (static_cast<std::uint32_t>(data[1]) << 16U) |
         (static_cast<std::uint32_t>(data[2]) << 8U) | data[3];
}
std::vector<unsigned char> rgb_png(
    const std::array<unsigned char, kPixels>& gray) {
  std::array<unsigned char, 3U * kPixels> rgb{};
  for (std::size_t pixel = 0U; pixel < gray.size(); ++pixel)
    for (std::size_t channel = 0U; channel < 3U; ++channel)
      rgb[3U * pixel + channel] = gray[pixel];
  return detail::encode_rgb_png(rgb, static_cast<std::uint32_t>(kSide),
                                static_cast<std::uint32_t>(kSide));
}

struct OutputBudget {
  std::uint64_t bytes = 0U;
  void consume(std::size_t count) {
    if (count > kMaxOutput - bytes)
      throw ResourceExhausted("Fashion-MNIST output exceeds 768 MiB");
    bytes += count;
  }
};
void write_file(const fs::path& path, std::span<const std::byte> content,
                OutputBudget& budget) {
  budget.consume(content.size());
  if (detail::present(path))
    throw IoError("Refusing to replace dataset staging file");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw IoError("Could not open Fashion-MNIST staging artifact");
  output.write(reinterpret_cast<const char*>(content.data()),
               static_cast<std::streamsize>(content.size()));
  detail::close_checked(output);
}
struct SplitReport {
  Json json;
  Sha256Digest digest;
};
SplitReport convert_split(const fs::path& source, const fs::path& staging,
                          const FashionSplit& info, std::string_view selection,
                          OutputBudget& budget) {
  const auto quota = selection == "full" ? info.samples_per_class : 100U;
  const auto index_file = "index/" + info.name + ".jsonl";
  std::ofstream index(staging / index_file, std::ios::binary | std::ios::trunc);
  if (!index) throw IoError("Could not open Fashion-MNIST staging index");
  for (const auto directory : kDirs)
    fs::create_directories(staging / info.name / directory);
  detail::GzipReader images(source / info.images.name, info.images.bytes,
                            info.images.sha256, kPixels);
  detail::GzipReader labels(source / info.labels.name, info.labels.bytes,
                            info.labels.sha256, kPixels);
  std::array<unsigned char, 16> ih{};
  std::array<unsigned char, 8> lh{};
  images.exact(ih);
  labels.exact(lh);
  if (be32(ih.data()) != 2051U || be32(ih.data() + 4U) != info.samples ||
      be32(ih.data() + 8U) != kSide || be32(ih.data() + 12U) != kSide ||
      be32(lh.data()) != 2049U || be32(lh.data() + 4U) != info.samples)
    throw CompatibilityError("Fashion-MNIST IDX magic/count/geometry mismatch");
  std::array<std::uint32_t, 10> source_counts{};
  std::array<std::uint32_t, 10> selected_counts{};
  Sha256 content;
  domain(content, "jyppx.liuding.fashion-mnist.split.v1");
  domain(content, info.name);
  domain(content, selection);
  Sha256 index_hash;
  std::uint64_t index_bytes = 0U;
  std::uint64_t png_bytes = 0U;
  for (std::uint32_t number = 0U; number < info.samples; ++number) {
    std::array<unsigned char, kPixels> gray{};
    std::array<unsigned char, 1> label{};
    images.exact(gray);
    labels.exact(label);
    if (label[0] >= kDirs.size())
      throw CompatibilityError("Fashion-MNIST source label is outside [0,9]");
    ++source_counts[label[0]];
    if (selected_counts[label[0]] == quota) continue;
    ++selected_counts[label[0]];
    const auto serial = std::to_string(number);
    const auto relative = info.name + "/" + std::string(kDirs[label[0]]) + "/" +
                          std::string(6U - serial.size(), '0') + serial +
                          ".png";
    const auto png = rgb_png(gray);
    const auto png_span = std::as_bytes(std::span(png));
    write_file(staging / relative, png_span, budget);
    png_bytes += png.size();
    const Json row{{"class_id", label[0]},
                   {"source_index", number},
                   {"path", relative},
                   {"grayscale_sha256", sha256_hex(bytes(gray))},
                   {"png_sha256", sha256_hex(png_span)}};
    const auto line = row.dump() + "\n";
    if (line.size() > kMaxIndexLine)
      throw ResourceExhausted("Fashion-MNIST index record exceeds 512 bytes");
    budget.consume(line.size());
    index.write(line.data(), static_cast<std::streamsize>(line.size()));
    if (!index) throw IoError("Could not write Fashion-MNIST index record");
    index_hash.update(bytes(line));
    index_bytes += line.size();
    std::array<unsigned char, 9> identity{};
    for (unsigned b = 0U; b < 8U; ++b)
      identity[b] = static_cast<unsigned char>(
          static_cast<std::uint64_t>(number) >> (8U * b));
    identity[8] = label[0];
    content.update(bytes(identity));
    content.update(bytes(gray));
  }
  // Both readers finish even when the requested subset was satisfied early.
  images.finish();
  labels.finish();
  detail::close_checked(index);
  for (std::size_t c = 0U; c < kDirs.size(); ++c)
    if (source_counts[c] != info.samples_per_class ||
        selected_counts[c] != quota)
      throw CompatibilityError(
          "Fashion-MNIST full-source class balance or selection quota "
          "mismatch");
  const auto digest = content.finish();
  return {{{"source_samples", info.samples},
           {"source_class_counts", source_counts},
           {"samples", 10U * quota},
           {"class_counts", selected_counts},
           {"content_sha256", sha256_digest_hex(digest)},
           {"png_bytes", png_bytes},
           {"index",
            {{"file", index_file},
             {"bytes", index_bytes},
             {"sha256", sha256_digest_hex(index_hash.finish())}}}},
          digest};
}
void validate_contract(const FashionContract& contract,
                       std::string_view selection) {
  constexpr std::array<std::string_view, 2> names{"train", "validation"};
  constexpr std::array<std::string_view, 4> files{
      "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
      "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"};
  for (std::size_t i = 0U; i < contract.size(); ++i) {
    const auto& split = contract[i];
    if (split.name != names[i] || split.samples == 0U ||
        split.samples > 60000U || split.samples_per_class == 0U ||
        split.samples_per_class > 6000U ||
        split.samples != 10U * split.samples_per_class ||
        (selection != "full" && split.samples_per_class < 100U) ||
        split.images.name != files[2U * i] ||
        split.labels.name != files[2U * i + 1U])
      throw InvalidArgument("Invalid private Fashion-MNIST source contract");
    for (const auto* archive : {&split.images, &split.labels}) {
      const auto valid_hex = [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      };
      if (archive->bytes == 0U || archive->bytes > kMaxSource ||
          archive->sha256.size() != 64U ||
          !std::all_of(archive->sha256.begin(), archive->sha256.end(),
                       valid_hex))
        throw InvalidArgument(
            "Invalid private Fashion-MNIST archive limits or SHA-256");
    }
  }
}
}  // namespace

const detail::FashionContract& detail::official_fashion_contract() {
  static const FashionContract contract{
      {{"train",
        60000U,
        6000U,
        {"train-images-idx3-ubyte.gz", 26421880U,
         "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"},
        {"train-labels-idx1-ubyte.gz", 29515U,
         "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
         "25c81989df183df01b3e8a0aad5dffbe"}},
       {"validation",
        10000U,
        1000U,
        {"t10k-images-idx3-ubyte.gz", 4422102U,
         "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
         "bef4ecab320f06d8554ea6380940ec79"},
        {"t10k-labels-idx1-ubyte.gz", 5148U,
         "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
         "bb300cfdad3c16e7a12a480ee83cd310"}}}};
  return contract;
}

fs::path detail::prepare_fashion_mnist_contract(
    const fs::path& source_dir, const fs::path& data_root,
    std::string_view selection, const FashionContract& contract) {
  try {
    const auto selected = selection_name(selection);
    validate_contract(contract, selected);
    if (source_dir.empty())
      throw InvalidArgument("Fashion-MNIST source directory is required");
    const auto absolute_source = fs::absolute(source_dir);
    reject_links(absolute_source);
    const auto source = absolute_source.lexically_normal();
    if (!fs::is_directory(source))
      throw InvalidArgument("Fashion-MNIST source must be a directory");
    const auto root = external_root(data_root);
    const auto destination = "fashion-mnist-" + selected + "-imagefolder-v1";
    if (present(root / destination))
      throw IoError("Refusing to replace prepared dataset");
    const auto license_bytes = bytes(kFashionLicense);
    if (sha256_hex(license_bytes) != kLicenseSha)
      throw CompatibilityError(
          "Embedded Fashion-MNIST MIT license identity mismatch");
    // Verify all four compressed identities before creating any output
    // artifact.
    Json sources = Json::object();
    for (const auto& split : contract) {
      for (const auto& kind : {std::pair{"images", &split.images},
                               std::pair{"labels", &split.labels}}) {
        verify_archive(source / kind.second->name, *kind.second);
        sources[split.name][kind.first] = {
            {"file", kind.second->name},
            {"bytes", kind.second->bytes},
            {"sha256", kind.second->sha256},
            {"official_md5", kind.second->upstream_md5},
            {"official_md5_provenance", "upstream declared; not computed"}};
      }
    }
    StagingDirectory transaction(root, destination,
                                 ".fashion-mnist-preparing-");
    const auto& staging = transaction.path();
    fs::create_directory(staging / "index");
    OutputBudget budget;
    Json splits = Json::object();
    Sha256 identity;
    domain(identity, "jyppx.liuding.fashion-mnist.imagefolder.v1");
    domain(identity, selected);
    for (const auto& split : contract) {
      const auto report =
          convert_split(source, staging, split, selected, budget);
      domain(identity, split.name);
      identity.update(report.digest);
      splits[split.name] = report.json;
    }
    Json classes = Json::array();
    for (std::size_t c = 0U; c < kDirs.size(); ++c)
      classes.push_back(
          {{"id", c}, {"name", kNames[c]}, {"directory", kDirs[c]}});
    const Json manifest{
        {"schema_version", 1U},
        {"dataset_id", kDataset},
        {"source",
         {{"repository", "https://github.com/zalandoresearch/fashion-mnist"},
          {"revision", "b2617bb6d3ffa2e429640350f613e3291e10b141"},
          {"raw_base",
           "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
           "b2617bb6d3ffa2e429640350f613e3291e10b141/data/fashion/"}}},
        {"source_files", sources},
        {"license",
         {{"spdx", "MIT"},
          {"file", "fashion-mnist.LICENSE"},
          {"sha256", kLicenseSha}}},
        {"selection", selected},
        {"classes", classes},
        {"splits", splits},
        {"pixels",
         "28x28 RGB uint8 PNG; each source grayscale byte repeated in R/G/B; "
         "no resizing"},
        {"split_policy",
         "original train/test; test is named validation; retain source order "
         "and indices"},
        {"content_sha256", sha256_digest_hex(identity.finish())},
        {"content_hash_format",
         {{"split",
           "SHA256(ASCII jyppx.liuding.fashion-mnist.split.v1 NUL + split NUL "
           "+ selection NUL + selected records in source order); record = "
           "uint64 little-endian source index + uint8 label + 784 original "
           "grayscale bytes"},
          {"dataset",
           "SHA256(ASCII jyppx.liuding.fashion-mnist.imagefolder.v1 NUL + "
           "selection NUL + train NUL + raw train split SHA256 + validation "
           "NUL + raw validation split SHA256)"}}},
        {"png_encoding",
         {{"profile", "rgb8-filter0-zlib-level1-single-idat/v1"},
          {"zlib_version", "1.3.2"},
          {"zlib_source_sha256", JYPPX_LIUDING_FASHION_ZLIB_SOURCE_SHA256},
          {"zlib_static_library_sha256",
           JYPPX_LIUDING_FASHION_ZLIB_LIBRARY_SHA256}}},
        {"resource_limits",
         {{"source_archive_bytes", kMaxSource},
          {"compressed_read_bytes", kChunk},
          {"decoded_read_bytes", kPixels},
          {"codec_live_allocation_bytes_per_stream", kCodecBudget},
          {"png_bytes", kMaxPng},
          {"index_line_bytes", kMaxIndexLine},
          {"total_output_bytes", kMaxOutput}}}};
    write_file(staging / "fashion-mnist.LICENSE", license_bytes, budget);
    const auto text = manifest.dump(2) + "\n";
    if (text.size() > 65536U)
      throw ResourceExhausted("Fashion-MNIST manifest exceeds 64 KiB");
    write_file(staging / "preparation.json", bytes(text), budget);
    transaction.publish();
    return transaction.destination();
  } catch (const fs::filesystem_error&) {
    throw IoError("Fashion-MNIST preparation filesystem operation failed");
  }
}

fs::path prepare_fashion_mnist(const fs::path& source_dir,
                               const fs::path& data_root,
                               std::string_view selection) {
  return detail::prepare_fashion_mnist_contract(
      source_dir, data_root, selection, detail::official_fashion_contract());
}
}  // namespace jyppx::liuding::datasets
