// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "cifar10.hpp"

#include "cifar10_detail.hpp"
#include "codec_support.hpp"
#include "preparation_support.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jyppx::liuding::datasets {
namespace {
namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;
constexpr std::string_view kPrefix = "cifar-10-batches-bin";
constexpr std::string_view kDestination = "cifar10-imagefolder-v1";
constexpr std::size_t kPixels = 32U * 32U;
constexpr std::size_t kRecordBytes = 1U + 3U * kPixels;
constexpr std::uint64_t kMaxCompressed = 170052171U;
constexpr std::uint64_t kMaxDecoded = 185000000U;
constexpr std::uint64_t kTarOverheadBudget = 64U * 1024U;
constexpr std::size_t kMetadataBytes = 65536U;
constexpr std::uint64_t kMaxOutput = 60000ULL * detail::kMaxDatasetPngBytes;
constexpr std::array<std::string_view, 10> kClasses{
    "airplane", "automobile", "bird",  "cat",  "deer",
    "dog",      "frog",       "horse", "ship", "truck"};
constexpr std::array<std::string_view, 2> kSplits{"train", "validation"};
constexpr std::array<std::string_view, 6> kBatchNames{
    "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
    "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"};

// The upstream MD5 is checked in addition to the mandatory pinned SHA-256.
// This private implementation streams bounded blocks and is not a security or
// signing primitive, a public API, or an alternative source-identity gate.
class Md5 final {
 public:
  void update(std::span<const unsigned char> input) {
    bytes_ += input.size();
    while (!input.empty()) {
      const auto count = std::min(input.size(), buffer_.size() - buffered_);
      std::copy_n(input.begin(), count, buffer_.begin() + buffered_);
      input = input.subspan(count);
      buffered_ += count;
      if (buffered_ == buffer_.size()) {
        transform();
        buffered_ = 0U;
      }
    }
  }

  std::string finish() {
    const auto bits = bytes_ * 8U;
    const std::array<unsigned char, 64> padding{0x80U};
    const auto count = buffered_ < 56U ? 56U - buffered_ : 120U - buffered_;
    update(std::span(padding).first(count));
    std::array<unsigned char, 8> length{};
    for (unsigned i = 0U; i < length.size(); ++i)
      length[i] = static_cast<unsigned char>((bits >> (8U * i)) & 0xffU);
    update(length);
    constexpr std::string_view alphabet = "0123456789abcdef";
    std::string result;
    result.reserve(32U);
    for (const auto word : state_)
      for (unsigned i = 0U; i < 4U; ++i) {
        const auto value = (word >> (8U * i)) & 0xffU;
        result.push_back(alphabet[value >> 4U]);
        result.push_back(alphabet[value & 15U]);
      }
    return result;
  }

 private:
  void transform() {
    constexpr std::array<int, 64> shift{
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};
    constexpr std::array<std::uint32_t, 64> constants{
        0xd76aa478U, 0xe8c7b756U, 0x242070dbU, 0xc1bdceeeU, 0xf57c0fafU,
        0x4787c62aU, 0xa8304613U, 0xfd469501U, 0x698098d8U, 0x8b44f7afU,
        0xffff5bb1U, 0x895cd7beU, 0x6b901122U, 0xfd987193U, 0xa679438eU,
        0x49b40821U, 0xf61e2562U, 0xc040b340U, 0x265e5a51U, 0xe9b6c7aaU,
        0xd62f105dU, 0x02441453U, 0xd8a1e681U, 0xe7d3fbc8U, 0x21e1cde6U,
        0xc33707d6U, 0xf4d50d87U, 0x455a14edU, 0xa9e3e905U, 0xfcefa3f8U,
        0x676f02d9U, 0x8d2a4c8aU, 0xfffa3942U, 0x8771f681U, 0x6d9d6122U,
        0xfde5380cU, 0xa4beea44U, 0x4bdecfa9U, 0xf6bb4b60U, 0xbebfbc70U,
        0x289b7ec6U, 0xeaa127faU, 0xd4ef3085U, 0x04881d05U, 0xd9d4d039U,
        0xe6db99e5U, 0x1fa27cf8U, 0xc4ac5665U, 0xf4292244U, 0x432aff97U,
        0xab9423a7U, 0xfc93a039U, 0x655b59c3U, 0x8f0ccc92U, 0xffeff47dU,
        0x85845dd1U, 0x6fa87e4fU, 0xfe2ce6e0U, 0xa3014314U, 0x4e0811a1U,
        0xf7537e82U, 0xbd3af235U, 0x2ad7d2bbU, 0xeb86d391U};
    std::array<std::uint32_t, 16> words{};
    for (std::size_t i = 0U; i < words.size(); ++i)
      for (unsigned byte = 0U; byte < 4U; ++byte)
        words[i] |= static_cast<std::uint32_t>(buffer_[4U * i + byte])
                    << (8U * byte);
    auto [a, b, c, d] = state_;
    for (std::size_t i = 0U; i < constants.size(); ++i) {
      std::uint32_t f = 0U;
      std::size_t index = 0U;
      if (i < 16U) {
        f = (b & c) | (~b & d);
        index = i;
      } else if (i < 32U) {
        f = (d & b) | (~d & c);
        index = (5U * i + 1U) % 16U;
      } else if (i < 48U) {
        f = b ^ c ^ d;
        index = (3U * i + 5U) % 16U;
      } else {
        f = c ^ (b | ~d);
        index = (7U * i) % 16U;
      }
      const auto old_d = d;
      d = c;
      c = b;
      b += std::rotl(a + f + constants[i] + words[index], shift[i]);
      a = old_d;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
  }

  std::array<std::uint32_t, 4> state_{0x67452301U, 0xefcdab89U, 0x98badcfeU,
                                      0x10325476U};
  std::array<unsigned char, 64> buffer_{};
  std::size_t buffered_ = 0U;
  std::uint64_t bytes_ = 0U;
};

bool lower_hex(std::string_view value, std::size_t size) {
  return value.size() == size &&
         std::all_of(value.begin(), value.end(), [](unsigned char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

void validate_contract(const detail::CifarContract& contract) {
  if (contract.archive_bytes == 0U || contract.archive_bytes > kMaxCompressed ||
      !lower_hex(contract.sha256, 64U) ||
      !lower_hex(contract.official_md5, 32U) ||
      contract.max_decoded_bytes == 0U ||
      contract.max_decoded_bytes > kMaxDecoded ||
      contract.batches.size() < 2U || contract.batches.size() > 6U ||
      contract.train_samples_per_class == 0U ||
      contract.train_samples_per_class > 5000U ||
      contract.validation_samples_per_class == 0U ||
      contract.validation_samples_per_class > 1000U ||
      contract.source.size() > 4096U || contract.dataset_license.size() > 4096U)
    throw InvalidArgument("Invalid private CIFAR-10 contract bounds/identity");
  std::set<std::string> names;
  std::array<std::uint64_t, 2> counts{};
  for (const auto& batch : contract.batches) {
    const auto known = std::find(kBatchNames.begin(), kBatchNames.end(),
                                 std::string_view(batch.name));
    const bool validation = batch.name == "test_batch.bin";
    if (known == kBatchNames.end() || !names.insert(batch.name).second ||
        batch.samples == 0U || batch.samples > 10000U ||
        batch.split != kSplits[validation ? 1U : 0U])
      throw InvalidArgument("Invalid private CIFAR-10 batch inventory");
    counts[validation ? 1U : 0U] += batch.samples;
  }
  if (counts[0] != 10U * contract.train_samples_per_class ||
      counts[1] != 10U * contract.validation_samples_per_class)
    throw InvalidArgument("Invalid private CIFAR-10 class/sample totals");
}

Json verify_archive(const fs::path& archive,
                    const detail::CifarContract& contract) {
  detail::reject_links(archive);
  if (!fs::is_regular_file(archive) ||
      fs::file_size(archive) != contract.archive_bytes)
    throw InvalidArgument("CIFAR-10 archive size mismatch");
  std::ifstream stream(archive, std::ios::binary);
  if (!stream) throw IoError("Could not open CIFAR-10 archive");
  std::array<unsigned char, detail::kCodecChunkBytes> buffer{};
  Sha256 sha;
  Md5 md5;
  std::uint64_t total = 0U;
  while (stream) {
    stream.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(buffer.size()));
    const auto count = static_cast<std::size_t>(stream.gcount());
    if (count > contract.archive_bytes - total)
      throw InvalidArgument("CIFAR-10 archive grew while reading");
    const auto input = std::span(buffer).first(count);
    sha.update(std::as_bytes(input));
    md5.update(input);
    total += count;
  }
  if (!stream.eof()) throw IoError("Could not read complete CIFAR-10 archive");
  const auto sha_text = sha256_digest_hex(sha.finish());
  const auto md5_text = md5.finish();
  if (total != contract.archive_bytes || sha_text != contract.sha256 ||
      md5_text != contract.official_md5)
    throw InvalidArgument("CIFAR-10 archive SHA-256/official MD5 mismatch");
  return {{"bytes", total}, {"sha256", sha_text}, {"official_md5", md5_text}};
}

class TarStream final {
 public:
  TarStream(detail::GzipReader& source, std::uint64_t decoded_budget)
      : source_(source), budget_(decoded_budget + kTarOverheadBudget) {}

  void exact(std::span<unsigned char> output) {
    account(output.size());
    source_.exact(output);
  }

  void finish_zero_padding() {
    std::array<unsigned char, detail::kCodecChunkBytes> block{};
    for (;;) {
      const auto count = source_.read(block);
      if (count == 0U) break;
      account(count);
      if (!std::all_of(block.begin(), block.begin() + count,
                       [](unsigned char value) { return value == 0U; }))
        throw InvalidArgument("Nonzero data after CIFAR-10 tar terminator");
    }
    source_.finish();
  }

 private:
  void account(std::uint64_t count) {
    if (count > budget_ - read_)
      throw ResourceExhausted("CIFAR-10 inflated tar byte budget exceeded");
    read_ += count;
  }
  detail::GzipReader& source_;
  std::uint64_t budget_;
  std::uint64_t read_ = 0U;
};

std::string tar_string(std::span<const unsigned char> field) {
  const auto end = std::find(field.begin(), field.end(), 0U);
  if (!std::all_of(end, field.end(),
                   [](unsigned char value) { return value == 0U; }))
    throw InvalidArgument("Malformed CIFAR-10 tar text field");
  return {reinterpret_cast<const char*>(field.data()),
          static_cast<std::size_t>(end - field.begin())};
}

std::uint64_t octal(std::span<const unsigned char> field) {
  std::size_t cursor = 0U;
  while (cursor < field.size() && field[cursor] == ' ') ++cursor;
  std::uint64_t value = 0U;
  while (cursor < field.size() && field[cursor] >= '0' &&
         field[cursor] <= '7') {
    const auto digit = static_cast<unsigned>(field[cursor++] - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 8U)
      throw InvalidArgument("CIFAR-10 tar integer overflow");
    value = value * 8U + digit;
  }
  for (; cursor < field.size(); ++cursor)
    if (field[cursor] != 0U && field[cursor] != ' ')
      throw InvalidArgument("Unsupported CIFAR-10 tar numeric encoding");
  return value;
}

struct TarMember {
  std::string key;
  std::uint64_t size;
  bool root;
};

TarMember member(const std::array<unsigned char, 512>& header) {
  const auto field = std::span(header);
  std::uint64_t checksum = 0U;
  for (std::size_t i = 0U; i < header.size(); ++i)
    checksum += i >= 148U && i < 156U ? 32U : header[i];
  if (octal(field.subspan(148U, 8U)) != checksum)
    throw InvalidArgument("CIFAR-10 tar header checksum mismatch");
  const auto magic = tar_string(field.subspan(257U, 6U));
  if (!magic.empty() && magic != "ustar" && magic != "ustar ")
    throw InvalidArgument("Unsupported CIFAR-10 tar header format");
  auto name = tar_string(field.first(100U));
  // GNU's old "ustar " layout reuses this region for atime/ctime/sparse
  // bookkeeping, rather than the POSIX prefix field. Long/sparse extension
  // members are rejected below; the fixed names already fit in 100 bytes.
  if (magic == "ustar") {
    const auto prefix = tar_string(field.subspan(345U, 155U));
    if (!prefix.empty()) name = prefix + "/" + name;
  }
  if (name.empty() || name.front() == '/' || name.find('\\') != name.npos ||
      name.find("..") != name.npos ||
      !tar_string(field.subspan(157U, 100U)).empty())
    throw InvalidArgument("Unsafe CIFAR-10 tar member path/link");
  const auto size = octal(field.subspan(124U, 12U));
  const auto type = header[156U];
  if (name == kPrefix || name == std::string(kPrefix) + "/") {
    if (type != '5' || size != 0U)
      throw InvalidArgument("Invalid CIFAR-10 tar root entry");
    return {"<root>", size, true};
  }
  const auto directory = std::string(kPrefix) + "/";
  if (!name.starts_with(directory) ||
      name.find('/', directory.size()) != name.npos ||
      (type != '0' && type != 0U))
    throw InvalidArgument("Unexpected/non-file CIFAR-10 tar member");
  return {name.substr(directory.size()), size, false};
}

void metadata_mapping(std::span<const unsigned char> input) {
  std::string content(reinterpret_cast<const char*>(input.data()),
                      input.size());
  if (std::any_of(content.begin(), content.end(),
                  [](unsigned char c) { return c > 127U; }))
    throw InvalidArgument("Non-ASCII CIFAR-10 class metadata");
  while (!content.empty() && (content.back() == '\r' || content.back() == '\n'))
    content.pop_back();
  std::size_t start = 0U;
  for (const auto expected : kClasses) {
    const auto end = content.find_first_of("\r\n", start);
    const auto stop = end == content.npos ? content.size() : end;
    if (std::string_view(content).substr(start, stop - start) != expected)
      throw InvalidArgument("CIFAR-10 category mapping differs");
    start = stop;
    if (start < content.size() && content[start] == '\r') ++start;
    if (start < content.size() && content[start] == '\n') ++start;
  }
  if (start != content.size())
    throw InvalidArgument("CIFAR-10 category mapping differs");
}

std::string image_name(std::string_view batch, std::uint32_t number) {
  const auto index = std::to_string(number);
  return std::string(batch.substr(0U, batch.size() - 4U)) + "-" +
         std::string(5U - index.size(), '0') + index + ".png";
}

void write_png(const fs::path& path, std::span<const unsigned char> content) {
  if (detail::present(path))
    throw IoError("Refusing to replace a staged CIFAR-10 image");
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw IoError("Could not create staged CIFAR-10 image");
  output.write(reinterpret_cast<const char*>(content.data()),
               static_cast<std::streamsize>(content.size()));
  detail::close_checked(output);
}

Json convert(const fs::path& archive, const fs::path& staging,
             const detail::CifarContract& contract) {
  std::map<std::string, const detail::CifarBatch*> batches;
  for (const auto& batch : contract.batches)
    batches.emplace(batch.name, &batch);
  for (const auto split : kSplits)
    for (const auto category : kClasses)
      fs::create_directories(staging / split / category);
  detail::GzipReader gzip(archive, contract.archive_bytes, contract.sha256);
  TarStream tar(gzip, contract.max_decoded_bytes);
  std::set<std::string> seen;
  std::array<std::array<std::uint32_t, 10>, 2> counts{};
  Json payload_hashes = Json::object();
  std::uint64_t decoded = 0U;
  std::uint64_t png_bytes = 0U;
  for (;;) {
    std::array<unsigned char, 512> header{};
    tar.exact(header);
    if (std::all_of(header.begin(), header.end(),
                    [](unsigned char value) { return value == 0U; })) {
      tar.exact(header);
      if (!std::all_of(header.begin(), header.end(),
                       [](unsigned char value) { return value == 0U; }))
        throw InvalidArgument("Incomplete CIFAR-10 tar terminator");
      tar.finish_zero_padding();
      break;
    }
    const auto item = member(header);
    if (!seen.insert(item.key).second || seen.size() > 9U)
      throw InvalidArgument("Duplicate/excess CIFAR-10 tar member");
    if (item.root) continue;
    const auto batch = batches.find(item.key);
    if (batch == batches.end() && item.key != "batches.meta.txt" &&
        item.key != "readme.html")
      throw InvalidArgument("Unexpected CIFAR-10 tar member");
    const auto limit =
        batch == batches.end()
            ? static_cast<std::uint64_t>(kMetadataBytes)
            : static_cast<std::uint64_t>(batch->second->samples) * kRecordBytes;
    if (item.size > limit || (batch != batches.end() && item.size != limit))
      throw InvalidArgument("Invalid CIFAR-10 decoded member size");
    if (item.size > contract.max_decoded_bytes - decoded)
      throw ResourceExhausted("CIFAR-10 decoded byte limit exceeded");
    decoded += item.size;
    if (batch == batches.end()) {
      std::vector<unsigned char> content(static_cast<std::size_t>(item.size));
      tar.exact(content);
      if (item.key == "batches.meta.txt") metadata_mapping(content);
    } else {
      Sha256 hash;
      const auto& info = *batch->second;
      const std::size_t split = info.split == "validation" ? 1U : 0U;
      std::array<unsigned char, kRecordBytes> record{};
      std::array<unsigned char, 3U * kPixels> rgb{};
      for (std::uint32_t index = 0U; index < info.samples; ++index) {
        tar.exact(record);
        const auto label = record.front();
        if (label >= kClasses.size())
          throw InvalidArgument("CIFAR-10 record label is outside 0..9");
        hash.update(std::as_bytes(std::span(record)));
        ++counts[split][label];
        for (std::size_t channel = 0U; channel < 3U; ++channel)
          for (std::size_t pixel = 0U; pixel < kPixels; ++pixel)
            rgb[3U * pixel + channel] = record[1U + channel * kPixels + pixel];
        const auto png = detail::encode_rgb_png(rgb, 32U, 32U);
        if (png.size() > detail::kMaxDatasetPngBytes ||
            png.size() > kMaxOutput - png_bytes)
          throw ResourceExhausted(
              "CIFAR-10 encoded output byte limit exceeded");
        png_bytes += png.size();
        write_png(staging / kSplits[split] / kClasses[label] /
                      image_name(item.key, index),
                  png);
      }
      payload_hashes[item.key] = sha256_digest_hex(hash.finish());
    }
    const auto padding = (512U - item.size % 512U) % 512U;
    std::array<unsigned char, 512> pad{};
    tar.exact(std::span(pad).first(static_cast<std::size_t>(padding)));
    if (!std::all_of(pad.begin(), pad.begin() + padding,
                     [](unsigned char value) { return value == 0U; }))
      throw InvalidArgument("Nonzero CIFAR-10 tar member padding");
  }
  if (!seen.contains("batches.meta.txt"))
    throw InvalidArgument("CIFAR-10 archive is missing class metadata");
  for (const auto& batch : contract.batches)
    if (!seen.contains(batch.name))
      throw InvalidArgument("CIFAR-10 archive is missing required batch");
  for (std::size_t split = 0U; split < counts.size(); ++split) {
    const auto expected = split == 0U ? contract.train_samples_per_class
                                      : contract.validation_samples_per_class;
    if (!std::all_of(
            counts[split].begin(), counts[split].end(),
            [expected](std::uint32_t count) { return count == expected; }))
      throw InvalidArgument("CIFAR-10 per-class split counts differ");
  }
  return {{"classes", kClasses},
          {"class_counts", {{"train", counts[0]}, {"validation", counts[1]}}},
          {"batch_sha256", std::move(payload_hashes)},
          {"decoded_bytes", decoded}};
}
}  // namespace

const detail::CifarContract& detail::official_cifar10_contract() {
  // Exact recipes/image-classification/cifar10.dataset.json contract. The
  // installed command does not read a mutable checkout/runtime contract file.
  static const CifarContract value{
      170052171U,
      "c4a38c50a1bc5f3a1c5537f2155ab9d68f9f25eb1ed8d9ddda3db29a59bca1dd",
      "c32a1d4ab5d03f1284b67883e8d87530",
      "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
      "NOASSERTION; original CIFAR-10 dataset terms apply; images are not "
      "covered by this project's Apache-2.0 license",
      185000000U,
      5000U,
      1000U,
      {{"data_batch_1.bin", "train", 10000U},
       {"data_batch_2.bin", "train", 10000U},
       {"data_batch_3.bin", "train", 10000U},
       {"data_batch_4.bin", "train", 10000U},
       {"data_batch_5.bin", "train", 10000U},
       {"test_batch.bin", "validation", 10000U}}};
  return value;
}

std::string detail::cifar10_md5_hex(std::span<const unsigned char> bytes) {
  Md5 hash;
  hash.update(bytes);
  return hash.finish();
}

fs::path detail::prepare_cifar10_contract(const fs::path& archive,
                                          const fs::path& data_root,
                                          const CifarContract& contract) {
  try {
    validate_contract(contract);
    if (archive.empty()) throw InvalidArgument("CIFAR-10 archive is required");
    const auto absolute = fs::absolute(archive);
    reject_links(absolute);
    const auto source = absolute.lexically_normal();
    const auto root = external_root(data_root);
    if (present(root / kDestination))
      throw IoError("Refusing to replace prepared CIFAR-10 dataset");
    const auto identity = verify_archive(source, contract);
    StagingDirectory transaction(root, kDestination, ".cifar10-preparing-");
    auto report = convert(source, transaction.path(), contract);
    report["schema_version"] = 1U;
    report["dataset"] = "CIFAR-10 binary version";
    report["source"] = contract.source;
    report["archive"] = identity;
    report["dataset_license"] = contract.dataset_license;
    report["pixels"] = "32x32 RGB uint8; exact planar-to-PNG conversion";
    report["split_policy"] =
        "complete source train/test; test is named validation; no subset";
    // Match the previous Python indent/order/terminal-newline representation.
    // Text mode preserves that reference's host-specific LF/CRLF convention.
    std::ofstream output(transaction.path() / "preparation.json");
    if (!output) throw IoError("Could not create CIFAR-10 preparation report");
    output << report.dump(2, ' ', false) << '\n';
    close_checked(output);
    const auto destination = transaction.destination();
    transaction.publish();
    return destination;
  } catch (const Error&) {
    throw;
  } catch (const fs::filesystem_error& error) {
    throw IoError(std::string("CIFAR-10 filesystem operation failed: ") +
                  error.what());
  }
}

fs::path prepare_cifar10(const fs::path& archive, const fs::path& data_root) {
  return detail::prepare_cifar10_contract(archive, data_root,
                                          detail::official_cifar10_contract());
}
}  // namespace jyppx::liuding::datasets
