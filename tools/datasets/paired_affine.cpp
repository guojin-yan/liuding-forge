// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "paired_affine.hpp"

#include "preparation_support.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <string_view>

namespace jyppx::liuding::datasets {
namespace {
namespace fs = std::filesystem;
using detail::close_checked;
using Json = nlohmann::json;
constexpr std::string_view kDataset = "jyppx.liuding.paired-affine/v1";
constexpr std::uint64_t kSeed = 20261028U;
constexpr std::size_t kMaxRecordBytes = 512U;
constexpr std::uint64_t kMaxDataBytes = 1024U * 1024U;

std::span<const std::byte> bytes(std::string_view text) {
  return std::as_bytes(std::span(text.data(), text.size()));
}

// Exact terminating decimal for numerator / 2^n. Python's original JSON
// encoding writes integral floating values as x.0; keep that frozen spelling.
// Integer arithmetic makes the bytes independent of locale and float printers.
std::string dyadic(int numerator, int denominator) {
  const bool negative = numerator < 0;
  auto magnitude = negative ? -numerator : numerator;
  std::string result = negative ? "-" : "";
  result += std::to_string(magnitude / denominator);
  result += '.';
  auto remainder = magnitude % denominator;
  do {
    remainder *= 10;
    result += static_cast<char>('0' + remainder / denominator);
    remainder %= denominator;
  } while (remainder != 0);
  return result;
}

struct Record {
  std::string content;
  std::size_t class_id;
};

Record record(std::string_view split, std::size_t index) {
  const auto key = std::string(kDataset) + ":" + std::to_string(kSeed) + ":" +
                   std::string(split) + ":" + std::to_string(index);
  const auto digest = sha256(bytes(key));
  std::array<int, 4> values{};
  for (std::size_t i = 0U; i < values.size(); ++i) {
    const auto word = std::to_integer<unsigned>(digest[2U * i]) |
                      (std::to_integer<unsigned>(digest[2U * i + 1U]) << 8U);
    values[i] = static_cast<int>(word % 2049U) - 1024;
  }
  const auto [a, b, c, d] = values;
  const auto label = static_cast<std::size_t>(4 * a + 2 * b + 4 * c - d >= 0);
  const auto number = std::to_string(index);
  const auto id =
      std::string(split) + ":" + std::string(6U - number.size(), '0') + number;
  // Lexicographic keys and compact separators match the frozen JSONL contract.
  return {"{\"class_id\":" + std::to_string(label) + ",\"id\":\"" + id +
              "\",\"left\":[" + dyadic(a, 1024) + "," + dyadic(b, 1024) +
              "],\"regression\":[" +
              dyadic(4 * a - 2 * b + 3 * c + d + 512, 8192) + "],\"right\":[" +
              dyadic(c, 1024) + "," + dyadic(d, 1024) + "]}\n",
          label};
}

Json write_split(const fs::path& staging, std::string_view split,
                 std::size_t count) {
  const auto filename = std::string(split) + ".jsonl";
  std::ofstream stream(staging / filename, std::ios::binary | std::ios::trunc);
  if (!stream) throw IoError("Could not open dataset staging file");
  Sha256 hash;
  std::uint64_t total = 0U;
  std::size_t largest = 0U;
  std::array<std::size_t, 2> classes{};
  for (std::size_t index = 0U; index < count; ++index) {
    const auto row = record(split, index);
    if (row.content.size() > kMaxRecordBytes ||
        row.content.size() > kMaxDataBytes - total)
      throw ResourceExhausted("Paired-affine data exceeds its byte limits");
    stream.write(row.content.data(),
                 static_cast<std::streamsize>(row.content.size()));
    if (!stream) throw IoError("Could not write complete dataset record");
    hash.update(bytes(row.content));
    total += row.content.size();
    largest = std::max(largest, row.content.size());
    ++classes[row.class_id];
  }
  close_checked(stream);
  return {
      {"file", filename},        {"samples", count},
      {"bytes", total},          {"sha256", sha256_digest_hex(hash.finish())},
      {"class_counts", classes}, {"max_record_bytes_observed", largest}};
}

Json expected_splits() {
  // Frozen by recipes/multi-io/paired-affine.dataset.json. The installed
  // preparation command needs no source checkout or runtime contract file.
  return {
      {"train",
       {{"file", "train.jsonl"},
        {"samples", 1024U},
        {"bytes", 137615U},
        {"sha256",
         "781c7f9fa4a0c7dd8418f13714e2318fc35f9a22d65d240649e1784195b59d81"},
        {"class_counts", {481U, 543U}},
        {"max_record_bytes_observed", 141U}}},
      {"validation",
       {{"file", "validation.jsonl"},
        {"samples", 256U},
        {"bytes", 35760U},
        {"sha256",
         "3155d7de4ff24ad1a8fb7f58e5d640f8b37e1fed43fabae6ca35097cb5368bf7"},
        {"class_counts", {123U, 133U}},
        {"max_record_bytes_observed", 146U}}}};
}

}  // namespace

fs::path prepare_paired_affine(const fs::path& data_root) {
  try {
    const auto root = detail::external_root(data_root);
    detail::StagingDirectory transaction(root, "paired-affine-v1",
                                         ".paired-affine-preparing-");
    const auto& staging = transaction.path();
    Json splits = Json::object();
    const auto expected = expected_splits();
    std::uint64_t total = 0U;
    for (const auto split : {"train", "validation"}) {
      const auto generated = write_split(
          staging, split, expected.at(split).at("samples").get<std::size_t>());
      if (generated != expected.at(split))
        throw CompatibilityError(
            "Paired-affine generated content identity mismatch");
      const auto size = generated.at("bytes").get<std::uint64_t>();
      if (size > kMaxDataBytes - total)
        throw ResourceExhausted(
            "Paired-affine total data exceeds its byte limit");
      total += size;
      splits[split] = generated;
    }
    const Json manifest{
        {"schema_id", "jyppx.liuding.paired-affine.preparation"},
        {"schema_version", 1U},
        {"dataset_id", kDataset},
        {"seed", kSeed},
        {"dataset_license", "Apache-2.0; self-authored synthetic example data"},
        {"source",
         "self-authored deterministic paired-affine example; no external "
         "data"},
        {"files", splits},
        {"max_record_bytes", kMaxRecordBytes},
        {"max_data_bytes", kMaxDataBytes},
        {"data_bytes", total}};
    const auto content = manifest.dump(2) + "\n";
    std::ofstream output(staging / "preparation.json",
                         std::ios::binary | std::ios::trunc);
    if (!output) throw IoError("Could not open dataset preparation manifest");
    output.write(content.data(), static_cast<std::streamsize>(content.size()));
    close_checked(output);
    transaction.publish();
    return transaction.destination();
  } catch (const fs::filesystem_error&) {
    throw IoError("Dataset preparation filesystem operation failed");
  }
}
}  // namespace jyppx::liuding::datasets
