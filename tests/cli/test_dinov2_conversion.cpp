// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include "dinov2_conversion.hpp"

#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/interop/safetensors.hpp>

#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace conversion = jyppx::liuding::tools::dinov2;
namespace st = jyppx::liuding::interop;

std::vector<std::byte> synthetic_source() {
  std::map<std::string, std::vector<std::uint64_t>> inventory;
  for (const auto& entry : conversion::mapping()) {
    auto shape = entry.shape;
    shape.front() /= entry.sources.size();
    for (const auto& key : entry.sources) inventory.emplace(key, shape);
  }
  inventory.emplace("embeddings.mask_token",
                    std::vector<std::uint64_t>{1, 384});
  std::uint64_t total = 0;
  for (const auto& [name, shape] : inventory) {
    std::uint64_t count = 4;
    for (const auto extent : shape) count *= extent;
    total += count;
  }
  std::vector<std::byte> storage(static_cast<std::size_t>(total));
  std::vector<st::SafeTensorSource> tensors;
  std::size_t offset = 0;
  unsigned marker = 0;
  for (const auto& [name, shape] : inventory) {
    std::size_t count = 4;
    for (const auto extent : shape) count *= static_cast<std::size_t>(extent);
    auto payload = std::span(storage).subspan(offset, count);
    // Distinct source identities make Q/K/V order and every mapping observable.
    for (std::size_t index = 0; index < count; ++index)
      payload[index] = static_cast<std::byte>((marker + index * 13U) % 256U);
    tensors.push_back({.name = name,
                       .dtype = st::SafeTensorDtype::f32,
                       .shape = shape,
                       .data = payload});
    offset += count;
    ++marker;
  }
  return st::serialize_safetensors(tensors);
}

TEST(DinoV2NativeConversion,
     FullSyntheticInventoryPreservesEveryPayloadAndQkvOrder) {
  auto source = synthetic_source();
  const auto encoded = conversion::encode_layout(source);
  ASSERT_EQ(encoded.size(), conversion::kOutputSize);
  std::uint64_t payload_offset = 0;
  {
    const st::SafeTensorsView before(source);
    const st::SafeTensorsView after(encoded);
    ASSERT_EQ(before.directory().tensors.size(), 223);
    ASSERT_EQ(after.directory().tensors.size(), 174);
    std::set<std::string> consumed{"embeddings.mask_token"};
    for (const auto& entry : conversion::mapping()) {
      const auto actual = after.tensor_bytes(entry.target);
      std::size_t offset = 0;
      for (const auto& key : entry.sources) {
        ASSERT_TRUE(consumed.insert(key).second);
        const auto expected = before.tensor_bytes(key);
        ASSERT_LE(offset + expected.size(), actual.size());
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(),
                               actual.begin() + offset))
            << entry.target << " <- " << key;
        offset += expected.size();
      }
      EXPECT_EQ(offset, actual.size());
    }
    EXPECT_EQ(consumed.size(), 223);
    // Fixed old-Python canonical grammar; catches nested descriptor key
    // sorting.
    const std::string first =
        "{\"blocks.0.attn.proj.bias\":{\"dtype\":\"F32\",\"shape\":[384],"
        "\"data_offsets\":[0,1536]},";
    ASSERT_GE(encoded.size(), 8 + first.size());
    EXPECT_TRUE(std::equal(first.begin(), first.end(),
                           reinterpret_cast<const char*>(encoded.data() + 8)));
    EXPECT_EQ(after.directory().payload_offset % 8, 0);
    EXPECT_THROW(after.tensor_bytes("embeddings.mask_token"),
                 jyppx::liuding::InvalidArgument);
    payload_offset = before.directory().payload_offset;
  }
  const auto replace_header = [&](const std::string& old,
                                  const std::string& replacement) {
    ASSERT_EQ(old.size(), replacement.size());
    auto* begin = reinterpret_cast<char*>(source.data() + 8);
    const auto end = reinterpret_cast<char*>(source.data() + payload_offset);
    const auto found = std::search(begin, end, old.begin(), old.end());
    ASSERT_NE(found, end);
    std::copy(replacement.begin(), replacement.end(), found);
  };
  // Valid alternate dtype/shape/name encodings must fail the fixed mapping,
  // rather than relying on malformed JSON or wrong payload extent alone.
  replace_header("F32", "U32");
  EXPECT_THROW(conversion::encode_layout(source), std::runtime_error);
  replace_header("U32", "F32");
  replace_header("[1536,384]", "[1152,512]");
  EXPECT_THROW(conversion::encode_layout(source), std::runtime_error);
  replace_header("[1152,512]", "[1536,384]");
  replace_header("embeddings.mask_token", "embeddings.mask_tokex");
  EXPECT_THROW(conversion::encode_layout(source), std::runtime_error);
}

TEST(DinoV2NativeConversion,
     RejectsIncompleteMalformedAndHeaderLimitBeforeLayout) {
  const auto empty = st::serialize_safetensors({});
  EXPECT_THROW(conversion::encode_layout(empty), std::runtime_error);
  auto malformed = empty;
  malformed[0] = std::byte{0xff};
  EXPECT_THROW(conversion::encode_layout(malformed), jyppx::liuding::Error);
  std::vector<std::byte> oversized(8, std::byte{0});
  // LE 1 MiB + 8 header, rejected before JSON parsing/large allocation.
  oversized[0] = std::byte{8};
  oversized[2] = std::byte{0x10};
  EXPECT_THROW(conversion::encode_layout(oversized), jyppx::liuding::Error);
}

TEST(DinoV2NativeConversion, MappingHasExactUniqueInventoryAndFixedIdentity) {
  const auto entries = conversion::mapping();
  ASSERT_EQ(entries.size(), 174);
  std::set<std::string> targets;
  std::set<std::string> sources;
  std::size_t concatenations = 0;
  for (const auto& entry : entries) {
    EXPECT_TRUE(targets.insert(entry.target).second);
    EXPECT_EQ(entry.shape.front() % entry.sources.size(), 0);
    for (const auto& source : entry.sources)
      EXPECT_TRUE(sources.insert(source).second);
    if (entry.sources.size() == 3)
      ++concatenations;
    else
      EXPECT_EQ(entry.sources.size(), 1);
  }
  EXPECT_EQ(sources.size(), 222);
  EXPECT_EQ(concatenations, 24);
  EXPECT_FALSE(sources.contains("embeddings.mask_token"));
  EXPECT_TRUE(std::is_sorted(
      entries.begin(), entries.end(),
      [](const auto& a, const auto& b) { return a.target < b.target; }));
  EXPECT_EQ(conversion::kOutputSha,
            "f3ef1ae01fd4af3cd15926a0d5e631fa0caea50ca68b29db8a68c8c84c9810e5");
}
}  // namespace
