// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::span<const std::byte> as_bytes(std::string_view value) {
  return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

TEST(Sha256Test, MatchesPublishedFips180Vectors) {
  EXPECT_EQ(jyppx::liuding::sha256_hex({}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(jyppx::liuding::sha256_hex(as_bytes("abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(jyppx::liuding::sha256_hex(as_bytes(
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string million_a(1000000U, 'a');
  EXPECT_EQ(jyppx::liuding::sha256_hex(as_bytes(million_a)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, ChunkingAndDigestReadsPreserveStreamState) {
  const std::string text(257U, 'a');
  for (const auto chunk : {1U, 55U, 56U, 63U, 64U, 65U}) {
    jyppx::liuding::Sha256 hash;
    hash.update({});
    EXPECT_EQ(hash.finish(), jyppx::liuding::sha256({}));
    for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
      hash.update(as_bytes(std::string_view(text).substr(offset, chunk)));
      EXPECT_EQ(hash.finish(), hash.finish());
    }
    EXPECT_EQ(jyppx::liuding::sha256_digest_hex(hash.finish()),
              jyppx::liuding::sha256_hex(as_bytes(text)));
    hash.update(as_bytes("b"));
    EXPECT_EQ(hash.finish(), jyppx::liuding::sha256(as_bytes(text + "b")));
  }
}

TEST(Sha256Test, UnalignedBinaryBlocksMatchIndependentBoundaryDigests) {
  // Independently computed with Python hashlib.sha256, including both padding
  // lengths, complete blocks, partial blocks and more than one input chunk.
  struct Vector {
    std::size_t length;
    std::string_view digest;
  };
  constexpr std::array vectors{
      Vector{
          55U,
          "c0b4484205bbdcec7ad076bf261620188fe3f57a33cfb211346e006791ad3905"},
      Vector{
          56U,
          "9be85a77dcb027693d23f0a603bf7a774c46a4f6e339d02930e94c2653532ae9"},
      Vector{
          63U,
          "82b27b66c2444a8a236bf8e70b3bcc9207ab5399d5be93e303a450e3cb659665"},
      Vector{
          64U,
          "306cee4d6392e9b04fa5a0e951be8d5c33d7f5e51a9d3ff3a1d67ae56b104dd8"},
      Vector{
          65U,
          "3d1040d0d845d05dc551bbc71ea82264bd33c96f42c13115ca52a3aa4fd71b45"},
      Vector{
          127U,
          "cbe91224d9e85b9997b4bd0d5b71647f45f9efc6c86291c02cdb0af8ddcd4696"},
      Vector{
          128U,
          "ae54f933e8775e711171edec7ca2858ffbcc50ad9f5fe80ac68da84aff8fdfaa"},
      Vector{
          129U,
          "8a336ff92189b02c1992976e67b5e4dba9f03a76def3379161422bb364c510f0"},
      Vector{
          65537U,
          "c4b891c38e83063a13f01e9e9c5389ebf19ff597b0c77591b55a5c7214fb9484"}};
  std::vector<std::byte> storage(65539U);
  for (std::size_t index = 0; index < storage.size(); ++index) {
    storage[index] =
        static_cast<std::byte>((index * 131U + index / 251U + 17U) & 255U);
  }
  for (const auto& [length, expected] : vectors) {
    const auto input = std::span<const std::byte>(storage).subspan(1U, length);
    EXPECT_EQ(jyppx::liuding::sha256_hex(input), expected);
    for (const auto chunk : {1U, 7U, 63U, 64U, 65U, 4093U, 65536U}) {
      SCOPED_TRACE(::testing::Message()
                   << "length=" << length << " chunk=" << chunk);
      jyppx::liuding::Sha256 hash;
      for (std::size_t offset = 0; offset < length; offset += chunk) {
        hash.update(input.subspan(
            offset, std::min<std::size_t>(chunk, length - offset)));
        (void)hash.finish();
        hash.update({});
      }
      EXPECT_EQ(jyppx::liuding::sha256_digest_hex(hash.finish()), expected);
    }
  }
}

TEST(Sha256Test, UpdatesConsumeFullBlocksAndTailBeforeCallerReusesStorage) {
  std::array<std::byte, 130> storage{};
  for (std::size_t index = 0; index < storage.size(); ++index) {
    storage[index] = static_cast<std::byte>((index * 131U + 17U) & 255U);
  }
  const auto input = std::span<const std::byte>(storage).subspan(1U);
  jyppx::liuding::Sha256 hash;
  hash.update(input.first(65U));
  std::fill_n(storage.begin() + 1, 65U, std::byte{0});
  hash.update(input.subspan(65U));
  storage.fill(std::byte{0});
  EXPECT_EQ(jyppx::liuding::sha256_digest_hex(hash.finish()),
            "8a336ff92189b02c1992976e67b5e4dba9f03a76def3379161422bb364c510f0");
}
}  // namespace
