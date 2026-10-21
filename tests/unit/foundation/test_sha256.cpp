// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/sha256.hpp>

#include <gtest/gtest.h>
#include <string>
#include <string_view>

namespace {
std::span<const std::byte> as_bytes(std::string_view value) {
  return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

TEST(Sha256Test, MatchesPublishedFips180Vectors) {
  EXPECT_EQ(jyppx::sha256_hex({}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(jyppx::sha256_hex(as_bytes("abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(jyppx::sha256_hex(as_bytes(
                "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string million_a(1000000U, 'a');
  EXPECT_EQ(jyppx::sha256_hex(as_bytes(million_a)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST(Sha256Test, ChunkingAndDigestReadsPreserveStreamState) {
  const std::string text(257U, 'a');
  for (const auto chunk : {1U, 55U, 56U, 63U, 64U, 65U}) {
    jyppx::Sha256 hash;
    hash.update({});
    EXPECT_EQ(hash.finish(), jyppx::sha256({}));
    for (std::size_t offset = 0; offset < text.size(); offset += chunk) {
      hash.update(as_bytes(std::string_view(text).substr(offset, chunk)));
      EXPECT_EQ(hash.finish(), hash.finish());
    }
    EXPECT_EQ(jyppx::sha256_digest_hex(hash.finish()),
              jyppx::sha256_hex(as_bytes(text)));
    hash.update(as_bytes("b"));
    EXPECT_EQ(hash.finish(), jyppx::sha256(as_bytes(text + "b")));
  }
}
}  // namespace
