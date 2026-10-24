// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/data_image/sample_random.hpp>
#include <jyppx/liuding/foundation/error.hpp>

#include <array>
#include <span>
#include <string_view>

namespace jyppx::liuding::data::image {
namespace {
void Bytes(Sha256& hash, std::string_view value) {
  hash.update(std::as_bytes(std::span(value.data(), value.size())));
}

void Integer(Sha256& hash, std::uint64_t value) {
  std::array<std::byte, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = static_cast<std::byte>(value >> (56U - 8U * i));
  hash.update(bytes);
}

void Identity(Sha256& hash, std::string_view value) {
  if (value.empty() || value.size() > 4096)
    throw InvalidArgument("Invalid sample random identity",
                          ErrorDetails{.component = "data.image.random",
                                       .operation = "random.key",
                                       .expected = "1..4096 identity bytes"});
  Integer(hash, value.size());
  Bytes(hash, value);
}
}  // namespace

SampleRandom::SampleRandom(const SampleRandomKey& key) {
  if (key.algorithm_version != 1)
    throw InvalidArgument("Unsupported sample random algorithm",
                          ErrorDetails{.component = "data.image.random",
                                       .operation = "random.version",
                                       .expected = "algorithm_version=1"});
  Sha256 hash;
  Bytes(hash, "liuding.sample-random.key.v1");
  Identity(hash, key.dataset_id);
  Identity(hash, key.sample_id);
  Identity(hash, key.transform_id);
  Integer(hash, key.seed);
  Integer(hash, key.epoch);
  Integer(hash, key.occurrence);
  Integer(hash, key.view);
  key_ = hash.finish();
}

std::uint64_t SampleRandom::attempt(std::uint64_t draw,
                                    std::uint64_t retry) const {
  Sha256 hash;
  Bytes(hash, "liuding.sample-random.draw.v1");
  hash.update(key_);
  Integer(hash, draw);
  Integer(hash, retry);
  std::uint64_t result = 0;
  const auto digest = hash.finish();
  for (const auto byte : std::span(digest).first<8>())
    result = (result << 8U) | std::to_integer<std::uint8_t>(byte);
  return result;
}

std::uint64_t SampleRandom::bits(std::uint64_t draw) const {
  return attempt(draw, 0);
}

std::uint64_t SampleRandom::bounded(std::uint64_t upper_exclusive,
                                    std::uint64_t draw) const {
  if (upper_exclusive == 0)
    throw InvalidArgument("Empty sample random range",
                          ErrorDetails{.component = "data.image.random",
                                       .operation = "random.bounded",
                                       .expected = "positive upper bound"});
  const auto threshold = (std::uint64_t{0} - upper_exclusive) % upper_exclusive;
  for (std::uint64_t retry = 0; retry < 128; ++retry) {
    const auto value = attempt(draw, retry);
    if (value >= threshold) return value % upper_exclusive;
  }
  throw ResourceExhausted("Sample random rejection budget exceeded",
                          ErrorDetails{.component = "data.image.random",
                                       .operation = "random.bounded"});
}

double SampleRandom::unit(std::uint64_t draw) const {
  return static_cast<double>(bits(draw) >> 11U) * 0x1.0p-53;
}

std::string SampleRandom::identity() const { return sha256_digest_hex(key_); }

}  // namespace jyppx::liuding::data::image
