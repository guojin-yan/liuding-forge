// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/liuding/foundation/sha256.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace jyppx::liuding {
class Sha256::Impl final {
 public:
  Impl()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(std::span<const std::byte> data) {
    if (data.empty()) return;
    bit_count_ += static_cast<std::uint64_t>(data.size()) * 8U;
    const auto* next = data.data();
    auto remaining = data.size();
    if (buffer_size_ != 0U) {
      const auto copied = std::min(64U - buffer_size_, remaining);
      std::memcpy(buffer_ + buffer_size_, next, copied);
      buffer_size_ += copied;
      next += copied;
      remaining -= copied;
      if (buffer_size_ != 64U) return;
      transform(buffer_);
      buffer_size_ = 0;
    }
    while (remaining >= 64U) {
      transform(next);
      next += 64U;
      remaining -= 64U;
    }
    if (remaining != 0U) {
      std::memcpy(buffer_, next, remaining);
      buffer_size_ = remaining;
    }
  }

  std::array<std::byte, 32> finish() {
    const auto original_bits = bit_count_;
    buffer_[buffer_size_++] = std::byte{0x80};
    if (buffer_size_ > 56U) {
      std::memset(buffer_ + buffer_size_, 0, 64U - buffer_size_);
      transform(buffer_);
      buffer_size_ = 0;
    }
    std::memset(buffer_ + buffer_size_, 0, 56U - buffer_size_);
    buffer_size_ = 56U;
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[buffer_size_++] = static_cast<std::byte>(original_bits >> shift);
    }
    transform(buffer_);

    std::array<std::byte, 32> digest{};
    for (std::size_t index = 0; index != 8U; ++index) {
      for (unsigned shift = 0; shift != 32; shift += 8) {
        digest[index * 4U + shift / 8U] =
            static_cast<std::byte>(state_[index] >> (24U - shift));
      }
    }
    return digest;
  }

 private:
  static constexpr std::uint32_t kRoundConstants[64]{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
      0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
      0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
      0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
      0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
      0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
      0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
      0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
      0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
      0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  void transform(const std::byte* block) {
    std::uint32_t schedule[64];
    for (std::size_t index = 0; index != 16U; ++index) {
      schedule[index] =
          (static_cast<std::uint32_t>(block[index * 4U]) << 24U) |
          (static_cast<std::uint32_t>(block[index * 4U + 1U]) << 16U) |
          (static_cast<std::uint32_t>(block[index * 4U + 2U]) << 8U) |
          static_cast<std::uint32_t>(block[index * 4U + 3U]);
    }
    for (std::size_t index = 16U; index != 64U; ++index) {
      const auto x = schedule[index - 15U];
      const auto y = schedule[index - 2U];
      const auto s0 =
          ((x >> 7U) | (x << 25U)) ^ ((x >> 18U) | (x << 14U)) ^ (x >> 3U);
      const auto s1 =
          ((y >> 17U) | (y << 15U)) ^ ((y >> 19U) | (y << 13U)) ^ (y >> 10U);
      schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
    }
    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0; index != 64U; ++index) {
      const auto s1 = ((e >> 6U) | (e << 26U)) ^ ((e >> 11U) | (e << 21U)) ^
                      ((e >> 25U) | (e << 7U));
      const auto choice = (e & f) ^ (~e & g);
      const auto temp1 =
          h + s1 + choice + kRoundConstants[index] + schedule[index];
      const auto s0 = ((a >> 2U) | (a << 30U)) ^ ((a >> 13U) | (a << 19U)) ^
                      ((a >> 22U) | (a << 10U));
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temp2 = s0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::uint32_t state_[8]{};
  std::byte buffer_[64]{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

Sha256::Sha256() : impl_(std::make_unique<Impl>()) {}
Sha256::~Sha256() = default;
void Sha256::update(std::span<const std::byte> bytes) noexcept {
  impl_->update(bytes);
}
Sha256Digest Sha256::finish() const noexcept {
  auto snapshot = *impl_;
  return snapshot.finish();
}
Sha256Digest sha256(std::span<const std::byte> bytes) {
  Sha256 hash;
  hash.update(bytes);
  return hash.finish();
}
std::string sha256_digest_hex(const Sha256Digest& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(64U);
  for (const auto value : digest) {
    const auto byte = static_cast<unsigned char>(value);
    hex.push_back(digits[byte >> 4U]);
    hex.push_back(digits[byte & 15U]);
  }
  return hex;
}
std::string sha256_hex(std::span<const std::byte> bytes) {
  return sha256_digest_hex(sha256(bytes));
}
}  // namespace jyppx::liuding
