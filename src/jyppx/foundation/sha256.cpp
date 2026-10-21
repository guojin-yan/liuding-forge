// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <jyppx/foundation/sha256.hpp>

#include <cstdint>

namespace jyppx {
class Sha256::Impl final {
 public:
  Impl()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(std::span<const std::byte> data) {
    for (const auto byte : data) {
      buffer_[buffer_size_++] = static_cast<std::uint8_t>(byte);
      if (buffer_size_ == buffer_.size()) {
        transform();
        buffer_size_ = 0;
      }
      bit_count_ += 8U;
    }
  }

  std::array<std::byte, 32> finish() {
    const auto original_bits = bit_count_;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56U) {
      while (buffer_size_ < 64U) buffer_[buffer_size_++] = 0U;
      transform();
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56U) buffer_[buffer_size_++] = 0U;
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[buffer_size_++] =
          static_cast<std::uint8_t>(original_bits >> shift);
    }
    transform();

    std::array<std::byte, 32> digest{};
    for (std::size_t index = 0; index != state_.size(); ++index) {
      for (unsigned shift = 0; shift != 32; shift += 8) {
        digest[index * 4U + shift / 8U] =
            static_cast<std::byte>(state_[index] >> (24U - shift));
      }
    }
    return digest;
  }

 private:
  static constexpr std::array<std::uint32_t, 64> kRoundConstants{
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

  static std::uint32_t rotate(std::uint32_t value, unsigned count) noexcept {
    return (value >> count) | (value << (32U - count));
  }

  void transform() {
    std::array<std::uint32_t, 64> schedule{};
    for (std::size_t index = 0; index != 16U; ++index) {
      schedule[index] =
          (static_cast<std::uint32_t>(buffer_[index * 4U]) << 24U) |
          (static_cast<std::uint32_t>(buffer_[index * 4U + 1U]) << 16U) |
          (static_cast<std::uint32_t>(buffer_[index * 4U + 2U]) << 8U) |
          static_cast<std::uint32_t>(buffer_[index * 4U + 3U]);
    }
    for (std::size_t index = 16U; index != schedule.size(); ++index) {
      const auto s0 = rotate(schedule[index - 15U], 7U) ^
                      rotate(schedule[index - 15U], 18U) ^
                      (schedule[index - 15U] >> 3U);
      const auto s1 = rotate(schedule[index - 2U], 17U) ^
                      rotate(schedule[index - 2U], 19U) ^
                      (schedule[index - 2U] >> 10U);
      schedule[index] = schedule[index - 16U] + s0 + schedule[index - 7U] + s1;
    }
    auto working = state_;
    for (std::size_t index = 0; index != schedule.size(); ++index) {
      const auto s1 = rotate(working[4], 6U) ^ rotate(working[4], 11U) ^
                      rotate(working[4], 25U);
      const auto choice =
          (working[4] & working[5]) ^ (~working[4] & working[6]);
      const auto temp1 =
          working[7] + s1 + choice + kRoundConstants[index] + schedule[index];
      const auto s0 = rotate(working[0], 2U) ^ rotate(working[0], 13U) ^
                      rotate(working[0], 22U);
      const auto majority = (working[0] & working[1]) ^
                            (working[0] & working[2]) ^
                            (working[1] & working[2]);
      const auto temp2 = s0 + majority;
      working = {temp1 + temp2,      working[0], working[1], working[2],
                 working[3] + temp1, working[4], working[5], working[6]};
    }
    for (std::size_t index = 0; index != state_.size(); ++index) {
      state_[index] += working[index];
    }
  }

  std::array<std::uint32_t, 8> state_{};
  std::array<std::uint8_t, 64> buffer_{};
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
}  // namespace jyppx
