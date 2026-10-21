// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_FOUNDATION_SHA256_HPP
#define JYPPX_FOUNDATION_SHA256_HPP

/** @file sha256.hpp
 * @brief Computes SHA-256 identities for storage-neutral byte streams.
 */

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace jyppx {

/** @brief Standard 256-bit digest in network byte order. */
using Sha256Digest = std::array<std::byte, 32>;

/**
 * @brief Incrementally hashes bytes using FIPS 180-4 SHA-256.
 *
 * Instances are independent and not thread-safe. Reading the digest does not
 * close or mutate the stream, so additional updates and repeated reads are
 * allowed. This primitive computes content identities; it does not authenticate
 * an untrusted producer.
 */
class Sha256 final {
 public:
  /** @brief Creates the initial empty-stream state. */
  Sha256();
  /** @brief Releases private hashing state. */
  ~Sha256();

  /** @brief Hash streams cannot be copied. */
  Sha256(const Sha256&) = delete;
  /** @brief Hash streams cannot be copy assigned. */
  Sha256& operator=(const Sha256&) = delete;
  /** @brief Hash streams retain a stable implementation. */
  Sha256(Sha256&&) = delete;
  /** @brief Hash streams cannot be move assigned. */
  Sha256& operator=(Sha256&&) = delete;

  /**
   * @brief Consumes bytes before returning without retaining their storage.
   * @param bytes Next range in stream order, possibly empty.
   */
  void update(std::span<const std::byte> bytes) noexcept;

  /** @brief Returns the digest of all bytes consumed so far.
   * @return SHA-256 in network byte order without changing the stream.
   */
  Sha256Digest finish() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Private standard hash state.
};

/**
 * @brief Hashes a complete in-memory byte range.
 * @param bytes Complete content to hash.
 * @return SHA-256 in network byte order.
 */
Sha256Digest sha256(std::span<const std::byte> bytes);

/**
 * @brief Encodes a digest as 64 lowercase hexadecimal characters.
 * @param digest Raw SHA-256 digest.
 * @return Canonical lowercase hexadecimal content identity.
 */
std::string sha256_digest_hex(const Sha256Digest& digest);

/**
 * @brief Hashes a complete byte range and encodes the digest as lowercase hex.
 * @param bytes Complete content to hash.
 * @return 64 lowercase hexadecimal characters.
 */
std::string sha256_hex(std::span<const std::byte> bytes);

}  // namespace jyppx

#endif  // JYPPX_FOUNDATION_SHA256_HPP
