// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_FOUNDATION_ARTIFACT_HPP
#define JYPPX_LIUDING_FOUNDATION_ARTIFACT_HPP
/**
 * @file artifact.hpp
 * @brief Defines storage-neutral immutable artifact identities and references.
 */

#include "jyppx/liuding/foundation/error.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

namespace jyppx::liuding {

/** @brief The digest algorithm used by an immutable artifact identity. */
enum class ArtifactDigestAlgorithm : std::uint8_t {
  sha256 = 1,  ///< SHA-256 encoded as 64 lowercase hexadecimal characters.
};

/** @brief A storage-neutral digest identity without ownership of the bytes. */
struct ArtifactDigest {
  ArtifactDigestAlgorithm algorithm =
      ArtifactDigestAlgorithm::sha256;  ///< Algorithm defining the identity.
  std::string hex;  ///< Canonical lowercase hexadecimal digest.
};

/**
 * @brief Compares two immutable content identities.
 * @param left First digest identity.
 * @param right Second digest identity.
 * @return True when algorithm and hexadecimal content identity match.
 */
inline bool operator==(const ArtifactDigest& left,
                       const ArtifactDigest& right) noexcept {
  return left.algorithm == right.algorithm && left.hex == right.hex;
}

/**
 * @brief Returns a stable lowercase digest algorithm name.
 * @param algorithm Algorithm to render.
 * @return Stable name, or `unknown` for an invalid enum value.
 */
inline std::string_view artifact_digest_algorithm_name(
    ArtifactDigestAlgorithm algorithm) noexcept {
  if (algorithm == ArtifactDigestAlgorithm::sha256) return "sha256";
  return "unknown";
}

/**
 * @brief Validates a digest without touching storage.
 * @param digest Content identity to validate.
 * @throws InvalidArgument If the algorithm or lowercase hex is invalid.
 */
inline void validate_artifact_digest(const ArtifactDigest& digest) {
  const bool valid =
      digest.algorithm == ArtifactDigestAlgorithm::sha256 &&
      digest.hex.size() == 64U &&
      std::all_of(digest.hex.begin(), digest.hex.end(), [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
      });
  if (!valid) {
    throw InvalidArgument(
        "Invalid artifact digest identity",
        ErrorDetails{.component = "foundation.artifact",
                     .operation = "digest",
                     .expected = "sha256 and 64 lowercase hex characters",
                     .actual = std::string(artifact_digest_algorithm_name(
                                   digest.algorithm)) +
                               ":" + digest.hex,
                     .hint = "Use the complete canonical SHA-256 digest."});
  }
}

/**
 * @brief Metadata reference to one immutable content-addressed artifact.
 *
 * The fields describe content and provenance only.  A reference never embeds
 * a storage path, network credential, or executable content.
 */
struct ArtifactRef {
  ArtifactDigest digest;         ///< Identity of the exact stored bytes.
  std::uint64_t size_bytes = 0;  ///< Unencoded blob byte count.
  std::string logical_name;      ///< Producer-facing stable display name.
  std::string media_type =
      "application/octet-stream";  ///< Stored content type.
  std::string producer;            ///< Component, recipe, or producer identity.
};

}  // namespace jyppx::liuding

#endif  // JYPPX_LIUDING_FOUNDATION_ARTIFACT_HPP