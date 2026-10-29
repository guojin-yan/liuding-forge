// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_SAMPLE_RANDOM_HPP
#define JYPPX_LIUDING_DATA_IMAGE_SAMPLE_RANDOM_HPP
/** @file sample_random.hpp
 * @brief Stateless, versioned randomness keyed by logical sample identity.
 */

#include <jyppx/liuding/foundation/sha256.hpp>

#include <cstdint>
#include <string>

namespace jyppx::liuding::data::image {

/** @brief Complete logical identity of one sample's augmentation stream. */
struct SampleRandomKey {
  std::string dataset_id;    ///< Frozen dataset/content/configuration identity.
  std::string sample_id;     ///< Stable dataset-local ID, never a worker ID.
  std::string transform_id;  ///< Versioned transform/configuration identity.
  std::uint64_t seed = 0;    ///< User-selected experiment seed.
  std::uint64_t epoch = 0;   ///< Logical epoch selected by the coordinator.
  std::uint64_t occurrence = 0;  ///< Repetition of this sample in the epoch.
  std::uint64_t view = 0;        ///< Explicit independent augmentation view.
  std::uint32_t algorithm_version = 1;  ///< Only SHA-256 counter v1 is defined.
};

/**
 * @brief Immutable counter stream, independent of thread and global torch RNG.
 *
 * Strings are nonempty opaque bytes, each bounded to 4096 bytes. Version 1
 * hashes a domain, length-prefixed strings and big-endian integers. Draws hash
 * the key digest with a separate domain, draw index and rejection counter;
 * the first eight digest bytes are a big-endian uint64. This is a reproducible
 * sampling primitive, not a secret-key or authentication API.
 * Concurrent draws share no mutable cursor or global RNG state.
 */
class SampleRandom final {
 public:
  /** @brief Freezes a validated logical identity.
   * @param key Dataset, sample, transform and stream coordinates.
   * @throws InvalidArgument Missing/oversized identity or unsupported version.
   */
  explicit SampleRandom(const SampleRandomKey& key);

  /** @brief Returns a reproducible full-width unsigned draw.
   * @param draw Logical draw index; repeated requests return the same value.
   * @return Unsigned 64-bit sample for rejection counter zero.
   */
  std::uint64_t bits(std::uint64_t draw) const;

  /** @brief Samples [0, upper_exclusive) using unbiased rejection sampling.
   * @param upper_exclusive Positive upper bound, including UINT64_MAX.
   * @param draw Logical draw index, independent of other draws' rejections.
   * @return Uniform integer strictly below upper_exclusive.
   * @throws InvalidArgument The bound is zero.
   * @throws ResourceExhausted All 128 bounded rejection attempts fail.
   */
  std::uint64_t bounded(std::uint64_t upper_exclusive,
                        std::uint64_t draw) const;

  /** @brief Returns the upper 53 bits scaled by 2^-53, in [0,1).
   * @param draw Logical draw index.
   * @return Reproducible double in the half-open unit interval.
   */
  double unit(std::uint64_t draw) const;

  /** @brief Returns the SHA-256 key fingerprint for provenance.
   * @return Lowercase hex; bind transform/source configuration separately too.
   */
  std::string identity() const;

 private:
  /** @brief Hashes one draw and rejection coordinate without cursor mutation.
   * @param draw Logical draw index.
   * @param retry Rejection counter belonging only to this draw.
   * @return First eight digest bytes interpreted as a big-endian uint64.
   */
  std::uint64_t attempt(std::uint64_t draw, std::uint64_t retry) const;
  Sha256Digest key_;  ///< Frozen key digest; no mutable sampling cursor.
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_SAMPLE_RANDOM_HPP
