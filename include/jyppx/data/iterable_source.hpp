// Copyright 2026 JYPPX Training Framework Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_DATA_ITERABLE_SOURCE_HPP
#define JYPPX_DATA_ITERABLE_SOURCE_HPP

/**
 * @file iterable_source.hpp
 * @brief Defines sequential source factories and isolated epoch cursors.
 */

#include <jyppx/foundation/checkpoint.hpp>

#include <ATen/core/ivalue.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::data {

/**
 * @brief Describes one value pulled from an iterable source.
 *
 * Sample IDs are stable within `dataset_version`. A watermark is an opaque
 * source position after this item; JYPPX stores and reports it but never
 * compares source-specific ordering. Exact and bounded sources must provide a
 * non-empty watermark for every item.
 */
struct IterableSample {
  c10::IValue value;            ///< Source value owned by the active cursor.
  std::uint64_t sample_id = 0;  ///< Stable identifier within the dataset.
  std::string watermark;        ///< Opaque position after this source item.
};

/**
 * @brief Freezes replay, finiteness, and checkpoint behavior for a source.
 */
struct IterableSourceDescriptor {
  std::string source_id;  ///< Stable type and immutable configuration identity.
  bool finite_epoch = true;  ///< Whether every opened epoch eventually ends.
  bool deterministic_epoch =
      false;  ///< Whether the same epoch and state reproduce source values.
  bool checkpointable =
      false;  ///< Whether cursor save/load state is supported.
  RestoreLevel restore_level =
      RestoreLevel::best_effort;  ///< Strongest continuation guarantee.
  std::optional<std::uint64_t>
      max_replayed_samples;  ///< Exact: zero; bounded: finite; best: absent.
  std::optional<std::uint64_t>
      max_omitted_samples;  ///< Exact: zero; bounded: finite; best: absent.
  std::vector<std::string>
      degradation_reasons;  ///< Required reasons for non-exact guarantees.
};

/**
 * @brief Owns mutable sequential state for exactly one logical epoch.
 *
 * A cursor is single-consumer. `next()` and `save_state()` run on the pipeline
 * producer thread. `close()` may run concurrently to unblock `next()` during
 * cancellation and must be idempotent and noexcept. After end-of-source,
 * repeated `next()` calls must continue returning empty.
 */
class IterableSourceCursor {
 public:
  /** @brief Enables destruction through the cursor interface. */
  virtual ~IterableSourceCursor() = default;

  /**
   * @brief Pulls the next source item.
   * @param stop Cooperative cancellation for the active epoch.
   * @return The next item, or empty after the epoch is exhausted.
   */
  virtual std::optional<IterableSample> next(std::stop_token stop) = 0;

  /**
   * @brief Serializes the position after all successfully pulled items.
   * @return Opaque bounded bytes understood by `load_state()`.
   * @throws NotSupported When the source descriptor is not checkpointable.
   */
  virtual std::string save_state() const = 0;

  /**
   * @brief Restores an opaque position before the cursor is consumed.
   * @param state Bytes produced by a compatible cursor's `save_state()`.
   * @throws CompatibilityError or SerializationError for invalid state.
   *
   * Implementations must validate before mutation. The pipeline calls this on
   * a newly opened cursor, so a failure cannot alter the live cursor.
   */
  virtual void load_state(std::string_view state) = 0;

  /**
   * @brief Unblocks pending work and permanently closes this cursor.
   *
   * This method may run concurrently with `next()` or `save_state()`.
   * Implementations must make it idempotent, noexcept, and safe for those
   * concurrency boundaries.
   */
  virtual void close() noexcept = 0;
};

/**
 * @brief Immutable factory for independent iterable epoch cursors.
 *
 * Map-style random access is deliberately absent. `open_epoch()` creates a
 * replacement cursor before the pipeline retires its live cursor, which gives
 * epoch changes and checkpoint restore a transactional preparation boundary.
 */
class IterableSource {
 public:
  /** @brief Enables destruction through the source interface. */
  virtual ~IterableSource() = default;

  /**
   * @brief Returns the immutable source capability descriptor.
   * @return A descriptor that remains unchanged for the source lifetime.
   */
  virtual IterableSourceDescriptor descriptor() const = 0;

  /**
   * @brief Opens an isolated cursor at the beginning of one logical epoch.
   * @param epoch Explicit non-negative logical epoch number.
   * @return A non-null, unconsumed cursor owned by the caller.
   * @throws Error If the epoch cannot be opened without mutating other cursors.
   */
  virtual std::unique_ptr<IterableSourceCursor> open_epoch(
      std::uint64_t epoch) const = 0;
};

/**
 * @brief Validates a source descriptor before any cursor is opened.
 * @param descriptor Immutable source contract to validate.
 * @throws InvalidArgument or ResourceExhausted for inconsistent declarations.
 */
void validate_iterable_source_descriptor(
    const IterableSourceDescriptor& descriptor);

}  // namespace jyppx::data

#endif  // JYPPX_DATA_ITERABLE_SOURCE_HPP
