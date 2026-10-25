// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_RESOURCE_BUDGET_HPP
#define JYPPX_LIUDING_DATA_IMAGE_RESOURCE_BUDGET_HPP
/**
 * @file resource_budget.hpp
 * @brief Thread-safe aggregate resource leases for image pipelines.
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Pipeline resources tracked by the aggregate image budget. */
enum class ImageResourceKind : std::uint8_t {
  encoded_bytes,
  decoded_bytes,
  pixels,
  codec_scratch_bytes,
  annotation_bytes,
  index_bytes,
  pinned_bytes,
  gpu_bytes,
  staged_batches,
  render_queue,
  encoded_queue_bytes,
  disk_spool_bytes,
};

/** @brief Aggregate limits shared by cooperating image stages. */
struct ImageResourceBudgetLimits {
  std::uint64_t max_encoded_bytes = 512ULL << 20U;
  std::uint64_t max_decoded_bytes = 1ULL << 30U;
  std::uint64_t max_pixels = 268'435'456ULL;
  std::uint64_t max_codec_scratch_bytes = 1ULL << 30U;
  std::uint64_t max_annotation_bytes = 256ULL << 20U;
  std::uint64_t max_index_bytes = 256ULL << 20U;
  std::uint64_t max_pinned_bytes = 1ULL << 30U;
  std::uint64_t max_gpu_bytes = 2ULL << 30U;
  std::uint64_t max_staged_batches = 64;
  std::uint64_t max_render_queue = 32;
  std::uint64_t max_encoded_queue_bytes = 512ULL << 20U;
  std::uint64_t max_disk_spool_bytes = 4ULL << 30U;
};

/** @brief Current and high-water usage for one resource kind. */
struct ImageResourceUsage {
  std::uint64_t current = 0;
  std::uint64_t peak = 0;
  std::uint64_t limit = 0;
};

namespace resource_detail {
struct ImageResourceBudgetState;
}  // namespace resource_detail

/**
 * @brief Move-only RAII lease returned by an aggregate image budget.
 *
 * A lease keeps its budget state alive until release, so a result object may
 * safely retain a reservation even when the budget owner goes out of scope.
 */
class ImageResourceLease {
 public:
  ImageResourceLease() = default;
  ~ImageResourceLease();

  ImageResourceLease(const ImageResourceLease&) = delete;
  ImageResourceLease& operator=(const ImageResourceLease&) = delete;
  ImageResourceLease(ImageResourceLease&& other) noexcept;
  ImageResourceLease& operator=(ImageResourceLease&& other) noexcept;

  /** @brief Releases the reservation immediately; safe to call repeatedly. */
  void release() noexcept;

  /** @return Whether this lease still owns a reservation. */
  [[nodiscard]] bool active() const noexcept {
    return state_ != nullptr && amount_ != 0;
  }

  /** @return Resource category reserved by this lease. */
  [[nodiscard]] ImageResourceKind kind() const noexcept { return kind_; }

  /** @return Amount reserved by this lease. */
  [[nodiscard]] std::uint64_t amount() const noexcept { return amount_; }

 private:
  friend class ImageResourceBudget;
  ImageResourceLease(
      std::shared_ptr<resource_detail::ImageResourceBudgetState> state,
      ImageResourceKind kind, std::uint64_t amount) noexcept;

  std::shared_ptr<resource_detail::ImageResourceBudgetState> state_;
  ImageResourceKind kind_ = ImageResourceKind::encoded_bytes;
  std::uint64_t amount_ = 0;
};

/**
 * @brief Thread-safe aggregate budget for image pipeline stages.
 *
 * `reserve` is non-blocking: callers either receive a lease or a structured
 * `ResourceExhausted` error. This makes backpressure policy explicit at the
 * queue or pipeline boundary and prevents hidden waits inside codec calls.
 */
class ImageResourceBudget {
 public:
  explicit ImageResourceBudget(ImageResourceBudgetLimits limits = {});
  ~ImageResourceBudget();

  ImageResourceBudget(const ImageResourceBudget&) = delete;
  ImageResourceBudget& operator=(const ImageResourceBudget&) = delete;
  ImageResourceBudget(ImageResourceBudget&&) = delete;
  ImageResourceBudget& operator=(ImageResourceBudget&&) = delete;

  /**
   * @brief Reserves an amount or throws `ResourceExhausted`.
   * @param kind Resource category to charge.
   * @param amount Number of bytes, pixels, or queue slots.
   * @return RAII lease released on destruction or `release()`.
   */
  [[nodiscard]] ImageResourceLease reserve(ImageResourceKind kind,
                                           std::uint64_t amount);

  /** @return Configured aggregate limit for one resource category. */
  [[nodiscard]] std::uint64_t limit(ImageResourceKind kind) const noexcept;

  /** @return Current and high-water usage for one resource category. */
  [[nodiscard]] ImageResourceUsage usage(ImageResourceKind kind) const noexcept;

 private:
  std::shared_ptr<resource_detail::ImageResourceBudgetState> state_;
};

/** @brief Shared lease collection retained by an image batch result. */
using ImageResourceLeaseSet = std::vector<ImageResourceLease>;

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_RESOURCE_BUDGET_HPP
