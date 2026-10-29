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
#include <span>
#include <stop_token>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Pipeline resources tracked by the aggregate image budget. */
enum class ImageResourceKind : std::uint8_t {
  encoded_bytes,  ///< Retained encoded image payload bytes.
  decoded_bytes,  ///< Retained host tensor payload bytes.
  pixels,  ///< Retained image-canvas pixels, independent of channel count.
  codec_scratch_bytes,  ///< Temporary codec workspace bytes.
  annotation_bytes,     ///< Retained host annotation metadata bytes.
  index_bytes,          ///< Retained source/index metadata bytes.
  pinned_bytes,         ///< Retained pinned-host tensor bytes.
  gpu_bytes,            ///< Retained or temporary device tensor bytes.
  staged_batches,       ///< Live staged-batch slots.
  render_queue,  ///< Render-stage queue slots reserved by cooperating callers.
  encoded_queue_bytes,        ///< Encoded bytes admitted to an output queue.
  disk_spool_bytes,           ///< Temporary file-publication bytes.
  pipeline_inflight_batches,  ///< Pipeline admission slots held until release.
  pipeline_inflight_bytes,    ///< Pipeline admission's reserved byte capacity.
};

/** @brief Aggregate limits shared by cooperating image stages. */
struct ImageResourceBudgetLimits {
  std::uint64_t max_encoded_bytes = 512ULL << 20U;  ///< Encoded-payload cap.
  std::uint64_t max_decoded_bytes = 1ULL << 30U;  ///< Host tensor-payload cap.
  std::uint64_t max_pixels = 268'435'456ULL;      ///< Image-canvas pixel cap.
  std::uint64_t max_codec_scratch_bytes = 1ULL
                                          << 30U;      ///< Codec workspace cap.
  std::uint64_t max_annotation_bytes = 256ULL << 20U;  ///< Annotation-byte cap.
  std::uint64_t max_index_bytes = 256ULL << 20U;  ///< Source/index-byte cap.
  std::uint64_t max_pinned_bytes = 1ULL << 30U;   ///< Pinned-host-byte cap.
  std::uint64_t max_gpu_bytes = 2ULL << 30U;      ///< Device tensor-byte cap.
  std::uint64_t max_staged_batches = 64;  ///< Simultaneous staged-batch slots.
  std::uint64_t max_render_queue = 32;    ///< Cooperating render-queue slots.
  std::uint64_t max_encoded_queue_bytes = 512ULL << 20U;  ///< Output-queue cap.
  std::uint64_t max_disk_spool_bytes = 4ULL
                                       << 30U;  ///< Temporary file-byte cap.
  std::uint64_t max_pipeline_inflight_batches = 4;  ///< Live admission slots.
  std::uint64_t max_pipeline_inflight_bytes = 4ULL
                                              << 30U;  ///< Admission bytes.
};

/** @brief Current and high-water usage for one resource kind. */
struct ImageResourceUsage {
  std::uint64_t current = 0;  ///< Currently reserved units.
  std::uint64_t peak = 0;  ///< Highest simultaneous reservation since creation.
  std::uint64_t limit = 0;  ///< Configured cap in the category's units.
};

namespace resource_detail {
struct ImageResourceBudgetState;
}  // namespace resource_detail

/**
 * @brief Move-only RAII lease returned by an aggregate image budget.
 *
 * A lease keeps its budget state alive until release, so a result object may
 * safely retain a reservation even when the budget owner goes out of scope.
 * Different leases may release or reserve against the same state concurrently.
 * Moving or releasing the same lease object requires caller synchronization.
 */
class ImageResourceLease {
 public:
  /** @brief Creates an inactive lease with no reservation. */
  ImageResourceLease() = default;
  /** @brief Releases any reservation without throwing. */
  ~ImageResourceLease();

  ImageResourceLease(const ImageResourceLease&) = delete;
  ImageResourceLease& operator=(const ImageResourceLease&) = delete;
  /** @brief Transfers ownership without changing aggregate usage.
   * @param other Lease left inactive after the move.
   */
  ImageResourceLease(ImageResourceLease&& other) noexcept;
  /** @brief Releases this reservation, then transfers the other lease.
   * @param other Lease left inactive, unless this is self-assignment.
   * @return This lease after ownership transfer.
   */
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

  /** @brief Reserves another resource against this lease's budget state.
   * @param kind Resource category to charge.
   * @param amount Bytes, pixels or slots; zero returns an inactive lease.
   * @return Independent lease sharing this lease's budget state.
   * @throws InvalidArgument This lease is inactive or the category is unknown.
   * @throws ResourceExhausted The new reservation exceeds remaining capacity.
   */
  [[nodiscard]] ImageResourceLease reserve_related(ImageResourceKind kind,
                                                   std::uint64_t amount) const;

 private:
  friend class ImageResourceBudget;
  /** @brief Acquires a nonblocking reservation on an existing state.
   * @param state Shared live budget state.
   * @param kind Resource category to charge.
   * @param amount Bytes, pixels or slots; zero returns an inactive lease.
   * @return Owning reservation with the state retained.
   * @throws InvalidArgument The nonzero reservation's category is unknown.
   * @throws ResourceExhausted Insufficient remaining capacity.
   */
  static ImageResourceLease Reserve(
      const std::shared_ptr<resource_detail::ImageResourceBudgetState>& state,
      ImageResourceKind kind, std::uint64_t amount);
  /** @brief Adopts an already-accounted reservation without changing usage.
   * @param state Shared live budget state.
   * @param kind Category of the acquired reservation.
   * @param amount Units already charged to the state.
   */
  ImageResourceLease(
      std::shared_ptr<resource_detail::ImageResourceBudgetState> state,
      ImageResourceKind kind, std::uint64_t amount) noexcept;

  std::shared_ptr<resource_detail::ImageResourceBudgetState> state_;
  ///< Shared accounting state, retained independently of the public budget.
  ImageResourceKind kind_ = ImageResourceKind::encoded_bytes;  ///< Lease units.
  std::uint64_t amount_ = 0;  ///< Charged amount; zero means inactive.
};

/** @brief One amount in an atomic multi-resource admission request. */
struct ImageResourceRequest {
  ImageResourceKind kind;  ///< Category participating in the atomic request.
  std::uint64_t
      amount;  ///< Bytes, pixels or slots to reserve in this category.
};

/** @brief Shared lease collection retained by an image batch result. */
using ImageResourceLeaseSet = std::vector<ImageResourceLease>;

/**
 * @brief Thread-safe aggregate budget for image pipeline stages.
 *
 * `reserve` is non-blocking: callers either receive a lease or a structured
 * `ResourceExhausted` error. This makes backpressure policy explicit at the
 * queue or pipeline boundary and prevents hidden waits inside codec calls.
 * Limits count explicit cooperating reservations, not allocator overhead or
 * process RSS. A zero limit disables nonzero reservations in that category.
 */
class ImageResourceBudget {
 public:
  /** @brief Creates shared accounting state with immutable category limits.
   * @param limits Caps in bytes, pixels or slots according to each category.
   */
  explicit ImageResourceBudget(ImageResourceBudgetLimits limits = {});
  /** @brief Drops this owner; outstanding leases retain the accounting state.
   */
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
   * @throws InvalidArgument The nonzero reservation's category is unknown.
   * @throws ResourceExhausted Insufficient remaining aggregate capacity.
   * @note Zero returns an inactive lease without validating the category.
   */
  [[nodiscard]] ImageResourceLease reserve(ImageResourceKind kind,
                                           std::uint64_t amount);

  /**
   * @brief Waits for pipeline admission capacity, or throws on cancellation.
   *
   * An amount larger than the configured limit fails immediately. Use this at
   * a source/queue boundary, never inside a codec or while holding a pipeline
   * runtime lock.
   * @param kind Resource category to charge.
   * @param amount Bytes, pixels or slots; zero returns without waiting.
   * @param stop Cooperative cancellation while waiting for capacity.
   * @return RAII reservation held until release or destruction.
   * @throws InvalidArgument The nonzero reservation's category is unknown.
   * @throws ResourceExhausted Amount exceeds the category's entire limit.
   * @throws Cancelled A nonzero admission is cancelled before completion.
   */
  [[nodiscard]] ImageResourceLease reserve_wait(ImageResourceKind kind,
                                                std::uint64_t amount,
                                                std::stop_token stop = {});

  /**
   * @brief Waits until all requested resources fit, then reserves them
   * together.
   * @note Duplicate kinds are summed; any amount above its limit fails before
   * waiting. No partial reservation survives cancellation or an exception.
   * @param requests Finite category/amount list; zero amounts add no lease.
   * @param stop Cooperative cancellation before or during admission.
   * @return One lease for each category with a nonzero summed amount.
   * @throws InvalidArgument A request category is unknown.
   * @throws ResourceExhausted Summed amount overflows or exceeds its limit.
   * @throws Cancelled Admission is cancelled before all resources are acquired.
   */
  [[nodiscard]] ImageResourceLeaseSet reserve_wait_all(
      std::span<const ImageResourceRequest> requests,
      std::stop_token stop = {});

  /** @param kind Resource category to inspect.
   * @return Configured cap, or zero for an unknown category.
   */
  [[nodiscard]] std::uint64_t limit(ImageResourceKind kind) const noexcept;

  /** @param kind Resource category to inspect.
   * @return Atomic category snapshot, or all zeros for an unknown category.
   */
  [[nodiscard]] ImageResourceUsage usage(ImageResourceKind kind) const noexcept;

 private:
  std::shared_ptr<resource_detail::ImageResourceBudgetState> state_;
  ///< Shared mutex, immutable limits, counters and admission notification.
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_RESOURCE_BUDGET_HPP
