// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_COPY_STREAM_HPP
#define JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_COPY_STREAM_HPP
/** @file image_batch_copy_stream.hpp
 * @brief Bounded pinned-host CUDA copies with explicit stream handoff.
 */

#include <jyppx/liuding/data_image/image_batch.hpp>

#include <memory>
#include <optional>

namespace jyppx::liuding::data::image {

/** @brief Observed dependency and optional CUDA event interval for one copy. */
struct ImageBatchCopyTimeline {
  Device device{torch::kCUDA, 0};  ///< Explicit indexed destination device.
  std::int64_t copy_stream_id =
      0;  ///< LibTorch stream ID, not a native handle.
  std::optional<std::int64_t> consumer_stream_id;  ///< Set by receive().
  bool copy_complete = false;  ///< Completion observed by query or wait.
  std::optional<double> copy_elapsed_ms;  ///< Event interval after completion.
};

namespace copy_stream_detail {
struct Pool;
struct Transfer;
}  // namespace copy_stream_detail

class ImageBatchCopyStream;

/**
 * @brief Move-only owner of a submitted copy and its source resource leases.
 *
 * receive() inserts a CUDA event wait on the consumer's current stream; it
 * does not wait on the CPU. Keep this ticket alive to overlap copy and compute.
 * wait(), destruction, replacement and stream.close() drain this copy before
 * releasing its source storage and in-flight slot. They may block the CPU.
 * Input aliases must not be mutated until completion. Tensor aliases retained
 * beyond a returned ImageBatch must retain its resource leases separately.
 * Operations on distinct tickets/stream objects are serialized internally;
 * moving or destroying an object requires exclusive caller access.
 */
class ImageBatchTransfer {
 public:
  /** @brief Drains the copy; terminates if backend cleanup cannot be confirmed.
   * @note Use wait() to observe backend errors before noexcept destruction.
   */
  ~ImageBatchTransfer();
  ImageBatchTransfer(const ImageBatchTransfer&) = delete;
  ImageBatchTransfer& operator=(const ImageBatchTransfer&) = delete;
  /** @brief Transfers the ticket, leaving other empty.
   * @param other Ticket whose copy ownership is transferred.
   */
  ImageBatchTransfer(ImageBatchTransfer&& other) noexcept;
  /** @brief Drains this copy, then transfers another ticket.
   * @param other Source ticket, left empty except on self-assignment.
   * @return This ticket after replacement.
   */
  ImageBatchTransfer& operator=(ImageBatchTransfer&& other) noexcept;

  /** @brief Returns the batch once, with an event dependency on current stream.
   * @return Batch retaining its destination storage and aggregate leases.
   * @throws InvalidArgument Empty or already received ticket.
   * @throws CompatibilityError Same stream or unsupported CUDA Graph capture.
   * @throws DeviceError Backend handoff or allocator stream registration fails.
   * @note Every destination storage is recorded on the consumer stream before
   * publication. No CPU synchronization or device-wide barrier is performed.
   */
  ImageBatch receive();
  /** @brief Queries this copy without releasing its slot or source leases.
   * @return Whether the copy completion event is ready.
   * @throws InvalidArgument Empty ticket.
   * @throws DeviceError Backend query fails.
   */
  [[nodiscard]] bool ready() const;
  /** @brief Waits for this copy and releases its source and in-flight slot.
   * @throws InvalidArgument Empty ticket.
   * @throws DeviceError Backend synchronization fails; owners remain retained.
   * @note Idempotent; an unread destination remains available to receive().
   */
  void wait();
  /** @brief Queries event completion and the observed stream dependency.
   * @return IDs and optional begin/end event interval, in milliseconds.
   * @throws InvalidArgument Empty ticket.
   * @throws DeviceError Backend query or timing fails.
   * @note The interval can include stream starvation, not just DMA activity;
   * it does not measure compute overlap or establish a speed improvement.
   */
  [[nodiscard]] ImageBatchCopyTimeline timeline() const;

 private:
  friend class ImageBatchCopyStream;
  /** @brief Adopts an already submitted transfer.
   * @param pool Shared stream and bounded submission state.
   * @param transfer Copy event and retained batch owners.
   */
  ImageBatchTransfer(std::shared_ptr<copy_stream_detail::Pool> pool,
                     std::shared_ptr<copy_stream_detail::Transfer> transfer);
  void DrainNoexcept() noexcept;  ///< Confirms completion before owner release.
  std::shared_ptr<copy_stream_detail::Pool> pool_;  ///< Shared stream state.
  std::shared_ptr<copy_stream_detail::Transfer> transfer_;  ///< Copy owners.
};

/**
 * @brief Explicit bounded CUDA copy stream for prepared ImageBatch values.
 *
 * CPU-only builds expose the same API and reject construction with DeviceError.
 * The selected LibTorch pool stream differs from the constructor's current
 * stream; it is not an exclusive native stream. No worker or background poller
 * is started. Completed copies retain their slot until ticket.wait(), ticket
 * destruction or close(); receive() alone preserves asynchronous operation.
 * Logical image budgets exclude allocator residency and private workspace.
 */
class ImageBatchCopyStream {
 public:
  /** @brief Selects a non-current CUDA pool stream and immutable copy policy.
   * @param memory Same policy used by prepare_image_batch(); requires pinned
   * CPU input, non_blocking=true and an explicitly indexed CUDA target.
   * @param max_in_flight Undrained tickets, in [1,4096]; full submit fails.
   * @param record_timing Whether to record timed CUDA begin/end events.
   * @param limits Per-batch structural and logical payload bounds.
   * @param resource_budget Optional destination accounting before allocation.
   * @throws InvalidArgument Invalid memory, capacity or image limits.
   * @throws DeviceError CUDA backend/device is unavailable.
   */
  explicit ImageBatchCopyStream(
      data::BatchMemoryOptions memory, std::size_t max_in_flight = 2,
      bool record_timing = false, ImageBatchLimits limits = {},
      std::shared_ptr<ImageResourceBudget> resource_budget = {});
  /** @brief Closes and drains; terminates on unconfirmed backend cleanup. */
  ~ImageBatchCopyStream();
  ImageBatchCopyStream(const ImageBatchCopyStream&) = delete;
  ImageBatchCopyStream& operator=(const ImageBatchCopyStream&) = delete;
  /** @brief Transfers this stream's state, leaving other empty.
   * @param other Stream owner to transfer.
   */
  ImageBatchCopyStream(ImageBatchCopyStream&& other) noexcept;
  /** @brief Closes this stream, then transfers another stream's state.
   * @param other Stream owner, left empty except on self-assignment.
   * @return This object after replacement.
   */
  ImageBatchCopyStream& operator=(ImageBatchCopyStream&& other) noexcept;

  /** @brief Submits a prepared pinned CPU batch on the explicit copy stream.
   * @param prepared Unmodified result of prepare_image_batch() with memory.
   * @param stop Cancellation checked before submission; queued DMA is drained.
   * @return Ticket retaining source and destination until copy completion.
   * @throws InvalidArgument Closed/empty stream or malformed batch.
   * @throws CompatibilityError Invalid prepared policy or CUDA Graph capture.
   * @throws ResourceExhausted Capacity or logical image budget is exhausted.
   * @throws Cancelled Cancellation was requested before submission.
   * @throws DeviceError Backend copy fails; partial work is drained first.
   * @note Caller device/current stream is restored on success and failure.
   */
  ImageBatchTransfer submit(ImageBatch prepared, std::stop_token stop = {});
  /** @brief Counts undrained tickets, including completed but unwaited copies.
   * @return Current slot usage; zero for a moved-from object.
   */
  [[nodiscard]] std::size_t pending() const;
  /** @brief Rejects future submissions and drains all outstanding copies.
   * @note Idempotent; existing tickets can still receive their batch.
   * @throws DeviceError Drain fails; pending owners remain retained for retry.
   */
  void close();

 private:
  void CloseNoexcept() noexcept;  ///< Drains before replacing the stream owner.
  std::shared_ptr<copy_stream_detail::Pool> pool_;  ///< Shared bounded state.
};

}  // namespace jyppx::liuding::data::image
#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_BATCH_COPY_STREAM_HPP
