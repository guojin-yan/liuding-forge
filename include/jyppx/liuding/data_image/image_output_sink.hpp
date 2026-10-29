// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_IMAGE_OUTPUT_SINK_HPP
#define JYPPX_LIUDING_DATA_IMAGE_IMAGE_OUTPUT_SINK_HPP

/** @file image_output_sink.hpp
 * @brief Bounded asynchronous publication of encoded PNG/JPEG images.
 */

#include <jyppx/liuding/data_image/image_encoder.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Action when the sink's pending item or byte capacity is full. */
enum class ImageSinkOverflowPolicy : std::uint8_t {
  block,  ///< Wait for local queue capacity, with cooperative cancellation.
  drop_newest,  ///< Drop the new selected submission when local capacity is
                ///< full.
};

/** @brief Result of the admission decision, before the background write. */
enum class ImageSinkDisposition : std::uint8_t {
  queued,       ///< Accepted for asynchronous publication.
  sampled_out,  ///< Stable-key sampling excluded the image.
  dropped,      ///< Local capacity rejected the image under drop_newest policy.
};

/** @brief File publication and bounded queue policy. */
struct ImageOutputSinkOptions {
  std::size_t max_pending_items = 32;  ///< Positive unsettled-submission cap.
  std::size_t max_pending_bytes = 512U
                                  << 20U;  ///< Positive admission-byte cap.
  std::size_t max_item_bytes = 512U
                               << 20U;  ///< Positive single-image byte cap.
  std::size_t max_reported_failures =
      64;  ///< Retained diagnostics; zero omits.
  std::uint32_t sample_every =
      1;  ///< Positive stable-key hash modulus; 1 keeps all.
  ImageSinkOverflowPolicy overflow = ImageSinkOverflowPolicy::block;
  ///< Local queue overflow policy; aggregate budget failures never wait here.
  bool continue_after_error = false;  ///< Continue later tasks after a failure.
  bool replace_existing =
      false;  ///< Permit staged publication over an old file.
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Optional aggregate encoded-queue and temporary-file accounting.
};

/** @brief Successful publication identity for artifact/event registration. */
struct ImageSinkWriteResult {
  std::filesystem::path path;  ///< Destination used for successful publication.
  std::size_t bytes = 0;       ///< Exact published encoded byte count.
  std::string
      sha256;  ///< Lowercase SHA-256 of the successfully published bytes.
};

/** @brief Admission result; only accepted work has a file-completion future. */
struct ImageSinkSubmission {
  ImageSinkDisposition disposition = ImageSinkDisposition::dropped;
  ///< Admission decision, independent of the eventual write outcome.
  std::future<ImageSinkWriteResult> completion;
  ///< Valid only when queued; returns publication identity or rethrows failure.
  std::future<void> evidence_completion;
  ///< Valid only for queued evidence work; reports callback or write failure.
};

/** @brief Outstanding accepted work and cumulative admission/task counters. */
struct ImageOutputSinkStats {
  std::uint64_t submitted =
      0;  ///< Valid requests observed while the sink is open.
  std::uint64_t queued = 0;  ///< Cumulative accepted submissions.
  std::uint64_t completed =
      0;  ///< Settled successful writes and evidence work.
  std::uint64_t sampled_out = 0;  ///< Requests excluded by stable-key sampling.
  std::uint64_t dropped = 0;      ///< Requests dropped for local capacity.
  std::uint64_t failed = 0;  ///< Settled failed writes or evidence callbacks.
  std::size_t pending_items = 0;  ///< Accepted work not yet fully settled.
  std::size_t pending_bytes =
      0;  ///< Admission bytes held through evidence work.
};

/** @brief One failed asynchronous publication or callback diagnostic.
 */
struct ImageSinkFailure {
  std::filesystem::path path;  ///< Destination associated with the failure.
  std::string message;  ///< Bounded-count diagnostic copied from the exception.
};

/**
 * @brief Owns one writer thread and a bounded queue of encoded images.
 *
 * The caller encodes before admission; no Tensor or CUDA transfer is hidden in
 * the sink. Sampling hashes a stable key and is independent of submission
 * order. Encoded-byte leases are released after publication settles; item and
 * byte admission remain held until the optional callback finishes. Queue and
 * temporary-file bytes use separate aggregate budget categories.
 * A successful ticket identifies the published bytes, while flush/close also
 * surface any asynchronous failure. An optional per-submission evidence
 * callback runs on the same writer thread after publication. The destructor
 * drains without throwing;
 * callers requiring evidence must inspect tickets or call flush/close.
 * Public operations are synchronized across producer threads; destruction
 * requires callers to have finished accessing the sink. Callbacks must not
 * reenter it, including close/flush, because they execute on its writer thread.
 */
class ImageOutputSink final {
 public:
  /** @brief Validates queue policy and starts one background writer.
   * @param options Positive queue, per-image and sampling limits.
   * @throws InvalidArgument Unsupported overflow or zero required limits.
   */
  explicit ImageOutputSink(ImageOutputSinkOptions options = {});
  /** @brief Rejects new work and joins the writer, suppressing task errors. */
  ~ImageOutputSink();

  ImageOutputSink(const ImageOutputSink&) = delete;
  ImageOutputSink& operator=(const ImageOutputSink&) = delete;
  ImageOutputSink(ImageOutputSink&&) = delete;
  ImageOutputSink& operator=(ImageOutputSink&&) = delete;

  /**
   * @brief Admits an encoded image, potentially waiting for queue capacity.
   * @param path File path with an extension matching the encoded format.
   * @param image Owning PNG/JPEG bytes, consumed even when skipped or dropped.
   * @param sample_key Stable nonempty key when sample_every is greater than
   * one.
   * @param stop Cancels a blocked admission without publishing the image.
   * @return Admission decision and, when queued, a file-completion future.
   * @throws InvalidArgument Invalid policy, path, key, or encoded image.
   * @throws ResourceExhausted One image exceeds a configured hard limit.
   * @throws Cancelled Admission was cancelled while waiting.
   * @note A closed sink is invalid. With continue_after_error disabled, the
   * first asynchronous error is rethrown by later submissions. Accepted write
   * failures are returned through completion and flush/close.
   */
  ImageSinkSubmission submit(std::filesystem::path path, EncodedImage image,
                             std::string_view sample_key = {},
                             std::stop_token stop = {});

  /**
   * @brief Admits an image and binds evidence work to its published bytes.
   *
   * The callback runs only after successful publication, on the writer thread,
   * without holding the sink mutex. It must finish promptly and must not call
   * back into this sink. Captured objects must outlive callback completion.
   * File completion remains successful if the callback
   * fails; evidence_completion and flush/close report that failure. The caller
   * can retry evidence registration using the file completion's path/digest.
   * This callback does not make file publication and evidence atomic.
   * Sampled or dropped submissions invoke no callback and have no futures.
   * @param path Destination with an extension matching the encoded format.
   * @param image Encoded image ownership consumed even when skipped or dropped.
   * @param record_evidence Nonempty callback receiving successful file
   * identity.
   * @param sample_key Stable nonempty key when sampling is configured.
   * @param stop Cooperative cancellation of blocked admission.
   * @return Admission decision and both futures when accepted.
   * @throws InvalidArgument The callback is empty or submit input is invalid.
   * @throws ResourceExhausted Per-image, local byte or aggregate cap exceeded.
   * @throws Cancelled Admission cancelled before acceptance.
   */
  ImageSinkSubmission submit_with_evidence(
      std::filesystem::path path, EncodedImage image,
      std::function<void(const ImageSinkWriteResult&)> record_evidence,
      std::string_view sample_key = {}, std::stop_token stop = {});

  /** @brief Waits for all work accepted before this call and rethrows errors.
   * @param stop Cooperative cancellation of the wait; accepted work continues.
   * @throws Cancelled The wait is cancelled before completion.
   * @note Rethrows the first publication/callback failure in the waited range.
   */
  void flush(std::stop_token stop = {});

  /** @brief Rejects new work, joins the writer and rethrows the first task
   * error.
   * @note Repeated calls are safe; a retained asynchronous error is rethrown.
   */
  void close();

  /** @brief Reads cumulative counters and outstanding queue usage.
   * @return Atomic detached snapshot, including any running callback.
   */
  [[nodiscard]] ImageOutputSinkStats stats() const;

  /** @brief Reads at most max_reported_failures write/callback diagnostics.
   * @return Detached diagnostics retained in settlement order.
   */
  [[nodiscard]] std::vector<ImageSinkFailure> failures() const;

 private:
  /** @brief Owned queue, writer, synchronization and publication state. */
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Sole owner of the writer implementation.
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_OUTPUT_SINK_HPP
