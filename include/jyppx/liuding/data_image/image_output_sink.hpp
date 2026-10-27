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
  block,
  drop_newest,
};

/** @brief Result of the admission decision, before the background write. */
enum class ImageSinkDisposition : std::uint8_t {
  queued,
  sampled_out,
  dropped,
};

/** @brief File publication and bounded queue policy. */
struct ImageOutputSinkOptions {
  std::size_t max_pending_items = 32;
  std::size_t max_pending_bytes = 512U << 20U;
  std::size_t max_item_bytes = 512U << 20U;
  std::size_t max_reported_failures = 64;
  std::uint32_t sample_every = 1;
  ImageSinkOverflowPolicy overflow = ImageSinkOverflowPolicy::block;
  bool continue_after_error = false;
  bool replace_existing = false;
  std::shared_ptr<ImageResourceBudget> resource_budget;
};

/** @brief Successful publication identity for artifact/event registration. */
struct ImageSinkWriteResult {
  std::filesystem::path path;
  std::size_t bytes = 0;
  std::string sha256;
};

/** @brief Admission result; futures are valid only when queued. */
struct ImageSinkSubmission {
  ImageSinkDisposition disposition = ImageSinkDisposition::dropped;
  std::future<ImageSinkWriteResult> completion;
  std::future<void> evidence_completion;
};

/** @brief Outstanding accepted work and cumulative admission/task counters. */
struct ImageOutputSinkStats {
  std::uint64_t submitted = 0;
  std::uint64_t queued = 0;
  std::uint64_t completed = 0;
  std::uint64_t sampled_out = 0;
  std::uint64_t dropped = 0;
  std::uint64_t failed = 0;
  std::size_t pending_items = 0;
  std::size_t pending_bytes = 0;
};

/** @brief One failed asynchronous publication or callback diagnostic.
 */
struct ImageSinkFailure {
  std::filesystem::path path;
  std::string message;
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
 */
class ImageOutputSink final {
 public:
  explicit ImageOutputSink(ImageOutputSinkOptions options = {});
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
   * @throws InvalidArgument Invalid policy, path, key, or encoded image.
   * @throws ResourceExhausted One image exceeds a configured hard limit.
   * @throws Cancelled Admission was cancelled while waiting.
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
   * @throws InvalidArgument The callback is empty or submit input is invalid.
   */
  ImageSinkSubmission submit_with_evidence(
      std::filesystem::path path, EncodedImage image,
      std::function<void(const ImageSinkWriteResult&)> record_evidence,
      std::string_view sample_key = {}, std::stop_token stop = {});

  /** @brief Waits for all work accepted before this call and rethrows errors.
   */
  void flush(std::stop_token stop = {});

  /** @brief Rejects new work, drains accepted work, and rethrows errors. */
  void close();

  /** @brief Reads cumulative counters and outstanding queue usage. */
  [[nodiscard]] ImageOutputSinkStats stats() const;

  /** @brief Reads at most max_reported_failures write diagnostics. */
  [[nodiscard]] std::vector<ImageSinkFailure> failures() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_IMAGE_OUTPUT_SINK_HPP
