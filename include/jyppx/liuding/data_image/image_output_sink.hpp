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

/** @brief Admission result; completion is valid only when queued. */
struct ImageSinkSubmission {
  ImageSinkDisposition disposition = ImageSinkDisposition::dropped;
  std::future<ImageSinkWriteResult> completion;
};

/** @brief Outstanding accepted work and cumulative admission/write counters. */
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

/** @brief One failed asynchronous write retained in a bounded diagnostic list.
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
 * order. Accepted images retain their encoded-byte lease until completion.
 * Queue and temporary-file bytes use separate aggregate budget categories.
 * A successful ticket identifies the published bytes, while flush/close also
 * surface any asynchronous failure. The destructor drains without throwing;
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
