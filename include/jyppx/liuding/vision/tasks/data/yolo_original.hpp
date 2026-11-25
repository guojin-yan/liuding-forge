// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_YOLO_ORIGINAL_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_YOLO_ORIGINAL_HPP
/** @file yolo_original.hpp @brief Explicit original YOLO box augmentation
 * policy. */

#include <cstddef>
#include <cstdint>
#include <string>

namespace jyppx::liuding::vision::tasks {
/** @brief OpenCV box-only profile with persistent logical-worker partner
 * history. Pixel processing is parallel; immutable epoch plans select Mosaic
 * partners from each logical worker's FIFO image buffer and independently apply
 * a complete pre-transform to a uniformly selected MixUp partner. Mixing uses
 * Beta(32,32), closes with Mosaic in the final configured epochs, then HSV,
 * flips and optional BGR reversal follow. Box-only CopyPaste is a no-op.
 * No global Python, NumPy or Torch random state participates.
 */
struct YoloOriginalDataOptions {
  bool enabled = false;
  bool training =
      true;  ///< False selects deterministic longest-side/LetterBox.
  std::uint64_t max_plan_bytes = 256ULL << 20;
  ///< Per-epoch immutable plan payload; excludes a prior in-flight epoch.
  std::uint64_t decoded_cache_bytes = 256ULL << 20;
  ///< Aggregate raw RGB memoization cap, divided among logical workers.
  ///< Memoization affects decode work only, not the logical partner buffer.
};
/** @brief Validates and fingerprints this explicit sampling/resource policy.
 * Full augmentation values and lineage are also bound by yolo_data_identity.
 * @throws InvalidArgument Invalid limits. */
std::string yolo_original_data_identity(const YoloOriginalDataOptions& options);
}  // namespace jyppx::liuding::vision::tasks
#endif
