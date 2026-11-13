// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_ORIENTATION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_ORIENTATION_HPP
/** @file orientation.hpp @brief Explicit JPEG EXIF orientation and label
 * coordinates. */
#include <jyppx/liuding/data_image/image_dataset.hpp>
namespace jyppx::liuding::vision::tasks {
enum class ExifPolicy {
  reject_non_identity,
  encoded_coordinates,
  already_oriented_coordinates,
  ignore
};
/** @brief Reads bounded JPEG APP1/TIFF orientation metadata; other formats
 * return 1. */
std::uint8_t image_exif_orientation(const std::filesystem::path& path);
/** @brief Applies exact integer transpose/flips to pixels and declared targets.
 * encoded_coordinates transforms all object geometry/RLE/keypoints together;
 * already_oriented_coordinates preserves target geometry and checks its extent.
 * Output source coordinates are the oriented display canvas with a new identity
 * trace. No interpolation, semantic keypoint permutation or global RNG occurs.
 */
data::image::VisionSample orient_vision_sample(
    data::image::VisionSample sample, const std::filesystem::path& source,
    ExifPolicy policy = ExifPolicy::reject_non_identity,
    std::stop_token stop = {});
}  // namespace jyppx::liuding::vision::tasks
#endif
