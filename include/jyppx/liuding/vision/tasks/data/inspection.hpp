// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_TASKS_DATA_INSPECTION_HPP
#define JYPPX_LIUDING_VISION_TASKS_DATA_INSPECTION_HPP
/** @file inspection.hpp @brief Data statistics, bounded decode cache and rect
 * ordering. */
#include <jyppx/liuding/data/samplers/sampler.hpp>
#include <jyppx/liuding/vision/tasks/data/dataset.hpp>
namespace jyppx::liuding::vision::tasks {
struct DecodeCacheOptions {
  std::uint64_t ram_bytes = 0;
  std::uint64_t disk_bytes = 0;
  std::filesystem::path directory;
  std::string decoder_revision = "stb-rgb-u8-chw/encoded-orientation-v1";
  bool reject_corrupt = false;
};
struct DecodeCacheStats {
  std::uint64_t ram_hits = 0, disk_hits = 0, misses = 0, invalid = 0,
                ram_bytes = 0, disk_bytes = 0;
};
/** @brief Thread-safe cache of deterministic CPU RGB uint8 decoded pixels only.
 * Keys cover complete source content and decoding semantics; annotations attach
 * fresh after read. Disk entries use a bounded native format with SHA-256 pixel
 * integrity, atomic publication and explicit corruption behavior. Logical cache
 * limits exclude samples retained by consumers and OS page/allocator caches.
 */
class ImageDecodeCache final {
 public:
  explicit ImageDecodeCache(DecodeCacheOptions options);
  ~ImageDecodeCache();
  data::image::DecodedImage read(const std::filesystem::path& path,
                                 const data::image::ImageDecodeOptions& options,
                                 std::stop_token stop = {});
  DecodeCacheStats stats() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
/** @brief Metadata/data audit JSON with per-class/size/negative/missing fields.
 * Optional image reads validate decoder dimensions and record complete content
 * hashes, with cooperative cancellation. Does not rewrite or repair source
 * labels.
 */
std::string inspect_dataset(const DetectionDatasetImport& source,
                            bool decode_images = false,
                            std::stop_token stop = {});
/** @brief Deterministic aspect-sorted batches; each source appears exactly
 * once. Batch blocks shuffle by logical seed/epoch. Identity binds all source
 * sizes and batch size; changing them invalidates exact resume. No
 * worker/global RNG state.
 */
std::shared_ptr<data::samplers::Sampler> make_rect_sampler(
    std::vector<data::image::ImageExtent> extents, std::size_t batch_size,
    std::uint64_t seed = 0);
}  // namespace jyppx::liuding::vision::tasks
#endif  // JYPPX_LIUDING_VISION_TASKS_DATA_INSPECTION_HPP
