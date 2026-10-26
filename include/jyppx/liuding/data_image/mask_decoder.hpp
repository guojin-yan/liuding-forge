// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_MASK_DECODER_HPP
#define JYPPX_LIUDING_DATA_IMAGE_MASK_DECODER_HPP
/**
 * @file mask_decoder.hpp
 * @brief Lossless, bounded PNG label decoding independent of photo policies.
 */

#include <jyppx/liuding/data_image/image_decoder.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stop_token>
#include <vector>

namespace jyppx::liuding::data::image {

/** @brief Explicit interpretation of encoded PNG label samples. */
enum class MaskEncoding : std::uint8_t {
  class_index,    ///< Grayscale 1/2/4/8/16-bit samples, without intensity
                  ///< scaling.
  palette_index,  ///< Original 1/2/4/8-bit palette indices, including
                  ///< transparent entries.
  rgb_mapping,    ///< 8-bit RGB colors mapped through an explicit class table.
  panoptic_rgb,  ///< 8-bit RGB packed as R + 256*G + 65536*B (COCO convention).
};

/** @brief One exact RGB-to-label association, without color conversion. */
struct MaskColorMapping {
  std::array<std::uint8_t, 3> rgb = {};  ///< Encoded RGB bytes.
  std::int64_t id = 0;  ///< Nonnegative label, or explicit ignore/void ID.
};

/** @brief One palette entry; alpha is metadata and does not replace its index.
 */
struct MaskPaletteEntry {
  std::array<std::uint8_t, 3> rgb = {};  ///< Encoded display color.
  std::uint8_t alpha = 255;  ///< Encoded tRNS alpha, or opaque by default.
};

/** @brief Explicit label semantics and independent resource limits. */
struct MaskDecodeOptions {
  MaskEncoding encoding =
      MaskEncoding::class_index;  ///< Required source convention.
  ImageDecodeLimits limits =
      {};  ///< Encoded, dimension, pixel and int64 output bounds.
  std::size_t max_working_bytes = 512U << 20U;
  ///< Bound on compressed copy plus filtered raster; excludes input and output
  ///< storage.
  std::size_t max_chunks =
      65'536;  ///< Bound on all PNG chunks, including empty ones.
  std::size_t max_color_entries = 65'536;  ///< Bound on the explicit RGB table.
  std::vector<MaskColorMapping> colors;    ///< Used only by rgb_mapping.
  std::optional<std::int64_t>
      ignore_id;  ///< Preserved metadata; no implicit remapping.
  std::optional<std::int64_t>
      void_id;  ///< Preserved metadata; no implicit remapping.
  bool unknown_color_is_ignore = false;
  ///< Opt-in unknown RGB fallback; requires rgb_mapping and an explicit
  ///< ignore_id.
  std::shared_ptr<ImageResourceBudget> resource_budget;
  ///< Optional aggregate budget for known allocations, not process RSS or
  ///< private codec/backend buffers.
};

/** @brief Owning discrete CPU labels with their source interpretation. */
struct DecodedMask {
  std::shared_ptr<ImageResourceLeaseSet> resource_leases;
  ///< Holds decoded storage and pixel reservations while labels are retained.
  Tensor ids;  ///< Contiguous, owning int64 [H,W]; never normalized or
               ///< color-scaled.
  MaskEncoding encoding = MaskEncoding::class_index;  ///< Interpretation used.
  std::uint32_t width = 0;  ///< Current canvas width; PNG width when decoded.
  std::uint32_t height =
      0;  ///< Current canvas height; PNG height when decoded.
  std::uint8_t source_bit_depth = 0;  ///< Encoded sample precision.
  bool source_interlaced = false;     ///< True when Adam7 was decoded.
  std::vector<MaskPaletteEntry>
      palette;  ///< Original PLTE/tRNS for indexed labels.
  std::optional<std::int64_t> ignore_id;  ///< Caller-declared ignore marker.
  std::optional<std::int64_t> void_id;    ///< Caller-declared void marker.
};

/**
 * @brief Decodes PNG labels with exact ID preservation and CRC/Adler checks.
 * @param bytes Encoded PNG; caller retains ownership for this call.
 * @param options Explicit convention, mapping and resource budgets.
 * @param stop Cooperative cancellation checked while parsing and publishing
 * rows.
 * @return Owning CPU int64 labels; alpha never silently becomes background.
 * @note Caller-owned encoded bytes are not charged. Keep the returned object
 * or its resource_leases alive while retaining its tensor separately.
 * @throws InvalidArgument Invalid options or an ambiguous/duplicate color
 * mapping.
 * @throws SerializationError Corrupt PNG or unsupported source
 * subtype/convention.
 * @throws ResourceExhausted Encoded, raster, output, chunk or mapping limit
 * exceeded.
 * @throws DeviceError LibTorch allocation failed for a non-resource backend
 * reason.
 * @throws Cancelled Cancellation was requested before publication.
 *
 * Both noninterlaced and Adam7 PNG are accepted. Alpha-channel PNG, animated
 * PNG, CgBI, and grayscale/RGB tRNS are rejected. Palette tRNS is retained as
 * metadata. EXIF, ICC, gamma and display palettes never transform IDs or RGB
 * mapping keys.
 */
DecodedMask decode_mask(std::span<const std::byte> bytes,
                        const MaskDecodeOptions& options = {},
                        std::stop_token stop = {});

/**
 * @brief Reads a local PNG through the same bounded mask decoder.
 * @param path Local file; dataset-relative path validation belongs to
 * ImageDataset.
 * @param options Label interpretation and limits.
 * @param stop Cancellation checked while reading and decoding.
 * @return Decoded CPU int64 mask.
 * @note With a resource budget, file bytes are charged before allocation and
 * released after decode; the result retains decoded and pixel leases.
 * @throws IoError File metadata, open or read failed.
 * @throws InvalidArgument, SerializationError, ResourceExhausted, DeviceError,
 *         or Cancelled As documented by decode_mask().
 */
DecodedMask read_mask_file(const std::filesystem::path& path,
                           const MaskDecodeOptions& options = {},
                           std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_MASK_DECODER_HPP
