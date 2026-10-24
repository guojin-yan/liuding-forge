// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_PANOPTIC_HPP
#define JYPPX_LIUDING_DATA_IMAGE_PANOPTIC_HPP
/** @file panoptic.hpp
 * @brief Validates an explicit panoptic ID tensor against its segment table.
 */

#include <jyppx/liuding/data_image/annotations.hpp>
#include <jyppx/liuding/data_image/mask_decoder.hpp>

#include <cstddef>
#include <stop_token>

namespace jyppx::liuding::data::image {

/** @brief Explicit source-table verification and bounded lookup scratch. */
struct PanopticBindOptions {
  AnnotationLimits limits;  ///< Canvas, object, category and string limits.
  std::size_t max_working_bytes = 256U << 20U;
  ///< Conservative lookup/statistics payload; excludes input and returned
  ///< table.
  bool verify_declared_area =
      true;  ///< Check source area against exact pixels.
  bool verify_declared_box =
      true;  ///< Check declared box against tight bounds.
};

/**
 * @brief Binds all current segment IDs to exact pixel counts and tight boxes.
 * @param record Segment metadata with a source panoptic_mask_path reference.
 * @param mask Aligned CPU int64 panoptic IDs, with an explicit void_id.
 * @param category_count Size of the frozen contiguous category map.
 * @param options Source verification and independent resource limits.
 * @param stop Cancellation checked between rows and table entries.
 * @return Copied record with current boxes, mask_pixel_area and materialized
 * flag.
 * @throws InvalidArgument Invalid options, mask type or missing void semantics.
 * @throws SerializationError Unknown/duplicate/missing IDs or inconsistent
 * table.
 * @throws ResourceExhausted Canvas, table, lookup budget or allocation
 * exceeded.
 * @throws Cancelled Cancellation was requested before publication.
 *
 * Source segment IDs are unique positive 24-bit COCO IDs. Void/ignore IDs are
 * explicit and cannot collide with segments. Every table entry must occur in
 * the mask; every nonreserved mask ID must occur in the table. Source area,
 * category IDs and flags remain unchanged. Rebinding a materialized record
 * always checks its current mask_pixel_area, independently of the source-area
 * policy. The path remains provenance for the original file, not current
 * pixels.
 */
VisionAnnotationRecord bind_panoptic_mask(
    const VisionAnnotationRecord& record, const DecodedMask& mask,
    std::size_t category_count, const PanopticBindOptions& options = {},
    std::stop_token stop = {});

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_PANOPTIC_HPP
