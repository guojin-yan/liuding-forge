// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_DATA_IMAGE_GEOMETRY_TRACE_HPP
#define JYPPX_LIUDING_DATA_IMAGE_GEOMETRY_TRACE_HPP
/**
 * @file geometry_trace.hpp
 * @brief Loss-aware coordinate mapping for resize, crop, pad and axis flips.
 *
 * Coordinates are continuous pixel boundaries in [0,W] x [0,H], not pixel
 * centers. The trace records geometry only; it does not resample image pixels.
 * Removed source regions and synthetic padding never acquire an inverse.
 */

#include <jyppx/liuding/data_image/annotations.hpp>

#include <array>
#include <cstdint>
#include <optional>

namespace jyppx::liuding::data::image {

/** @brief A positive integer image extent. */
struct ImageExtent {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

/**
 * @brief Immutable, bounded geometry trace for one image and its annotations.
 *
 * Each operation returns a new value, preserving the input if validation fails.
 * At most 128 operations can be composed. Affine coefficients and surviving
 * source bounds are retained; no per-pixel storage or unbounded history is
 * used.
 */
class GeometryTrace {
 public:
  /** @brief Starts an identity trace in the decoder's published image space.
   * @param extent Positive source width and height.
   * @throws InvalidArgument A dimension is zero.
   */
  explicit GeometryTrace(ImageExtent extent);

  /** @brief Returns a trace after boundary-aligned resizing.
   * @param extent Positive output width and height.
   * @return New trace; interpolation is the caller's separate pixel policy.
   * @throws InvalidArgument A dimension is zero or coefficients overflow.
   * @throws ResourceExhausted More than 128 operations are requested.
   */
  GeometryTrace resized(ImageExtent extent) const;

  /** @brief Returns a trace for a crop fully inside the current canvas.
   * @param left Horizontal crop offset in pixels.
   * @param top Vertical crop offset in pixels.
   * @param extent Positive crop width and height.
   * @return New trace with discarded source regions removed from valid bounds.
   * @throws InvalidArgument The crop is empty or outside the current canvas.
   * @throws ResourceExhausted More than 128 operations are requested.
   */
  GeometryTrace cropped(std::uint32_t left, std::uint32_t top,
                        ImageExtent extent) const;

  /** @brief Returns a trace with synthetic padding on each canvas edge.
   * @param left,top,right,bottom Nonnegative padding in pixels.
   * @return New trace; padding is excluded from inverse mapping.
   * @throws InvalidArgument Output dimensions overflow uint32.
   * @throws ResourceExhausted More than 128 operations are requested.
   */
  GeometryTrace padded(std::uint32_t left, std::uint32_t top,
                       std::uint32_t right, std::uint32_t bottom) const;

  /** @brief Returns a trace flipped about the current vertical center line.
   * @return New trace, with coordinates x mapped to width-x.
   * @throws ResourceExhausted More than 128 operations are requested.
   */
  GeometryTrace flipped_horizontal() const;

  /** @brief Returns a trace flipped about the current horizontal center line.
   * @return New trace, with coordinates y mapped to height-y.
   * @throws ResourceExhausted More than 128 operations are requested.
   */
  GeometryTrace flipped_vertical() const;

  /** @brief Maps a source point only if its source content survives.
   * @param point Finite point in source boundary coordinates.
   * @return Current coordinate, or nullopt outside surviving source content.
   * @throws InvalidArgument A coordinate is not finite.
   */
  std::optional<ImagePoint> forward_point(ImagePoint point) const;

  /** @brief Maps a current point only if it represents real source content.
   * @param point Finite point in current boundary coordinates.
   * @return Source coordinate, or nullopt for padding/outside content.
   * @throws InvalidArgument A coordinate is not finite.
   */
  std::optional<ImagePoint> inverse_point(ImagePoint point) const;

  /** @brief Clips a positive source box to surviving content, then maps it.
   * @param box Finite positive xyxy box in source coordinates.
   * @return Mapped clipped box, or nullopt for no positive-area intersection.
   * @throws InvalidArgument The input is not finite or has nonpositive area.
   */
  std::optional<ImageBox> forward_box(ImageBox box) const;

  /** @brief Clips a current box to real content before mapping to source.
   * @param box Finite positive xyxy box in current coordinates.
   * @return Source box, or nullopt when only padding/outside content is
   * covered.
   * @throws InvalidArgument The input is not finite or has nonpositive area.
   */
  std::optional<ImageBox> inverse_box(ImageBox box) const;

  /** @brief Returns the original decoder-space extent. */
  ImageExtent source_extent() const noexcept;
  /** @brief Returns the current canvas extent, including padding. */
  ImageExtent output_extent() const noexcept;
  /** @brief Returns the surviving source rectangle, or nullopt if all was cut.
   */
  std::optional<ImageBox> source_region() const noexcept;
  /** @brief Returns current real-content bounds, excluding synthetic padding.
   */
  std::optional<ImageBox> output_region() const noexcept;
  /** @brief Reports whether any original region was discarded. */
  bool retains_full_source() const noexcept;
  /** @brief Returns the row-major 3x3 source-to-output boundary transform.
   * @return Mathematical mapping; use point/box APIs to enforce valid regions.
   */
  std::array<double, 9> matrix() const noexcept;

 private:
  GeometryTrace next() const;
  void validate_mapping() const;

  ImageExtent source_;
  ImageExtent output_;
  std::optional<ImageBox> source_region_;
  double scale_x_ = 1;
  double scale_y_ = 1;
  double offset_x_ = 0;
  double offset_y_ = 0;
  std::uint16_t operations_ = 0;
};

}  // namespace jyppx::liuding::data::image

#endif  // JYPPX_LIUDING_DATA_IMAGE_GEOMETRY_TRACE_HPP
