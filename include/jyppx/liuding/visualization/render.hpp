// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef JYPPX_LIUDING_VISUALIZATION_RENDER_HPP
#define JYPPX_LIUDING_VISUALIZATION_RENDER_HPP
/**
 * @file render.hpp
 * @brief Headless CPU rasterization for common image annotations.
 */

#include <jyppx/liuding/data_image/annotations.hpp>
#include <jyppx/liuding/data_image/pixel_spec.hpp>
#include <jyppx/liuding/data_image/resource_budget.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stop_token>

namespace jyppx::liuding::visualization {

/** @brief RGB color used for compositing and annotation outlines. */
struct RgbColor {
  std::uint8_t red = 255;  ///< Red intensity in [0, 255].
  std::uint8_t green = 0;  ///< Green intensity in [0, 255].
  std::uint8_t blue = 0;   ///< Blue intensity in [0, 255].
};

/** @brief Independent resource and drawing policy for one CPU render. */
struct RenderOptions {
  RgbColor background{0, 0, 0};    ///< Background for input alpha compositing.
  std::uint32_t line_width = 2;    ///< Outline width in pixels, in [1, 128].
  std::uint32_t point_radius = 3;  ///< Keypoint radius in pixels, in [0, 512].
  std::uint8_t mask_alpha = 112;   ///< Mask opacity; 0 transparent, 255 opaque.
  std::uint8_t heatmap_alpha = 176;  ///< Maximum heatmap opacity in [0, 255].
  bool draw_boxes = true;            ///< Draw axis-aligned box outlines.
  bool draw_oriented_boxes = true;   ///< Draw oriented-box outlines.
  bool draw_polygons = true;         ///< Draw polygon outlines.
  bool draw_keypoints = true;  ///< Draw visible keypoints and skeleton links.
  bool draw_instance_rle = true;  ///< Composite instance RLE masks.
  std::optional<std::int64_t>
      mask_ignore_id;                     ///< Dense mask ID left unpainted.
  std::uint64_t max_pixels = 67'108'864;  ///< Positive output-canvas pixel cap.
  std::size_t max_output_bytes = 512U << 20U;  ///< Positive RGB8 payload cap.
  std::size_t max_objects = 100'000;  ///< Positive annotation-object count cap.
  std::size_t max_points =
      5'000'000;  ///< Vertices, keypoints, links and RLE runs.
  std::uint64_t max_raster_ops = 250'000'000;  ///< Positive pixel-write budget.
  std::shared_ptr<data::image::ImageResourceBudget> resource_budget;
  ///< Optional shared accounting for pixels, temporary work and output bytes.
};

/** @brief Owning HWC RGB8 render result with explicit pixel semantics. */
struct RenderedImage {
  Tensor pixels;                ///< Owned contiguous CPU uint8 [H,W,3] output.
  data::image::PixelSpec spec;  ///< HWC RGB unsigned-full-scale semantics.
  /** @brief Holds the aggregate decoded-byte lease while pixels are retained.
   */
  data::image::ImageResourceLease output_lease;
};

/**
 * @brief Renders annotation overlays into a new HWC RGB8 CPU image.
 *
 * The source pixels and metadata are never modified. Input images must use
 * uint8 full-scale encoding. Masks are exact int64 [H,W] IDs; heatmaps are
 * finite float32/float64 [H,W] values in [0,1]. Geometry is clipped to the
 * output canvas. Text, image encoding, file output, CUDA rendering, model
 * postprocessing and metric computation are separate capabilities.
 *
 * @param pixels CPU CHW or HWC uint8 image.
 * @param pixel_spec Explicit input channel, alpha and layout semantics.
 * @param annotations Current-canvas annotations matching the image extent.
 * @param categories Optional class metadata used for keypoint skeleton edges.
 * @param mask Optional CPU int64 [H,W] segmentation/panoptic IDs.
 * @param heatmap Optional CPU float32/float64 [H,W] intensity map.
 * @param options Drawing and per-call resource limits.
 * @param stop Cooperative cancellation checked between rows and primitives.
 * @return A separate, contiguous HWC RGB8 CPU image.
 * @throws InvalidArgument Invalid tensor, geometry, options or schema.
 * @throws ResourceExhausted Pixel, output, object, point or raster budget
 *         exceeded.
 * @throws Cancelled Cancellation was requested before publication.
 */
RenderedImage render_annotations(
    const Tensor& pixels, const data::image::PixelSpec& pixel_spec,
    const data::image::VisionAnnotationRecord& annotations,
    std::span<const data::image::VisionCategory> categories = {},
    const Tensor& mask = {}, const Tensor& heatmap = {},
    const RenderOptions& options = {}, std::stop_token stop = {});

}  // namespace jyppx::liuding::visualization

#endif  // JYPPX_LIUDING_VISUALIZATION_RENDER_HPP
