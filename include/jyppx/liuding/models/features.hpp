// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_FEATURES_HPP
#define JYPPX_LIUDING_MODELS_FEATURES_HPP
/** @file features.hpp
 * @brief Layout and spatial contracts for registered image features.
 */
#include <cstdint>
#include <string>

namespace jyppx::liuding::models {
/** @brief Axis meanings; token grids are not implicit feature pyramids. */
enum class FeatureLayout : std::uint8_t {
  nc = 1,    ///< Batch, channels: a global embedding.
  ntc = 2,   ///< Batch, tokens, channels: an ordered token sequence.
  nchw = 3,  ///< Batch, channels, height, width: a spatial feature map.
};

/** @brief One ordered output of the default architecture configuration. */
struct FeatureInfo {
  std::string name;  ///< Unique stable name in the extractor output tuple.
  FeatureLayout layout = FeatureLayout::nchw;  ///< Explicit tensor axis order.
  std::int64_t channels = 0;  ///< Positive channel or embedding dimension.
  std::int64_t reduction = 0;
  ///< Pixel stride of spatial/patch features; zero for a global embedding.
  std::int64_t prefix_tokens = 0;
  ///< Leading non-patch tokens in an ntc output; zero in other layouts.
};

/** @brief Spatial input rule, applied before computing registered features. */
struct FeatureInputSpec {
  std::int64_t spatial_multiple = 1;
  ///< Positive minimum and multiple for both input height and width.
  bool crop_to_multiple = false;
  ///< True crops bottom/right to the multiple; false requires exact multiples.
};
}  // namespace jyppx::liuding::models
#endif
