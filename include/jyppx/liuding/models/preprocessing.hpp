// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_PREPROCESSING_HPP
#define JYPPX_LIUDING_MODELS_PREPROCESSING_HPP
/** @file preprocessing.hpp
 * @brief Data-only image input contracts associated with particular weights.
 */

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace jyppx::liuding::models {

/** @brief Deterministic evaluation geometry, without stochastic augmentation.
 */
enum class PreprocessResize : std::uint8_t {
  stretch = 1,  ///< Resize directly to input_size, without cropping.
  shortest_edge_center_crop = 2,  ///< Keep aspect, then crop to input_size.
};

/** @brief Tensor resize policy; bilinear/bicubic use antialias and half pixels.
 * align_corners=false, antialias=true; no PIL bitwise-equivalence claim.
 */
enum class PreprocessInterpolation : std::uint8_t {
  nearest_exact = 1,  ///< Exact integer nearest sampling.
  bilinear = 2,       ///< Antialiased bilinear photo sampling.
  bicubic = 3,        ///< Antialiased bicubic; floating overshoot is retained.
};

/** @brief Channel order expected before channel-wise normalization. */
enum class PreprocessColor : std::uint8_t {
  gray = 1,  ///< One luminance channel.
  rgb = 2,   ///< Red, green and blue channels in that order.
  bgr = 3    ///< Blue, green and red channels in that order.
};

/** @brief Explicit unit-intensity image evaluation recipe.
 * Geometry precedes conversion to unit intensities and then (pixel-mean)/std.
 * Execution dtype/device, decoding and augmentation are caller choices, not
 * inferred from weights. RGB/BGR exchange is explicit in the image pipeline;
 * alpha, ICC, EXIF and grayscale conversion are outside this contract.
 */
struct ImagePreprocessSpec {
  std::array<std::uint32_t, 2> input_size{224, 224};  ///< Output height,width.
  PreprocessResize resize = PreprocessResize::stretch;  ///< Geometry policy.
  std::uint32_t resize_shorter_side = 0;  ///< Positive only for shortest_edge.
  PreprocessInterpolation interpolation = PreprocessInterpolation::bicubic;
  ///< Sampling semantics; integer photos round/saturate before unit conversion.
  PreprocessColor color = PreprocessColor::rgb;  ///< One or three channels.
  std::vector<double> mean{0, 0, 0};  ///< Finite means in output channel order.
  std::vector<double> std{1, 1, 1};   ///< Finite strictly positive deviations.
};

/** @brief Resolved resize and crop rectangle, without allocating image data. */
struct ImagePreprocessGeometry {
  std::array<std::uint32_t, 2> resized_size{};  ///< Resize height,width.
  std::uint32_t crop_top = 0;   ///< Top of the output rectangle after resize.
  std::uint32_t crop_left = 0;  ///< Left of the output rectangle after resize.
  std::array<std::uint32_t, 2> output_size{};  ///< Final height,width.
};

/** @brief One classifier output index maps to this stable category identity. */
struct ClassificationCategory {
  std::string
      id;  ///< Nonempty unique ID; order is the classifier column order.
  std::string name;  ///< Optional display name; duplicate names are permitted.
};

/** @brief Provenance and interpretation of a particular weight artifact.
 * Revisions are producer-verified immutable identifiers, not fetch commands.
 * Metadata validation cannot prove their authenticity, topology or accuracy.
 */
struct WeightInputSpec {
  std::string source_revision;  ///< Immutable revision of the weight source.
  std::string state_mapping_version;  ///< Exact producer-selected key mapping.
  std::string
      reference_uri;  ///< Source of preprocessing/category declarations.
  std::string reference_revision;  ///< Immutable version of that declaration.
  ImagePreprocessSpec image;  ///< Explicit image recipe, never model-inferred.
  std::optional<std::vector<ClassificationCategory>> categories;
  ///< Absent means no known classification map, as for embedding weights.
};

/** @brief Validates supported geometry, channels and normalization metadata.
 * @param spec Recipe to check without constructing a model or reading pixels.
 * @throws InvalidArgument For invalid enums, dimensions or mean/std.
 */
void validate_image_preprocess_spec(const ImagePreprocessSpec& spec);

/** @brief Resolves deterministic evaluation geometry using integer arithmetic.
 * @param spec Validated explicit recipe.
 * @param source_size Original height,width; both must be positive.
 * @return Resize dimensions and final crop rectangle. Shortest-edge resize
 * floors the long edge; centered crop uses nearest-even half-offset rounding.
 * @throws InvalidArgument For invalid metadata, overflow or an oversized crop.
 * @note No pixels, RNG or device operations are performed. The caller composes
 * these coordinates with data_image transforms and retains GeometryTrace.
 */
ImagePreprocessGeometry resolve_image_preprocess_geometry(
    const ImagePreprocessSpec& spec, std::array<std::uint32_t, 2> source_size);

/** @brief Validates complete provenance, image metadata and category order.
 * @param spec Producer-supplied input interpretation to check.
 * @throws InvalidArgument For missing provenance or invalid category metadata.
 * @note At most 100,000 categories and 16 MiB of category text; each nonempty
 * text field is at most 4096 bytes and excludes ASCII control characters.
 */
void validate_weight_input_spec(const WeightInputSpec& spec);

}  // namespace jyppx::liuding::models
#endif  // JYPPX_LIUDING_MODELS_PREPROCESSING_HPP
