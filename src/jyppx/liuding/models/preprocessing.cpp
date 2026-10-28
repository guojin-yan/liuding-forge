// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#include <jyppx/liuding/foundation/error.hpp>
#include <jyppx/liuding/models/preprocessing.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <unordered_set>

namespace jyppx::liuding::models {
namespace {
[[noreturn]] void Invalid(const char* operation, const char* expected) {
  throw InvalidArgument(
      "Invalid weight input contract",
      {.component = "models.preprocessing",
       .operation = operation,
       .expected = expected,
       .hint = "Use explicit weight-specific input metadata."});
}

bool Text(std::string_view value) {
  return !value.empty() && value.size() <= 4096U &&
         std::none_of(value.begin(), value.end(),
                      [](unsigned char c) { return c < 0x20U || c == 0x7fU; });
}

std::uint32_t Center(std::uint32_t difference) {
  const auto floor = difference / 2U;
  return floor + ((difference % 2U != 0 && floor % 2U != 0) ? 1U : 0U);
}
}  // namespace

void validate_image_preprocess_spec(const ImagePreprocessSpec& spec) {
  if (spec.input_size[0] == 0 || spec.input_size[1] == 0)
    Invalid("preprocess.input_size", "positive output height and width");
  if (spec.resize == PreprocessResize::stretch) {
    if (spec.resize_shorter_side != 0)
      Invalid("preprocess.resize", "stretch without a shortest-edge length");
  } else if (spec.resize == PreprocessResize::shortest_edge_center_crop) {
    if (spec.resize_shorter_side <
        std::min(spec.input_size[0], spec.input_size[1]))
      Invalid("preprocess.resize", "a shortest edge large enough for the crop");
  } else {
    Invalid("preprocess.resize", "a declared deterministic resize policy");
  }
  if (spec.interpolation != PreprocessInterpolation::nearest_exact &&
      spec.interpolation != PreprocessInterpolation::bilinear &&
      spec.interpolation != PreprocessInterpolation::bicubic)
    Invalid("preprocess.interpolation", "nearest_exact, bilinear or bicubic");
  std::size_t channels = 0;
  if (spec.color == PreprocessColor::gray)
    channels = 1;
  else if (spec.color == PreprocessColor::rgb ||
           spec.color == PreprocessColor::bgr)
    channels = 3;
  else
    Invalid("preprocess.color", "gray, RGB or BGR without alpha");
  if (spec.mean.size() != channels || spec.std.size() != channels)
    Invalid("preprocess.normalization", "one mean/std per output channel");
  for (std::size_t i = 0; i < channels; ++i)
    if (!std::isfinite(spec.mean[i]) || !std::isfinite(spec.std[i]) ||
        spec.std[i] <= 0)
      Invalid("preprocess.normalization",
              "finite mean and positive finite std");
}

ImagePreprocessGeometry resolve_image_preprocess_geometry(
    const ImagePreprocessSpec& spec, std::array<std::uint32_t, 2> source_size) {
  validate_image_preprocess_spec(spec);
  if (source_size[0] == 0 || source_size[1] == 0)
    Invalid("preprocess.source_size", "positive source height and width");
  ImagePreprocessGeometry result;
  result.output_size = spec.input_size;
  if (spec.resize == PreprocessResize::stretch) {
    result.resized_size = spec.input_size;
    return result;
  }
  const auto shorter = std::min(source_size[0], source_size[1]);
  for (std::size_t axis = 0; axis < 2; ++axis) {
    const auto extent = static_cast<std::uint64_t>(source_size[axis]) *
                        spec.resize_shorter_side / shorter;
    if (extent > std::numeric_limits<std::uint32_t>::max() ||
        extent < spec.input_size[axis])
      Invalid("preprocess.geometry",
              "representable resize containing the crop");
    result.resized_size[axis] = static_cast<std::uint32_t>(extent);
  }
  result.crop_top = Center(result.resized_size[0] - result.output_size[0]);
  result.crop_left = Center(result.resized_size[1] - result.output_size[1]);
  return result;
}

void validate_weight_input_spec(const WeightInputSpec& spec) {
  if (!Text(spec.source_revision) || !Text(spec.state_mapping_version) ||
      !Text(spec.reference_uri) || !Text(spec.reference_revision))
    Invalid("weights.input.provenance",
            "complete bounded source and mapping revisions");
  validate_image_preprocess_spec(spec.image);
  if (!spec.categories) return;
  const auto& categories = *spec.categories;
  if (categories.empty() || categories.size() > 100000U)
    Invalid("weights.input.categories", "one to 100000 ordered categories");
  std::size_t bytes = 0;
  for (const auto& item : categories) {
    if (!Text(item.id) || (!item.name.empty() && !Text(item.name)))
      Invalid("weights.input.categories",
              "bounded category IDs and optional names");
    bytes += item.id.size() + item.name.size();
    if (bytes > (16U << 20U))
      Invalid("weights.input.categories", "at most 16MiB of category text");
  }
  std::unordered_set<std::string_view> ids;
  ids.reserve(categories.size());
  for (const auto& item : categories)
    if (!ids.insert(item.id).second)
      Invalid("weights.input.categories",
              "unique category IDs in output order");
}
}  // namespace jyppx::liuding::models
