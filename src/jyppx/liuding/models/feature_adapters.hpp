// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_FEATURE_ADAPTERS_HPP
#define JYPPX_LIUDING_MODELS_FEATURE_ADAPTERS_HPP
#include <jyppx/liuding/models/dino.hpp>
#include <jyppx/liuding/models/dinov3_convnext.hpp>
#include <jyppx/liuding/models/features.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace jyppx::liuding::models::detail {
std::vector<FeatureInfo> TokenFeatureInfo(std::int64_t channels,
                                          std::int64_t patch,
                                          std::int64_t prefix);
std::vector<FeatureInfo> StageFeatureInfo(
    const std::array<std::int64_t, 4>& channels);
nn::ModelAdapter AdaptFeatures(DinoV2 model);
nn::ModelAdapter AdaptFeatures(DinoV3 model);
nn::ModelAdapter AdaptFeatures(DinoV3ConvNeXt model);
}  // namespace jyppx::liuding::models::detail
#endif
