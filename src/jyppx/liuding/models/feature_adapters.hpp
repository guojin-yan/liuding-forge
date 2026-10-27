// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_MODELS_FEATURE_ADAPTERS_HPP
#define JYPPX_LIUDING_MODELS_FEATURE_ADAPTERS_HPP
#include <jyppx/liuding/models/convnext.hpp>
#include <jyppx/liuding/models/densenet.hpp>
#include <jyppx/liuding/models/dino.hpp>
#include <jyppx/liuding/models/dinov3_convnext.hpp>
#include <jyppx/liuding/models/efficientnet.hpp>
#include <jyppx/liuding/models/efficientnetv2.hpp>
#include <jyppx/liuding/models/features.hpp>
#include <jyppx/liuding/models/mobilenetv2.hpp>
#include <jyppx/liuding/models/swin.hpp>
#include <jyppx/liuding/models/vision.hpp>
#include <jyppx/liuding/models/vit.hpp>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace jyppx::liuding::models::detail {
std::vector<FeatureInfo> TokenFeatureInfo(std::int64_t channels,
                                          std::int64_t patch,
                                          std::int64_t prefix);
std::vector<FeatureInfo> StageFeatureInfo(
    std::initializer_list<std::int64_t> channels,
    std::int64_t first_stride = 4);
FeatureSpec ViTFeatureSpec(std::int64_t channels, bool dynamic);
FeatureSpec SwinFeatureSpec(bool dynamic);
FeatureSpec MobileFeatureSpec(double width, bool large);
nn::ModelAdapter AdaptFeatures(ResNet model);
nn::ModelAdapter AdaptFeatures(DenseNet model);
nn::ModelAdapter AdaptFeatures(MobileNetV2 model);
nn::ModelAdapter AdaptFeatures(MobileNetV3 model);
nn::ModelAdapter AdaptFeatures(EfficientNet model);
nn::ModelAdapter AdaptFeatures(EfficientNetV2 model);
nn::ModelAdapter AdaptFeatures(ConvNeXt model);
nn::ModelAdapter AdaptFeatures(SwinTiny model);
nn::ModelAdapter AdaptFeatures(ViT model);
nn::ModelAdapter AdaptFeatures(DinoV2 model);
nn::ModelAdapter AdaptFeatures(DinoV3 model);
nn::ModelAdapter AdaptFeatures(DinoV3ConvNeXt model);
}  // namespace jyppx::liuding::models::detail
#endif
