// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_DENSE_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_DENSE_HPP
/** @file dense.hpp @brief Semantic, supervised depth, SimMIM and ReID models. */
#include <jyppx/liuding/vision/models/extended/pyramid.hpp>
namespace jyppx::liuding::models::extended {
enum class DenseFieldKind { fcn, deeplabv3, depth_fpn };
struct DenseFieldOptions {
  DenseFieldKind kind=DenseFieldKind::fcn;
  std::int64_t num_classes=21;
  bool auxiliary=true;
};
/** @brief FCN/DeepLab use an output-stride-8 dilated ResNet50; depth uses
 * the explicitly named R50-FPN supervised baseline. Semantic output is the
 * fixed (main,aux) tuple when auxiliary=true, otherwise a tensor. Predictions
 * match input H/W; depth is positive and retains numerical values. */
nn::ModelAdapter create_dense_field(DenseFieldOptions options={});
enum class SimMimEncoder { swin_tiny, vit_tiny, vit_base };
/** @brief Real token masking and pixel-shuffle reconstruction. Adapter inputs
 * are image [B,3,H,W] and bool patch mask [B,H/patch,W/patch]. No CNN input
 * masking substitute, pretrained assets or implicit normalization. */
nn::ModelAdapter create_simmim(SimMimEncoder encoder=SimMimEncoder::swin_tiny,bool activation_recomputation=false);
/** @brief R50+512-dimensional projection/BN neck. Fixed tuple (identity
 * logits, pre-neck embedding); the task normalizes retrieval embeddings. */
nn::ModelAdapter create_reid_resnet50(std::int64_t identities,
                                      std::int64_t embedding_dimension=512);
}
#endif
