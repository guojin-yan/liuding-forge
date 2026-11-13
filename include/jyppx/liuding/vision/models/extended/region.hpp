// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_REGION_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_REGION_HPP
/** @file region.hpp @brief VGG16, SSD300 and external-proposal Fast R-CNN. */
#include <jyppx/liuding/nn/model_adapter.hpp>
namespace jyppx::liuding::models::extended {
/** @brief Classical VGG16 with adaptive 7x7 pooling and 4096/4096 classifier.
 * This allocation exceeds 500 MiB for default float32 parameters. */
nn::ModelAdapter create_vgg16(std::int64_t classes=1000);
/** @brief SSD300 VGG16, ceil-mode pool3, trainable conv4_3 L2 scale,
 * dilated fc6, six maps and 8732 ordered default boxes. Exactly 300x300.
 * Raw tuple shares the five-tensor dense detector contract; background=0. */
nn::ModelAdapter create_ssd300_vgg16(std::int64_t classes=80);
enum class FastRcnnProfile { resnet50_fpn_roi_align, vgg16_roi_pool };
struct FastRcnnOptions {
  FastRcnnProfile profile=FastRcnnProfile::resnet50_fpn_roi_align;
  std::int64_t num_classes=80,roi_batch_size=512,max_proposals=4096;
  double positive_fraction=.25;
};
/** @brief Inputs: images, externally generated proposals [R,5] batch/xyxy,
 * targets [T,6] (empty for inference), valid sizes [B,2]. Returns the fixed
 * nine-tensor region schema with empty RPN leaves. No RPN is constructed.
 * Training appends GT and draws balanced ROIs with Torch RNG; the input
 * proposal generator and cache identities belong to the dataset/run. */
nn::ModelAdapter create_fast_rcnn(FastRcnnOptions options={});
/** @brief Export bridge: dense feature and FC stages share the same state;
 * region sampling must be authored with tensor-valued proposal coordinates. */
std::vector<Tensor> fast_rcnn_features(nn::ModelAdapter&,const Tensor& images);
std::vector<Tensor> fast_rcnn_box_head(nn::ModelAdapter&,const Tensor& pooled);
FastRcnnProfile fast_rcnn_profile(nn::ModelAdapter&);
}
#endif
