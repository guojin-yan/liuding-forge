// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_RTDETR_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_RTDETR_HPP
/** @file rtdetr.hpp @brief Native RT-DETR v1 with explicit training targets. */
#include <jyppx/liuding/nn/model_adapter.hpp>
namespace jyppx::liuding::models::extended {
enum class RtDetrBackbone { resnet18_vd, resnet50_vd };
struct RtDetrOptions {
  RtDetrBackbone backbone=RtDetrBackbone::resnet50_vd;
  std::int64_t num_classes=80, queries=300, decoder_layers=6;
  std::int64_t denoising_queries=100, max_memory_tokens=32768;
  double label_noise=.5, box_noise=1.;
};
/** @brief Deep-stem vd backbone, C5 transformer, CSPRep FPN/PAN and
 * iterative multiscale deformable decoder. Input images [B,3,H,W], H/W
 * divisible by 32, targets [T,6] batch/class/xyxy canvas pixels (empty in eval).
 * Eight leaves: decoder logits [L,B,Q,C], boxes [L,B,Q,4] normalized cxcywh,
 * encoder logits/boxes [B,Q,*], denoising logits/boxes [L,B,D,*], target
 * indices [B,D] (-1 padding) and positive flags [B,D]. No NMS or RPN.
 * Native implementation has its own state layout; official weights and
 * reference/accuracy qualification are independent of architecture support.
 */
class RtDetrImpl final:public torch::nn::Module {
 public:
  explicit RtDetrImpl(RtDetrOptions options={});
  ~RtDetrImpl() override;
  std::vector<Tensor> forward(const Tensor& images,const Tensor& targets);
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
TORCH_MODULE(RtDetr);
nn::ModelAdapter adapt(RtDetr model);
}
#endif
