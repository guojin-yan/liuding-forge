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
/** @brief r18_coco72 selects three decoders, trainable backbone BN and
 * half-width three-block CSP; native_v1 retains the original topology. */
enum class RtDetrProfile { native_v1, r18_coco72 };
struct RtDetrOptions {
  RtDetrBackbone backbone = RtDetrBackbone::resnet50_vd;
  std::int64_t num_classes = 80, queries = 300, decoder_layers = 6;
  std::int64_t denoising_queries = 100, max_memory_tokens = 32768;
  double label_noise = .5, box_noise = 1.;
  RtDetrProfile profile = RtDetrProfile::native_v1;
  std::int64_t max_encoder_elements = 16777216;
  ///< Per-allocation encoder element bound; explicit range 1..2^30.
  bool encoder_recomputation = false;
  ///< Rebuild selected encoder tokens after ranking during ROCm FP32 training.
  ///< CPU/NVIDIA, autocast, evaluation, no-grad and strict determinism retain
  ///< the original full encoder graph. Parameter and buffer layouts do not
  ///< change.
};
/** @brief Deep-stem vd backbone, C5 transformer, CSPRep FPN/PAN and
 * iterative multiscale deformable decoder. Input images [B,3,H,W], H/W
 * divisible by 32, targets [T,6] batch/class/xyxy canvas pixels (empty in
 * eval). Eight leaves: decoder logits [L,B,Q,C], boxes [L,B,Q,4] normalized
 * cxcywh, encoder logits/boxes [B,Q,*], denoising logits/boxes [L,B,D,*],
 * target indices [B,D] (-1 padding) and positive flags [B,D]. No NMS or RPN.
 * Native implementation has its own state layout; official weights and
 * reference/accuracy qualification are independent of architecture support.
 */
class RtDetrImpl final : public torch::nn::Module {
 public:
  explicit RtDetrImpl(RtDetrOptions options = {});
  ~RtDetrImpl() override;
  std::vector<Tensor> forward(const Tensor& images, const Tensor& targets);
  const RtDetrOptions& options() const noexcept;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
TORCH_MODULE(RtDetr);
nn::ModelAdapter adapt(RtDetr model);
}  // namespace jyppx::liuding::models::extended
#endif
