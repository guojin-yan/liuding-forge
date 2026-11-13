// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_VISION_MODELS_EXTENDED_VIT_ADAPTER_HPP
#define JYPPX_LIUDING_VISION_MODELS_EXTENDED_VIT_ADAPTER_HPP
/** @file vit_adapter.hpp @brief Spatial prior and interleaved ViT interactions. */
#include <jyppx/liuding/models/vit.hpp>
namespace jyppx::liuding::models::extended {
struct ViTAdapterOptions {
  ViTVariant variant=ViTVariant::tiny;
  bool activation_recomputation=false,add_vit_features=true;
  std::int64_t max_patch_tokens=4096;
};
/** @brief ViT/16 without CLS, 64-channel spatial prior, four three-block
 * interactions, zero-initialized injector gates, six-head four-point deformable
 * injection/extraction, convolution FFN ratio .25 and two extra extractors.
 * Returns C1..C4 [B,D,H/stride,W/stride], strides 4/8/16/32, H/W divisible
 * by 32. BatchNorm is local single-device normalization; distributed SyncBN
 * is not implied. Architecture implementation and upstream qualification are
 * recorded independently. */
class ViTAdapterImpl final:public torch::nn::Module {
 public:
  explicit ViTAdapterImpl(ViTAdapterOptions options={});
  ~ViTAdapterImpl() override;
  std::vector<Tensor> forward(const Tensor& images);
 private:
  struct Impl;std::shared_ptr<Impl> impl_;
};
TORCH_MODULE(ViTAdapter);
nn::ModelAdapter adapt(ViTAdapter model);
}
#endif
