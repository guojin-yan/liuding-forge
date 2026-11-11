// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.
#ifndef JYPPX_LIUDING_EXAMPLE_SEGMENTATION_MODEL_TINY_UNET_HPP
#define JYPPX_LIUDING_EXAMPLE_SEGMENTATION_MODEL_TINY_UNET_HPP

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>

namespace jyppx::liuding::examples::segmentation {
class TinyUNet final : public torch::nn::Module {
 public:
  TinyUNet();
  torch::Tensor forward(const torch::Tensor& images);

 private:
  torch::nn::Sequential e1, e2, middle, d2, d1;
  torch::nn::Conv2d head;
};
}  // namespace jyppx::liuding::examples::segmentation
#endif
