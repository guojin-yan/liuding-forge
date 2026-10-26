// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_SWIN_HPP
#define JYPPX_LIUDING_MODELS_SWIN_HPP

/** @file swin.hpp
 * @brief Trainable timm-aligned Swin-T patch4/window7 image classifier.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/container/modulelist.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace jyppx::liuding::models {

struct SwinOptions {
  std::int64_t num_classes = 1000;  ///< Zero returns the pooled embedding.
  std::int64_t input_channels = 3;
  bool dynamic_image_size = false;  ///< Accept patch-aligned images.
};

class SwinPatchEmbed;
class SwinHead;

class SwinTinyImpl final : public torch::nn::Module {
 public:
  explicit SwinTinyImpl(SwinOptions options = {});

  /** @brief Four NCHW stages with reductions 4, 8, 16 and 32. */
  std::vector<Tensor> forward_features(const Tensor& input);
  /** @brief Classification logits or the 768-wide pooled embedding. */
  Tensor forward(const Tensor& input);
  void reset_classifier(std::int64_t num_classes);
  const SwinOptions& options() const noexcept;

 private:
  SwinOptions options_;
  std::shared_ptr<SwinPatchEmbed> patch_embed_;
  torch::nn::ModuleList layers_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  std::shared_ptr<SwinHead> head_;
};

TORCH_MODULE(SwinTiny);

SwinTiny swin_tiny_patch4_window7_224(SwinOptions options = {});
nn::ModelAdapter adapt(SwinTiny model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_SWIN_HPP
