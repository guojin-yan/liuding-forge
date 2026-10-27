// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_VIT_HPP
#define JYPPX_LIUDING_MODELS_VIT_HPP

/** @file vit.hpp
 * @brief Trainable timm-aligned ViT/16 Tiny, Small and Base classifiers.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>

namespace jyppx::liuding::models {

enum class ViTVariant : std::uint8_t {
  tiny = 1,            ///< 192 dimensions and three attention heads.
  small_capacity = 2,  ///< 384 dimensions and six attention heads.
  base = 3,            ///< 768 dimensions and twelve attention heads.
};

struct ViTOptions {
  std::int64_t num_classes = 1000;  ///< Zero returns the normalized CLS token.
  std::int64_t input_channels = 3;
  bool dynamic_image_size =
      false;  ///< Interpolate positions for patch-aligned images.
};

/** @brief Randomly initialized ViT/16 with twelve pre-norm blocks. */
class ViTImpl final : public torch::nn::Module {
 public:
  explicit ViTImpl(ViTVariant variant, ViTOptions options = {});

  /** @brief Returns normalized CLS and row-major patch tokens. */
  Tensor forward_features(const Tensor& input);

  /** @brief Returns normalized patch tokens as an NCHW feature grid. */
  Tensor forward_patch_grid(const Tensor& input);

  /** @brief Returns classifier logits or the normalized CLS embedding. */
  Tensor forward(const Tensor& input);

  const ViTOptions& options() const noexcept;
  ViTVariant variant() const noexcept;

  /** @brief Replaces the classifier; zero selects the CLS embedding. */
  void reset_classifier(std::int64_t num_classes);

 private:
  ViTOptions options_;
  ViTVariant variant_;
  std::int64_t dimension_ = 0;
  std::shared_ptr<torch::nn::Module> patch_embed_;
  torch::nn::Sequential blocks_{nullptr};
  torch::nn::LayerNorm norm_{nullptr};
  torch::nn::Linear head_{nullptr};
  Tensor cls_token_;
  Tensor pos_embed_;
};

TORCH_MODULE(ViT);

ViT vit_tiny_patch16_224(ViTOptions options = {});
ViT vit_small_patch16_224(ViTOptions options = {});
ViT vit_base_patch16_224(ViTOptions options = {});

nn::ModelAdapter adapt(ViT model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_VIT_HPP
