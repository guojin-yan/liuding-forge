// Copyright 2026 JYPPX Training Framework Contributors
// Product: 六丁炼算 (Liuding Forge)
// Project alias: JYPPX Training Framework
// Licensed under the Apache License, Version 2.0.

#ifndef JYPPX_LIUDING_MODELS_DINO_HPP
#define JYPPX_LIUDING_MODELS_DINO_HPP
/** @file dino.hpp
 * @brief DINOv2 and the first DINOv3 ViT/16 image backbones.
 *
 * The public contract exposes the four DINOv2 capacities selected for v1.0.
 * Architectures are randomly initialized; pretrained weights remain a
 * separately qualified `WeightRecord` and are never downloaded implicitly.
 */

#include <jyppx/liuding/nn/model_adapter.hpp>

#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>
#include <torch/nn/pimpl.h>

#include <cstdint>
#include <memory>

namespace jyppx::liuding::models {

/** @brief Fixed DINOv2 ViT/14 capacities in the v1.0 model matrix. */
enum class DinoV2Variant : std::uint8_t {
  small_capacity = 1,  ///< 384-dimensional, 12-block, 6-head encoder.
  base = 2,            ///< 768-dimensional, 12-block, 12-head encoder.
  large = 3,           ///< 1024-dimensional, 24-block, 16-head encoder.
  giant = 4,           ///< 1536-dimensional, 40-block, 24-head SwiGLU encoder.
};

/** @brief Input, head and initialization settings shared by DINOv2 models. */
struct DinoV2Options {
  std::int64_t num_classes = 0;
  ///< Zero returns the CLS embedding; positive values add a classifier head.
  std::int64_t input_channels = 3;  ///< Positive input channel count.
  double layer_scale_init = 1e-5;   ///< Positive LayerScale initialization.
};

/** @brief Trainable DINOv2 ViT/14 image encoder with an optional head. */
class DinoV2Impl final : public torch::nn::Module {
 public:
  /** @brief Constructs one fixed-capacity randomly initialized encoder. */
  explicit DinoV2Impl(DinoV2Variant variant, DinoV2Options options = {});

  /**
   * @brief Returns CLS plus row-major patch tokens.
   * @param input Dense floating NCHW input on the module device and dtype.
   * @return `[N, 1 + floor(H/14)*floor(W/14), embedding]` tokens.
   * @note H and W must be positive multiples of 14. Positional embeddings are
   * bicubically interpolated for grids other than the 518x518 reference grid.
   */
  Tensor forward_features(const Tensor& input);

  /**
   * @brief Returns CLS embeddings or classifier logits.
   * @param input Dense floating NCHW input matching the module contract.
   * @return `[N, embedding]` when `num_classes == 0`, otherwise logits.
   */
  Tensor forward(const Tensor& input);

  /** @brief Returns immutable constructor options. */
  const DinoV2Options& options() const noexcept;

  /** @brief Returns the fixed variant selected at construction. */
  DinoV2Variant variant() const noexcept;

  /**
   * @brief Replaces or removes the linear classification head.
   * @param num_classes Zero selects the CLS embedding; positive values create
   * a freshly initialized classifier on the encoder's current device/dtype.
   * @note Create a new ModelAdapter after changing the head so its output
   * schema matches the updated module.
   */
  void reset_classifier(std::int64_t num_classes);

 private:
  DinoV2Options options_;
  DinoV2Variant variant_;
  std::int64_t embedding_dimension_ = 0;
  std::int64_t attention_heads_ = 0;
  std::int64_t block_count_ = 0;
  std::int64_t hidden_dimension_ = 0;
  bool swiglu_ = false;
  // Kept opaque so the implementation can expose the timm-compatible
  // `patch_embed.proj` state hierarchy without adding helper modules to the
  // public model API.
  std::shared_ptr<torch::nn::Module> patch_embed_;
  torch::nn::LayerNorm final_norm_{nullptr};
  torch::nn::Linear head_{nullptr};
  Tensor cls_token_;
  Tensor position_embedding_;
  torch::nn::Sequential blocks_{nullptr};
};

/** @brief Shared owning holder for a DINOv2 encoder. */
TORCH_MODULE(DinoV2);

/** @brief Constructs DINOv2 ViT-S/14. */
DinoV2 dinov2_vits14(DinoV2Options options = {});
/** @brief Constructs DINOv2 ViT-B/14. */
DinoV2 dinov2_vitb14(DinoV2Options options = {});
/** @brief Constructs DINOv2 ViT-L/14. */
DinoV2 dinov2_vitl14(DinoV2Options options = {});
/** @brief Constructs DINOv2 ViT-G/14. */
DinoV2 dinov2_vit_giant14(DinoV2Options options = {});

/** @brief Creates a schema-checked adapter for a DINOv2 holder. */
nn::ModelAdapter adapt(DinoV2 model);

/** @brief Low-resource DINOv3 ViT/16 capacities selected for v1.0. */
enum class DinoV3Variant : std::uint8_t {
  small_capacity = 1,  ///< 384-dimensional, 12-block, 6-head encoder.
  small_plus = 2,      ///< 384-dimensional SwiGLU, 12-block, 6-head encoder.
  base = 3,            ///< 768-dimensional, 12-block, 12-head encoder.
  large = 4,           ///< 1024-dimensional, 24-block, 16-head encoder.
  large_plus = 5,      ///< 1024-dimensional SwiGLU, 24-block, 16-head encoder.
  huge_plus = 6,       ///< 1280-dimensional SwiGLU, 32-block, 20-head encoder.
  seven_b = 7,         ///< 4096-dimensional SwiGLU, 40-block, 32-head encoder.
};

/** @brief Input, head and initialization settings shared by DINOv3 ViT models. */
struct DinoV3Options {
  std::int64_t num_classes = 0;
  ///< Zero returns the CLS embedding; positive values add a classifier head.
  std::int64_t input_channels = 3;  ///< Positive input channel count.
  std::int64_t extra_tokens = 4;    ///< Fixed DINOv3 storage/register tokens.
  double layer_scale_init = 1e-5;   ///< Positive LayerScale initialization.
};

/** @brief DINOv3 ViT/16 encoder with CLS, extra and patch tokens. */
class DinoV3Impl final : public torch::nn::Module {
 public:
  /** @brief Constructs a low-resource randomly initialized DINOv3 encoder. */
  explicit DinoV3Impl(DinoV3Variant variant, DinoV3Options options = {});

  /**
   * @brief Returns CLS, extra/storage and row-major patch tokens.
   * @param input Dense floating NCHW input on the module device and dtype.
   * @return `[N, 1 + extra_tokens + floor(H/16)*floor(W/16), embedding]`.
   * @note H and W are cropped to their largest positive multiples of 16.
   */
  Tensor forward_features(const Tensor& input);

  /** @brief Returns CLS embeddings or classifier logits. */
  Tensor forward(const Tensor& input);

  /** @brief Returns immutable constructor options. */
  const DinoV3Options& options() const noexcept;

  /** @brief Returns the fixed variant selected at construction. */
  DinoV3Variant variant() const noexcept;

  /** @brief Replaces or removes the linear classification head. */
  void reset_classifier(std::int64_t num_classes);

 private:
  DinoV3Options options_;
  DinoV3Variant variant_;
  std::int64_t embedding_dimension_ = 0;
  std::int64_t attention_heads_ = 0;
  std::int64_t block_count_ = 0;
  std::int64_t hidden_dimension_ = 0;
  bool swiglu_ = false;
  std::shared_ptr<torch::nn::Module> patch_embed_;
  std::shared_ptr<torch::nn::Module> rope_embed_;
  torch::nn::LayerNorm final_norm_{nullptr};
  torch::nn::Linear head_{nullptr};
  Tensor cls_token_;
  Tensor storage_tokens_;
  Tensor mask_token_;
  torch::nn::Sequential blocks_{nullptr};
};

/** @brief Shared owning holder for a DINOv3 encoder. */
TORCH_MODULE(DinoV3);

/** @brief Constructs DINOv3 ViT-S/16. */
DinoV3 dinov3_vits16(DinoV3Options options = {});
/** @brief Constructs DINOv3 ViT-S+/16. */
DinoV3 dinov3_vits16plus(DinoV3Options options = {});
/** @brief Constructs DINOv3 ViT-B/16. */
DinoV3 dinov3_vitb16(DinoV3Options options = {});
/** @brief Constructs DINOv3 ViT-L/16. */
DinoV3 dinov3_vitl16(DinoV3Options options = {});
/** @brief Constructs DINOv3 ViT-L+/16. */
DinoV3 dinov3_vitl16plus(DinoV3Options options = {});
/** @brief Constructs DINOv3 ViT-H+/16. */
DinoV3 dinov3_vith16plus(DinoV3Options options = {});
/** @brief Constructs DINOv3 ViT-7B/16. */
DinoV3 dinov3_vit7b16(DinoV3Options options = {});

/** @brief Creates a schema-checked adapter for a DINOv3 holder. */
nn::ModelAdapter adapt(DinoV3 model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_DINO_HPP
