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

/** @brief Fixed embedding dimensions for twelve-block ViT/16 backbones. */
enum class ViTVariant : std::uint8_t {
  tiny = 1,            ///< 192 dimensions and three attention heads.
  small_capacity = 2,  ///< 384 dimensions and six attention heads.
  base = 3,            ///< 768 dimensions and twelve attention heads.
};

/** @brief Construction settings; pretrained weights are loaded separately. */
struct ViTOptions {
  std::int64_t num_classes = 1000;  ///< Classes in [0, 1000000]; zero embeds.
  std::int64_t input_channels = 3;  ///< Input channels in [1, 1024].
  bool dynamic_image_size =
      false;  ///< Interpolate positions for patch-aligned images.
};

/** @brief Randomly initialized ViT/16 with twelve pre-norm blocks.
 * @note Inputs are NCHW floating tensors on the parameter device and dtype.
 * Default H/W is exactly 224; dynamic mode accepts positive multiples of 16
 * and interpolates the 14x14 patch positional embedding with antialiasing.
 * Callers synchronize training, mode/head changes and concurrent state access.
 */
class ViTImpl final : public torch::nn::Module {
 public:
  /** @brief Constructs and randomly initializes the selected capacity.
   * @param variant Tiny, Small or Base architecture.
   * @param options Input channels, classifier width and spatial-size policy.
   * @throws InvalidArgument Unsupported capacity or settings.
   * @throws ResourceExhausted Insufficient model allocation memory.
   * @throws BackendError Underlying LibTorch construction failure.
   */
  explicit ViTImpl(ViTVariant variant, ViTOptions options = {});

  /** @brief Returns normalized CLS and row-major patch tokens with autograd.
   * @param input Image batch satisfying this class's input contract.
   * @return [N, 1 + (H/16)*(W/16), D], with CLS first and capacity width D.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  Tensor forward_features(const Tensor& input);

  /** @brief Returns normalized patch tokens as an NCHW feature grid.
   * @param input Image batch satisfying the forward_features() contract.
   * @return [N, D, H/16, W/16], without the CLS token and with normal autograd.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  Tensor forward_patch_grid(const Tensor& input);

  /** @brief Returns classifier logits without softmax or the CLS embedding.
   * @param input Image batch satisfying the forward_features() contract.
   * @return [N, num_classes], or [N, D] when num_classes is zero.
   * @throws InvalidArgument Input shape, device or dtype mismatch.
   * @throws ResourceExhausted Insufficient forward computation memory.
   * @throws BackendError Underlying LibTorch computation failure.
   */
  Tensor forward(const Tensor& input);

  /** @brief Queries settings, including the current classifier width.
   * @return Borrowed reference valid no longer than this model.
   */
  const ViTOptions& options() const noexcept;
  /** @brief Queries the fixed architecture.
   * @return Capacity selected during construction.
   */
  ViTVariant variant() const noexcept;

  /** @brief Replaces the classifier; zero selects the CLS embedding.
   * @param num_classes Classes in [0, 1000000].
   * @note New weights are random and retain device, dtype and mode. Rebuild
   * optimizers for the new parameters; use the unified reset_classifier API
   * to refresh an existing adapter's output schema.
   * @throws InvalidArgument Class count outside the supported range.
   * @throws ResourceExhausted Insufficient new-head allocation memory.
   * @throws BackendError Underlying LibTorch replacement failure.
   */
  void reset_classifier(std::int64_t num_classes);

 private:
  ViTOptions options_;  ///< Current input, head and spatial-size settings.
  ViTVariant variant_;  ///< Fixed backbone capacity.
  std::int64_t dimension_ = 0;                      ///< Selected token width D.
  std::shared_ptr<torch::nn::Module> patch_embed_;  ///< Registered patch stem.
  torch::nn::Sequential blocks_{
      nullptr};                         ///< Twelve registered encoder blocks.
  torch::nn::LayerNorm norm_{nullptr};  ///< Final token normalization.
  torch::nn::Linear head_{nullptr};     ///< Optional registered classifier.
  Tensor cls_token_;                    ///< Registered trainable CLS token.
  Tensor pos_embed_;  ///< Registered CLS plus 14x14 patch positions.
};

/** @brief Shared owning LibTorch holder for a ViT/16 module. */
TORCH_MODULE(ViT);

/** @brief Creates a randomly initialized ViT Tiny/16.
 * @param options Construction settings with the ViTImpl constraints.
 * @return Shared owning holder; no pretrained weights or preprocessing.
 */
ViT vit_tiny_patch16_224(ViTOptions options = {});
/** @brief Creates a randomly initialized ViT Small/16.
 * @param options Construction settings with the ViTImpl constraints.
 * @return Shared owning holder; no pretrained weights or preprocessing.
 */
ViT vit_small_patch16_224(ViTOptions options = {});
/** @brief Creates a randomly initialized ViT Base/16.
 * @param options Construction settings with the ViTImpl constraints.
 * @return Shared owning holder; no pretrained weights or preprocessing.
 */
ViT vit_base_patch16_224(ViTOptions options = {});

/** @brief Creates a single-Tensor adapter sharing the supplied module.
 * @param model Nonempty holder; parameters remain shared with callers.
 * @return Model entry with input-channel and output-width constraints.
 * @throws InvalidArgument The supplied holder is empty.
 */
nn::ModelAdapter adapt(ViT model);

}  // namespace jyppx::liuding::models

#endif  // JYPPX_LIUDING_MODELS_VIT_HPP
